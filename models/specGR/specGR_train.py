import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Optional, Tuple, List, Any
from models.genrec.TIGER.model import TIGER
from models.genrec.TIGER.tokenizer import TIGERTokenizer

class SpecGR(nn.Module):
    def __init__(self, genrec: TIGER, config: Dict[str, Any]):
        super().__init__()
        self.genrec = genrec
        self.genrec_config = self.genrec.config
        self.hidden_size = self.genrec.config['d_model']

        self.temperature = config["temperature"]
        self.encoder_batch_size = config["encoder_batch_size"]

        self.emb_loss_fct = nn.CrossEntropyLoss()
        self.ft_loss_fct = nn.CrossEntropyLoss()

        projection_dim = config.get("projection", self.hidden_size)
        self.projection = (
            torch.nn.Linear(self.hidden_size, int(projection_dim))
            if projection_dim else None
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, normalize: bool = True) -> torch.Tensor:
        batch_size = self.encoder_batch_size
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.long).to(input_ids.device)

        all_embeddings = []

        for start_index in range(0, len(input_ids), batch_size):
            mini_batch_input_ids = input_ids[start_index : start_index + batch_size]
            mini_batch_attention_mask = attention_mask[start_index : start_index + batch_size]

            encoder_outputs = self.genrec.encoder_forward(input_ids=mini_batch_input_ids, attention_mask=mini_batch_attention_mask)
            embeddings = self.encoder_outputs_to_embedding(encoder_outputs, mini_batch_attention_mask, normalize=normalize)
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def encoder_outputs_to_embedding(self, encoder_outputs: Dict[str, torch.Tensor], attention_mask: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        output_hidden_state = encoder_outputs["last_hidden_state"]

        if self.projection:
            output_hidden_state = self.projection(output_hidden_state)

        embeddings = self.mean_pooling(output_hidden_state, attention_mask)

        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def mean_pooling(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor, recast: bool = False) -> torch.Tensor:
        s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        embedding = s / d

        return embedding.to(hidden_state.dtype) if recast else embedding

    def gather_distributed_tensors(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        gathered_tensors = []
        for tensor in tensors:
            tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list, tensor.contiguous())
            tensor_list[dist.get_rank()] = tensor
            gathered_tensors.append(torch.cat(tensor_list, dim=0))
        return tuple(gathered_tensors)

    def seq_item_contrastive_task(self, seq_output: torch.Tensor, pos_items_emb: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        same_pos_id = (item_ids.unsqueeze(1) == item_ids.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(item_ids.shape[0], dtype=torch.bool, device=item_ids.device))

        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()
    
    def calculate_emb_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        seq_output = self.encode(input_ids, attention_mask)
        labels = torch.cat([torch.full((labels.shape[0], 1), self.genrec.tokenizer.bos_token_id).to(labels.device), labels], dim=1)
        pos_items_emb = self.encode(labels)

        if dist.is_initialized() and self.training:
            seq_output, pos_items_emb, item_ids = self.gather_distributed_tensors(seq_output, pos_items_emb, item_ids)

        return self.seq_item_contrastive_task(seq_output, pos_items_emb, item_ids)

    def calculate_gen_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.genrec.calculate_loss(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, test_items_emb: torch.Tensor) -> torch.Tensor:
        seq_output = self.encode(input_ids, attention_mask)
        test_items_emb = F.normalize(test_items_emb, dim=-1)
        return torch.matmul(seq_output, test_items_emb.transpose(0, 1))

    def finetune(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, train_items_emb: torch.Tensor, target_item_ids: torch.Tensor) -> torch.Tensor:
        seq_output = self.encode(input_ids, attention_mask)
        train_items_emb = F.normalize(train_items_emb, dim=-1)
        logits = torch.matmul(seq_output, train_items_emb.transpose(0, 1)) / self.temperature
        return self.ft_loss_fct(logits, target_item_ids)