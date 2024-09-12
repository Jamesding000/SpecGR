from typing import Dict, List, Optional, Tuple, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch
from torch import nn
from torch.optim import AdamW
import torch.distributed as dist
import lightning as L
from transformers import get_scheduler
import os
from models.specGR.specGR_train import SpecGR
from evaluator import SpecGRForTrainEvaluator
from SpecGR.lightning_modules.utils import calculate_optimizer_config_in_distributed_setting

class SpecGRFinetuneLightningModule(L.LightningModule):
    def __init__(
        self, 
        model: SpecGR,
        lambda_emb: float,
        lambda_gen: float,
        lr: float,
        saved_model_path: str, 
        saved_embedding_path: str,
        weight_decay: float, 
        warmup_steps: int, 
        unseen_start_index: int,
        test_start_index: int,
        semantic_ids: torch.Tensor, 
        evaluator: SpecGRForTrainEvaluator
    ) -> None:
        super().__init__()
        
        self.model = model
        self.saved_model_path = saved_model_path
        self.saved_embedding_path = saved_embedding_path
        self.lambda_emb = lambda_emb
        self.lambda_gen = lambda_gen
        self.lr = lr 
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.unseen_start_index = unseen_start_index
        self.test_start_index = test_start_index
        self.evaluator = evaluator
        
        self.semantic_id_sequences = self._prepare_semantic_id_sequences(semantic_ids)
        self.item_embeddings: Optional[nn.Embedding] = None
        
        self.best_metrics: float = 0.0
        self.best_epoch: int = 0
        
        self.train_outputs: List[Dict[str, torch.Tensor]] = []
        self.valid_outputs: List[Dict[str, torch.Tensor]] = []
        self.test_outputs: List[Dict[str, torch.Tensor]] = []

    def _prepare_semantic_id_sequences(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat((
            torch.full((semantic_ids.shape[0], 1), self.model.genrec.tokenizer.bos_token_id),
            semantic_ids,
            torch.full((semantic_ids.shape[0], 1), self.model.genrec.tokenizer.eos_token_id),
        ), dim=1).requires_grad_(False)

    def setup(self, stage: Optional[str] = None) -> None:
        self.model = self.model.to(self.device)
        self.semantic_id_sequences = self.semantic_id_sequences.to(self.device)
        self.evaluator.device = self.device
        
        print('self.item_embeddings', self.item_embeddings)
        print(self.saved_model_path, os.path.exists(self.saved_model_path))
        print(self.saved_embedding_path, os.path.exists(self.saved_embedding_path))
        
        if self.item_embeddings is None:
            self.encode_items_embeddings()
    
        if not os.path.exists(self.saved_model_path) or not os.path.exists(self.saved_embedding_path):
            print("Model Path or Embedding not found. Copying the loaded model to the saved_model_path")
            self._save_checkpoint()
        
    def on_train_epoch_start(self) -> None:
        self.train_outputs.clear()

        if self.current_epoch == 0:
            self._load_checkpoint()

        self.model.train()
        assert self.model.training

    def init_item_embeddings(self, embeddings: Optional[torch.Tensor] = None) -> None:
        embedding_dim = (
            self.model.projection.out_features
            if self.model.projection
            else self.model.genrec.config['d_model']
        )
        
        if embeddings is not None:
            self.item_embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True).to(self.device)
            print('Initialized item embeddings from vectors.')
        else:
            self.item_embeddings = nn.Embedding(
                num_embeddings=self.semantic_id_sequences.shape[0],
                embedding_dim=embedding_dim,
            ).to(self.device)
            print('Initialized empty item embeddings.')
        
        self.item_embeddings.eval()
        self.item_embeddings.requires_grad_(False)

    @torch.no_grad()
    def encode_items_embeddings(self) -> None:
        self.model.eval()
        assert not self.model.training

        item_embeddings = self.model.encode(self.semantic_id_sequences).detach()
        self.init_item_embeddings(item_embeddings)
        print("Encoded all items.")

    def _load_checkpoint(self) -> None:
        self.model.load_state_dict(torch.load(self.saved_model_path))
        print(f'Loaded model checkpoint from {self.saved_model_path}.')

        self.init_item_embeddings()
        self.item_embeddings.load_state_dict(torch.load(self.saved_embedding_path))
        print(f'Loaded item embeddings from {self.saved_embedding_path}.')

        self.item_embeddings.eval()
        self.item_embeddings.requires_grad_(False)

    def _save_checkpoint(self) -> None:
        if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
            torch.save(self.model.state_dict(), self.saved_model_path)
            print(f'Model checkpoint saved at {self.saved_model_path}.')

            torch.save(self.item_embeddings.state_dict(), self.saved_embedding_path)
            print(f'Embedding checkpoint saved at {self.saved_embedding_path}.')

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self.model.train()
        assert self.model.training
        
        batch_size = batch['input_ids'].shape[0]
        
        emb_loss = self.model.finetune(
            batch['input_ids'], 
            batch['attention_mask'], 
            self.item_embeddings.weight[:self.unseen_start_index+1], 
            batch['item_id']
        )

        gen_loss = self.model.calculate_gen_loss(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['labels'],
        )

        loss = self.lambda_emb * emb_loss + self.lambda_gen * gen_loss

        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

        self._log_training_metrics(emb_loss, gen_loss, loss, batch_size)

        return loss

    def _log_training_metrics(self, emb_loss: torch.Tensor, gen_loss: torch.Tensor, loss: torch.Tensor, batch_size: int) -> None:
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=False)
        self.log('emb_loss', emb_loss, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=False)
        self.log('gen_loss', gen_loss, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=False)
        self.log('train_loss', loss, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=False)

        avg_loss = torch.stack([x['loss'] for x in self.train_outputs]).mean() if self.train_outputs else torch.tensor(-1.0)
        self.log("avg_train_loss", avg_loss, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=False)

        self.train_outputs.append({"loss": loss, "emb_loss": emb_loss, "gen_loss": gen_loss})

    def on_train_epoch_end(self) -> None:
        avg_emb_loss = torch.stack([x['emb_loss'] for x in self.train_outputs]).mean()
        avg_gen_loss = torch.stack([x['gen_loss'] for x in self.train_outputs]).mean()
        avg_loss = torch.stack([x['loss'] for x in self.train_outputs]).mean()
        self.log("avg_emb_loss", avg_emb_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("avg_gen_loss", avg_gen_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("avg_train_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    def on_validation_epoch_start(self) -> None:
        self.valid_outputs.clear()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        batch_size = batch['input_ids'].shape[0]
        metrics = self.evaluator.evaluation_step(batch, self.device, test_item_embs=self.item_embeddings.weight[:self.test_start_index+1])
        metrics = self.evaluator.convert_metrics_to_tensor(metrics, self.device)
        self.valid_outputs.append(metrics)
        self._log_validation_metrics(metrics, batch_size)
        return metrics

    def _log_validation_metrics(self, metrics: Dict[str, torch.Tensor], batch_size: int) -> None:
        avg_recall_h = torch.stack([x['recall_h_50'] for x in self.valid_outputs]).mean()
        avg_recall_b = torch.stack([x['recall_b_50'] for x in self.valid_outputs]).mean()
        avg_valid_loss = torch.stack([x['loss_h'] for x in self.valid_outputs]).mean()
        
        self.log('valid_Recall_Emb@50', avg_recall_h, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=True)
        self.log('valid_Recall_Gen@50', avg_recall_b, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=True)
        self.log('valid_loss', avg_valid_loss, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        avg_metrics = self.evaluator.process_evaluation_result(self.valid_outputs)
        avg_metrics = self.evaluator.convert_metrics_to_tensor(avg_metrics, self.device)
        if avg_metrics["recall_h_50"] > self.best_metrics:
            self.best_metrics = avg_metrics["recall_h_50"]
            self.best_epoch = self.current_epoch
            self._save_checkpoint()

    def on_test_epoch_start(self) -> None:
        self.test_outputs.clear()
        self._load_checkpoint()
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        metrics = self.evaluator.evaluation_step(batch, self.device, test_item_embs=self.item_embeddings.weight)
        metrics = self.evaluator.convert_metrics_to_tensor(metrics, self.device)
        self.test_outputs.append(metrics)
        return metrics

    def on_test_epoch_end(self) -> None:
        avg_metrics = self.evaluator.process_evaluation_result(self.test_outputs)
        avg_metrics = self.evaluator.convert_metrics_to_tensor(avg_metrics, self.device)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Union[_LRScheduler, str, int]]]]:
        total_training_steps, total_warmup_steps, scaled_lr = calculate_optimizer_config_in_distributed_setting(
            self.trainer, self.warmup_steps, self.lr, self.weight_decay
        )

        optimizer = AdamW(self.model.parameters(), lr=scaled_lr, weight_decay=self.weight_decay)
        scheduler = get_scheduler("cosine", optimizer, num_training_steps=total_training_steps, num_warmup_steps=total_warmup_steps)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]