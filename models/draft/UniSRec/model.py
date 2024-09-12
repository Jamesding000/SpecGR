import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
from models.draft.UniSRec.layers import SASRec, MoEAdaptorLayer
from models.draft.drafter import AbstractDrafter

"""
Code adapted from https://github.com/RUCAIBox/UniSRec/blob/master/unisrec.py
Original Author: Yupeng Hou
"""

class UniSRec(SASRec):
    def __init__(self, config, item_embeddings):
        
        config['n_items'] = item_embeddings.shape[0] - 1
        super(UniSRec, self).__init__(config)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']
        self.n_items = config['n_items']

        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None
            # for `transductive_ft`, `item_embedding` is defined in SASRec base model
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            # `plm_embedding` in pre-train stage will be carried via dataloader
            # assert item_embeddings.shape[0] == self.n_items
            self.plm_embedding = self.load_plm_embedding(item_embeddings)

        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob']
        )

    def load_plm_embedding(self, item_embeddings):    
        plm_embedding = nn.Embedding(item_embeddings.shape[0], item_embeddings.shape[1], padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(item_embeddings)
        return plm_embedding

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, input_ids, length, labels):
        item_seq, item_seq_len, labels = input_ids, length, labels
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_item_emb = self.moe_adaptor(self.plm_embedding.weight)

        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(test_item_emb, dim=1)

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        loss = self.loss_fct(logits, labels)
        return loss

    def full_sort_predict(self, item_seq, item_seq_len):
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = self.moe_adaptor(self.plm_embedding.weight)

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

    def predict(self, item_seq, item_seq_len, item_embeddings, test_item_embeddings):  # can pass in both in-sample items and out-of-sample items
        item_emb_list = self.moe_adaptor(item_embeddings[item_seq])
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = self.moe_adaptor(test_item_embeddings)  

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
    
    def seq_seq_contrastive_task(self, seq_output, same_pos_id, item_seq_aug, item_seq_len_aug):
        # seq_output: [B, H], same_pos_id: [B, B]
        # item_seq_aug: [B, L, H], item_seq_len_aug: [B, ]
        item_emb_list_aug = self.moe_adaptor(self.plm_embedding(item_seq_aug))
        seq_output_aug = self.forward(item_seq_aug, item_emb_list_aug, item_seq_len_aug)
        seq_output_aug = F.normalize(seq_output_aug, dim=1)

        # [B, H] * [B, H] -> [B, H]
        pos_logits = (seq_output * seq_output_aug).sum(dim=1) / self.temperature # [B, ]
        pos_logits = torch.exp(pos_logits)

        # [B, H] @ [H, B] -> [B, B]
        neg_logits = (
            torch.matmul(seq_output, seq_output_aug.transpose(0, 1)) / self.temperature
        )
        neg_logits = torch.where(
            same_pos_id,
            torch.tensor([0], dtype=torch.float, device=same_pos_id.device),
            neg_logits,
        ) # [B, B]
        neg_logits = torch.exp(neg_logits).sum(dim=1) # [B, ]

        loss = -torch.log(pos_logits / neg_logits) # [B, ]
        return loss.mean()

    def pretrain(self, item_seq, item_seq_len, labels):
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        # item_emb_list = self.moe_adaptor(interaction["item_emb_list"])
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)

        pos_id = labels
        
        # Remove sequences with the same next item
        # same_pos_id = pos_id.unsqueeze(1) == pos_id.unsqueeze(0)
        # same_pos_id = torch.logical_xor(
        #     same_pos_id,
        #     torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device),
        # )

        # Simply take every other in-batch sequence as negative sequence
        same_pos_id = torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device)

        loss = self.seq_seq_contrastive_task(
            seq_output, same_pos_id, item_seq, item_seq_len
        )
        
        return loss

class UniSRecDrafter(AbstractDrafter):

    def __init__(self, unisrec: UniSRec):
        super().__init__()
        self.unisrec = unisrec
        self.unisrec.eval()

    def score(self, item_seq, item_seq_len, item_embeddings, constraints=None, **kwargs):
        test_item_embeddings = item_embeddings[constraints] if constraints is not None else item_embeddings
        return self.unisrec.predict(item_seq, item_seq_len, item_embeddings, test_item_embeddings)