from transformers import T5Config, T5ForConditionalGeneration
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import os
from models.genrec.TIGER.rqvae import RQVAE
from models.genrec.tokenizer import AbstractTokenizer
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import numpy as np
from tqdm import tqdm

class TIGERTokenizer(AbstractTokenizer):
    def __init__(self, config: dict, semantic_ids=None):
        super().__init__(config)
        self.rqvae_config = config['RQ-VAE']
        self.model_path = f"results/RQVAE/{config['dataset']}{config['exp_id']}.pt"
        self.item_2_semantic_id = (
            {i: list(semantic_ids[i, :]) for i in range(len(semantic_ids))}
            if semantic_ids is not None
            else None
        )

    @property
    def n_digits(self):
        return self.rqvae_config['num_layers'] + 1

    @property
    def code_book_size(self):
        return self.rqvae_config['code_book_size']

    @property
    def max_inputs_length(self):
        return self.config['max_history_len'] * self.n_digits + 2

    @property
    def pad_token_id(self):
        return 0
     
    @property
    def bos_token_id(self):
        return self.n_digits * self.code_book_size + 1

    @property
    def eos_token_id(self):
        return self.bos_token_id + 1

    @property
    def decoder_start_token_id(self):
        return 0

    @property
    def vocab_size(self):
        return self.eos_token_id + 1
    
    def _init_rqvae(self, device):
        return RQVAE(
            hidden_sizes=[self.rqvae_config['sent_emb_dim']] + self.rqvae_config['hidden_dim'],
            n_codebooks=self.rqvae_config['num_layers'],
            codebook_size=self.rqvae_config['code_book_size'],
            dropout=self.rqvae_config['dropout'],
            low_usage_threshold=self.rqvae_config['rqvae_low_usage_threshold']
        ).to(device)
        
    def fit(self, embeddings_train, device):
        assert self.item_2_semantic_id is None, "Item_2_semantic_id mapping found, no need to retrain the Tokenizer."
        
        model = self._init_rqvae(device)
        model.generate_codebook(embeddings_train, device)
        
        optimizer = torch.optim.Adagrad(model.parameters(), lr=self.rqvae_config['lr'])
        dataloader = DataLoader(TensorDataset(embeddings_train), batch_size=self.rqvae_config['batch_size'], shuffle=True)

        model.train()
        for _ in tqdm(range(self.rqvae_config['epochs'])):
            for batch in dataloader:
                x_batch = batch[0]
                optimizer.zero_grad()
                recon_x, quant_loss, _ = model(x_batch)
                loss = F.mse_loss(recon_x, x_batch, reduction='mean') + self.rqvae_config['beta'] * quant_loss
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), self.model_path, pickle_protocol=4)
        print("Training complete.")

    def transform(self, embeddings, device):
        embeddings = torch.Tensor(embeddings).to(device)
        model = self._init_rqvae(device)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        semantic_ids = model.encode(embeddings)
        semantic_id_2_item = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        semantic_ids_full = []

        for i in range(len(semantic_ids)):
            id = semantic_ids[i]
            id_dict = semantic_id_2_item[id[0]][id[1]][id[2]]
            id_dict[len(id_dict)] = i+1
            semantic_ids_full.append(list(id) + [len(id_dict)])
            
        semantic_ids_full = np.array(semantic_ids_full)
        semantic_ids_with_offset = semantic_ids_full + (np.arange(self.n_digits) * self.code_book_size + 1).reshape(1,-1)
        
        return semantic_ids_with_offset

    def tokenize(self, input_sequence, item_id):
        
        input_ids = [self.bos_token_id]

        for i in range(len(input_sequence)):
            input_ids.extend(self.item_2_semantic_id[input_sequence[i]])

        input_ids = input_ids + [self.eos_token_id]

        labels = self.item_2_semantic_id[item_id]
        labels = np.array(labels + [self.eos_token_id])

        input_id_length = len(input_ids)
        input_ids = np.array(input_ids + [self.pad_token_id]*(self.max_inputs_length - input_id_length))
        attention_mask = (np.arange(self.max_inputs_length) < input_id_length).astype(int)

        assert not np.any(labels == self.pad_token_id), labels # No padding in labels
        labels[labels == self.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels' : labels
        }