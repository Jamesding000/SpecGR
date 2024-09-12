# TODO: from trainers.drafter_trainer import Trainer, ...

# TODO: check data inconsistency issue

import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from SpecGR_Aux.trainer import UniSRecTrainer, TIGERTrainer
from evaluator import TIGEREvaluator, UniSRecEvaluator
from dataloader import UniSRecDataProcessor, TIGERDataProcessor, get_dataloaders
from models.genrec.TIGER.tokenizer import TIGERTokenizer
from models.genrec.TIGER.model import TIGER
from models.draft.UniSRec.model import UniSRec
from utils import load_config, get_model_ckpt_path, get_logfile_path, get_saved_id_path, load_semantic_ids, load_item_embeddings

def train_drafter(config, device, saved_draft_model_path, log_file_path):
    domain = config['dataset']
    unisrec_config = config['UniSRec']
    
    # Load embeddings
    item_embeddings = np.fromfile(f'dataset/{domain}/{domain}.sent_emb', dtype=np.float32).reshape(-1, config['RQ-VAE']['sent_emb_dim'])
    zeros_row = torch.zeros(1, item_embeddings.shape[1])
    item_embeddings = torch.cat((zeros_row, torch.from_numpy(item_embeddings)), dim=0).to(device)
    
    unseen_start_index = config['unseen_start_index']
    test_start_index = config['test_start_index']
    
    train_embeddings = item_embeddings[:unseen_start_index+1]
    valid_embeddings = item_embeddings[:test_start_index+1]
    
    # Initialize the model
    model = UniSRec(unisrec_config, item_embeddings=train_embeddings)
    
    # Instantiate the DataProcessor
    data_processor = UniSRecDataProcessor(max_length=config['max_history_len'])
    
    # Get DataLoaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        domain=domain,
        splits=['train', 'valid', 'test'],
        train_batch_size=unisrec_config['trainer']['train_batch_size'],
        eval_batch_size=unisrec_config['trainer']['eval_batch_size'],
        data_processor=data_processor,
        num_workers=config['num_workers']
    )
    
    # Initialize the evaluator and trainer
    evaluator = UniSRecEvaluator(model, ks=config['eval_ks'], item_embeddings=valid_embeddings)
    trainer = UniSRecTrainer(config, device, model, evaluator, val_item_embeddings=item_embeddings[:test_start_index+1])
    
    # Train the model
    trainer.fit(train_dataloader, val_dataloader)
    
    # Evaluate the model
    evaluator.evaluate(test_dataloader, device=device)

def train_genrec(config, device, saved_id_path, saved_target_model_path, log_file_path):
    domain = config['dataset']
    exp_id = config['exp_id']
    tiger_config = config['TIGER']
    
    print('saved_id_path', saved_id_path)
    
    if not os.path.exists(saved_id_path):
        print('Generating and saving semantic IDs...')
        embeddings = load_item_embeddings(config)[1:, :] # remove the padding row during training
        unseen_start_index = config['unseen_start_index']
        embeddings_in_sample = embeddings[:unseen_start_index]
        
        embeddings_in_sample = torch.Tensor(embeddings_in_sample).to(device)
        embeddings = torch.Tensor(embeddings).to(device)
        
        tokenizer = TIGERTokenizer(config, item_2_semantic_id=None)
        tokenizer.fit(embeddings, device=device)
        semantic_ids = tokenizer.transform(embeddings, device=device)
        
        # Add a padding row to the embeddings matrix
        padding_row = np.zeros((1, semantic_ids.shape[1]), dtype=int)
        padded_semantic_ids = np.concatenate((padding_row, semantic_ids), axis=0)
    
        # Save semantic_ids to saved_id_path
        padded_semantic_ids.tofile(saved_id_path)
    
    # Load the semantic IDs from the saved path
    semantic_ids = load_semantic_ids(config)
    tokenizer = TIGERTokenizer(config, semantic_ids=semantic_ids)
    
    # Initialize the model
    model = TIGER(tiger_config, tokenizer)
    
    # Instantiate the DataProcessor
    data_processor = TIGERDataProcessor(max_length=config['max_history_len'], tokenizer=tokenizer)
    
    # Get DataLoaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        domain=domain,
        splits=['train', 'valid', 'test'],
        train_batch_size=tiger_config['trainer']['train_batch_size'],
        eval_batch_size=tiger_config['trainer']['eval_batch_size'],
        data_processor=data_processor,
        num_workers=config['num_workers']
    )
    
    # Initialize the evaluator and trainer
    evaluator = TIGEREvaluator(model, ks=config['eval_ks'])
    trainer = TIGERTrainer(config, device, model, evaluator)
    
    # Train the model
    trainer.fit(train_dataloader, val_dataloader)
    
    # Evaluate the model
    evaluator.evaluate(test_dataloader, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/quick_start.yaml', help='Path to the config file.')
    parser.add_argument('--device', type=int, default=0, help='GPU ID to use for training. Default is 0 (cuda:0).')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    
    print(config)
    
    draft_model_name = 'UniSRec'
    target_model_name = 'TIGER'
    
    domain, exp_id = config['dataset'], config['exp_id']
    draft_model_path = get_model_ckpt_path(draft_model_name, domain, exp_id)
    target_model_path = get_model_ckpt_path(target_model_name, domain, exp_id)
    saved_id_path = get_saved_id_path(domain, exp_id)
    draft_log_file_path = get_logfile_path(draft_model_name, domain, exp_id)
    target_log_file_path = get_logfile_path(target_model_name, domain, exp_id)
    
    os.makedirs(os.path.join('results', draft_model_name), exist_ok=True)
    os.makedirs(os.path.join('results', target_model_name), exist_ok=True)
    os.makedirs(os.path.join('logs', draft_model_name), exist_ok=True)
    os.makedirs(os.path.join('logs', target_model_name), exist_ok=True)
    
    if not os.path.exists(draft_model_path):
        print("Training the draft model...")
        train_drafter(config, device, draft_model_path, draft_log_file_path)
    
    if not os.path.exists(target_model_path):
        print("Training the generative model...")
        train_genrec(config, device, saved_id_path, target_model_path, target_log_file_path)

    print("Training completed.")
