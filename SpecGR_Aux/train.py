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

def train_drafter(
    config, 
    device, 
    saved_model_path = None, 
    log_file_path = None
):
    """
    Train auxiliary drafter model (Section 3.3: Auxiliary Model as Drafter).
    Uses UniSRec as inductive recommendation model for drafting unseen items.
    """
    domain = config['dataset']
    unisrec_config = config['UniSRec']
    saved_model_path = saved_model_path or get_model_ckpt_path('UniSRec', domain, exp_id)
    log_file_path = log_file_path or get_logfile_path('UniSRec', domain, exp_id)

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
    trainer = UniSRecTrainer(
        config=config,
        device=device,
        model=model,
        evaluator=evaluator,
        val_item_embeddings=item_embeddings[: test_start_index + 1],
        log_file_path=log_file_path,
        saved_model_ckpt=saved_model_path
    )

    # Train the model
    trainer.fit(train_dataloader, val_dataloader)

    # Evaluate the model
    model.load_state_dict(torch.load(saved_model_path))
    results = evaluator.evaluate(test_dataloader, device=device)
    
    return model, results


def train_genrec(
    config, 
    device, 
    saved_id_path = None,
    saved_model_path = None,
    log_file_path = None
):
    """
    Train generative recommendation model (Section 3.2: Target model for verification).
    Uses TIGER as the target GR model that acts as verifier in SpecGR framework.
    """
    domain = config['dataset']
    exp_id = config['exp_id']
    tiger_config = config['TIGER']
    saved_model_path = saved_model_path or get_model_ckpt_path(target_model_name, domain, exp_id)
    saved_id_path = saved_id_path or get_saved_id_path(domain, exp_id)
    log_file_path = log_file_path or get_logfile_path(target_model_name, domain, exp_id)

    print('saved_id_path', saved_id_path)
    if not os.path.exists(saved_id_path):
        print('Generating and saving semantic IDs...')
        embeddings = load_item_embeddings(config)[1:, :] # remove the padding row during training
        unseen_start_index = config['unseen_start_index']
        
        tokenizer = TIGERTokenizer(config, semantic_ids = None)
        semantic_ids = tokenizer.fit_transform(embeddings, unseen_start_index, device)

        # Add a padding row to the embeddings matrix
        padding_row = np.zeros((1, semantic_ids.shape[1]), dtype=int)
        padded_semantic_ids = np.concatenate((padding_row, semantic_ids), axis=0)

        # Save semantic_ids to saved_id_path
        padded_semantic_ids.tofile(saved_id_path)

    # Load the semantic IDs from the saved path
    semantic_ids = load_semantic_ids(config, saved_id_path)
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
    trainer = TIGERTrainer(
        config=config,
        device=device,
        model=model,
        evaluator=evaluator,
        log_file_path=log_file_path,
        saved_model_ckpt=saved_model_path
    )

    # Train the model
    trainer.fit(train_dataloader, val_dataloader)
    
    # Evaluate the model
    model.load_state_dict(torch.load(saved_model_path))
    results = evaluator.evaluate(test_dataloader, device=device)
    
    return model, results

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

    # if not os.path.exists(draft_model_path):
    #     print("Training the draft model...")
    #     train_drafter(config, device, draft_model_path, draft_log_file_path)
    
    if not os.path.exists(target_model_path):
        print("Training the generative model...")
        train_genrec(config, device, saved_id_path, target_model_path, target_log_file_path)

    print("Training completed.")
