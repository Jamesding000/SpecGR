import os
import torch
import argparse
import numpy as np
import json
from torch.utils.data import DataLoader
from evaluator import SpecGRAuxEvaluator
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from dataloader import UniSRecDataProcessor, TIGERDataProcessor, get_dataloaders
from models.genrec.TIGER.tokenizer import TIGERTokenizer
from models.genrec.TIGER.model import TIGER
from models.draft.UniSRec.model import UniSRec
from models.draft.UniSRec.model import UniSRecDrafter
from models.specGR.specGR_inference import SpecGRAuxForRec
from utils import load_config, get_logfile_path, get_model_ckpt_path, get_saved_id_path, load_semantic_ids, load_item_embeddings

def load_unisrec_data_and_model(config, device):
    domain = config['dataset']
    exp_id = config['exp_id']
    draft_model_name = 'UniSRec'
    unisrec_config = config[draft_model_name]

    draft_model_path = get_model_ckpt_path(draft_model_name, domain, exp_id)
    unseen_start_index = config["unseen_start_index"]

    # Load item embeddings
    item_embeddings = load_item_embeddings(config)
    item_embeddings = torch.from_numpy(item_embeddings).to(device)

    # Initialize and load the UniSRec model
    unisrec = UniSRec(unisrec_config, item_embeddings=item_embeddings[:unseen_start_index + 1]).to(device)
    unisrec.load_state_dict(torch.load(draft_model_path, map_location=device))
    print("Loaded UniSRec model weights from:", draft_model_path)

    # Instantiate the DataProcessor
    draft_data_processor = UniSRecDataProcessor(max_length=config['max_history_len'])
    draft_test_in_sample_dataloader, draft_test_cold_start_dataloader = get_dataloaders(
        domain=domain,
        splits=['test_in_sample', 'test_cold_start'],
        train_batch_size=1,
        eval_batch_size=1,
        data_processor=draft_data_processor,
        num_workers=config['num_workers']
    )

    return unisrec, item_embeddings, draft_test_in_sample_dataloader, draft_test_cold_start_dataloader

def load_tiger_data_and_model(config, device):
    domain = config['dataset']
    exp_id = config['exp_id']
    genrec_name = 'TIGER'
    tiger_config = config[genrec_name]

    genrec_path = get_model_ckpt_path(genrec_name, domain, exp_id)
    saved_id_path = get_saved_id_path(domain, exp_id)
    
    # Load semantic IDs
    semantic_ids = load_semantic_ids(config)
    tokenizer = TIGERTokenizer(config, semantic_ids=semantic_ids)
    semantic_ids = torch.from_numpy(semantic_ids).long().to(device)

    # Initialize tokenizer and TIGER model
    genrec = TIGER(tiger_config, tokenizer).to(device)
    genrec.load_state_dict(torch.load(genrec_path, map_location=device))
    print("Loaded TIGER model weights from:", genrec_path)

    # Instantiate the DataProcessor
    target_data_processor = TIGERDataProcessor(max_length=config['max_history_len'], tokenizer=tokenizer)
    target_test_in_sample_dataloader, target_test_cold_start_dataloader = get_dataloaders(
        domain=domain,
        splits=['test_in_sample', 'test_cold_start'],
        train_batch_size=1,
        eval_batch_size=1,
        data_processor=target_data_processor,
        num_workers=config['num_workers']
    )

    return genrec, semantic_ids, target_test_in_sample_dataloader, target_test_cold_start_dataloader


def run(config, device):
    # Load UniSRec data and model
    unisrec, item_embeddings, draft_test_in_sample_dataloader, draft_test_cold_start_dataloader = load_unisrec_data_and_model(config, device)

    # Load TIGER data and model
    genrec, semantic_ids, target_test_in_sample_dataloader, target_test_cold_start_dataloader = load_tiger_data_and_model(config, device)
    
    test_in_sample_dataloader = CombinedLoader({
        'draft': draft_test_in_sample_dataloader, 
        'generative': target_test_in_sample_dataloader
    }, mode='max_size_cycle')
    
    test_cold_start_dataloader = CombinedLoader({
        'draft': draft_test_cold_start_dataloader, 
        'generative': target_test_cold_start_dataloader
    }, mode='max_size_cycle')
    
    # Prepare SpecGR
    params = {
        "draft_size": args.draft_size,
        "threshold": args.threshold,
        "num_beams": args.num_beams,
    }
    
    draft_model = UniSRecDrafter(unisrec)
    specGR = SpecGRAuxForRec(genrec, draft_model, config, params)

    # Initialize the evaluator
    evaluator = SpecGRAuxEvaluator(specGR, ks=config['eval_ks'])

    # Evaluate the model
    evaluator.evaluate(test_in_sample_dataloader, device=device, item_embeddings=item_embeddings, semantic_ids=semantic_ids)
    evaluator.evaluate(test_cold_start_dataloader, device=device, item_embeddings=item_embeddings, semantic_ids=semantic_ids)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/quick_start.yaml', help='Path to the config file.')
    parser.add_argument('--device', type=int, default=0, help='GPU ID to use for training. Default is 0 (cuda:0).')
    parser.add_argument("--ks", type=json.loads, default=[10, 50], help="List of top K values to evaluate, passed as a JSON string. Example: '[5,10,20,50]'")
    parser.add_argument("--draft_size", type=int, default=50, help="Draft size for the process.")
    parser.add_argument("--threshold", type=float, default=-1.8, help="Threshold for acceptance.")
    parser.add_argument("--num_beams", type=int, default=20, help="Number of beams to use.")
    parser.add_argument("--num_retrieved", type=int, default=1000, help="Number of retrieved items if using a retriever model.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    
    print(config)
    
    run(config, device)