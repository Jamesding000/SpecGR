import argparse
import torch
import numpy as np
import lightning.pytorch as L
from models.genrec.TIGER.model import TIGER
from models.genrec.TIGER.tokenizer import TIGERTokenizer
# from models.SpecGR.specGR_inference import SpecGRForRec
from models.SpecGR.SpecGR_clean import SpecGRForRec
from SpecGR.lightning_modules.inference import SpecGRForRecLightningModule
from SpecGR.lightning_modules.callbacks import NICE_PROGRESS_BAR
from dataloader import SpecGRDataProcessor, get_dataloaders
from evaluator import SpecGREvaluator
from utils import (
    load_config,
    get_model_ckpt_path,
    get_saved_id_path,
    parse_devices,
    set_random_seed,
    load_semantic_ids
)

def load_data_and_model(config, params, devices, eval_mode='test'):
    
    assert eval_mode in ['valid', 'test'], "Eval mode can only be 'valid' or 'test'."
    
    print(config)
    
    domain = config['dataset']
    exp_id = config['exp_id']
    tiger_config = config["TIGER"]
    speGR_config = config['SpecGR']
    
    saved_model_path = get_model_ckpt_path('SpecGR', domain, exp_id, suffix='_ft')
    saved_embedding_path = get_model_ckpt_path('SpecGR', domain, exp_id, suffix='_emb_ft')
    saved_id_path = get_saved_id_path(domain, exp_id)

    semantic_ids = load_semantic_ids(config)
    tokenizer = TIGERTokenizer(config, semantic_ids=semantic_ids)
    semantic_ids = torch.from_numpy(semantic_ids)
    
    base_model = TIGER(tiger_config, tokenizer)
    model = SpecGRForRec(base_model, config, semantic_ids, params)
   
    model.eval()
    
    evaluator = SpecGREvaluator(model, ks=config['eval_ks'], device=devices[0])
    
    model_module = SpecGRForRecLightningModule(
        model=model,
        saved_model_path=saved_model_path,
        saved_embedding_path=saved_embedding_path,
        config=config,
        semantic_ids=semantic_ids,
        evaluator=evaluator
    )
    
    model_module._load_checkpoint()

    specGR_data_processor = SpecGRDataProcessor(max_length=config['max_history_len'], tokenizer=tokenizer)
    specGR_in_sample_dataloader, specGR_cold_start_dataloader = get_dataloaders(
        domain=domain,
        splits=[f'{eval_mode}_in_sample', f'{eval_mode}_cold_start'],
        train_batch_size=speGR_config['inference']['eval_batch_size'],
        eval_batch_size=speGR_config['inference']['eval_batch_size'],
        data_processor=specGR_data_processor,
        num_workers=config['num_workers']  # might be only working with 1
    )
    
    return model_module, specGR_cold_start_dataloader, specGR_in_sample_dataloader


def run_SpecGR(config, params, devices, eval_mode='test'):

    model_module, specGR_cold_start_dataloader, specGR_in_sample_dataloader = (
        load_data_and_model(config, params, devices, eval_mode)
    )

    def evaluate(dataloader):
        trainer = L.Trainer(
            strategy="ddp" if isinstance(devices, (list, tuple)) and len(devices) > 1 else 'auto',
            accelerator="gpu",
            devices=devices,
            precision='bf16-mixed',
            callbacks=[NICE_PROGRESS_BAR],
            # callbacks=[ManualProgressBar(evaluator=evaluator)],
            # limit_test_batches=max_steps if max_steps is not None else 1.0,
            logger=False  # Disable logging
        )
        result = trainer.test(model_module, dataloaders=dataloader)
        return result[0]

    cold_start_result = evaluate(specGR_cold_start_dataloader)
    in_sample_result = evaluate(specGR_in_sample_dataloader)

    return in_sample_result, cold_start_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--devices', type=str, required=True, help='GPU devices to use, e.g., 3 or [1,2,3]')
    parser.add_argument('--eval_mode', type=str, required=True, help="Evaluation mode: 'valid' or 'test'")
    parser.add_argument('--draft_size', type=int, default=20, help='Draft size for the process.')
    parser.add_argument('--threshold', type=float, default=-1.8, help='Threshold for acceptance.')
    parser.add_argument('--num_beams', type=int, default=50, help='Number of beams to use.')
    args = parser.parse_args()

    config = load_config(args.config)
    devices = parse_devices(args.devices)
    
    
    params = {
        'draft_size': args.draft_size,
        'threshold': args.threshold, 
        'num_beams': args.num_beams,
    }

    set_random_seed(42)
    
    run_SpecGR(config, params, devices, eval_mode='test')
