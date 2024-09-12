import argparse
import torch
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from models.genrec.TIGER.model import TIGER
from models.genrec.TIGER.tokenizer import TIGERTokenizer
from models.specGR.specGR_train import SpecGR
from utils import load_config, get_model_ckpt_path, get_saved_id_path, get_logfile_path, parse_devices, load_semantic_ids
from SpecGR.lightning_modules.pretrain import SpecGRPretrainLightningModule
from SpecGR.lightning_modules.finetune import SpecGRFinetuneLightningModule
from SpecGR.lightning_modules.callbacks import NICE_PROGRESS_BAR
from SpecGR.lightning_modules.data import SpecGRPretrainDataModule, SpecGRFinetuneDataModule
from dataloader import SpecGRDataProcessor
from evaluator import SpecGRForTrainEvaluator

def setup_paths(config):
    domain, exp_id = config['dataset'], config['exp_id']
    return {
        'pretrain_model_path': get_model_ckpt_path('SpecGR', domain, exp_id, suffix='_pt'),
        'finetune_model_path': get_model_ckpt_path('SpecGR', domain, exp_id, suffix='_ft'),
        'finetune_emb_path': get_model_ckpt_path('SpecGR', domain, exp_id, suffix='_emb_ft'),
        'saved_id_path': get_saved_id_path(domain, exp_id),
        'log_file_path': get_logfile_path('SpecGR', domain, exp_id)
    }

def prepare_model_and_data(config, paths, devices):
    semantic_ids = load_semantic_ids(config)
    tokenizer = TIGERTokenizer(config, semantic_ids=semantic_ids)
    semantic_ids = torch.from_numpy(semantic_ids)
    
    base_model = TIGER(config["TIGER"], tokenizer)
    model = SpecGR(base_model, config["SpecGR"])
    model.train()
    
    evaluator = SpecGRForTrainEvaluator(model, ks=config['eval_ks'], device=devices[0])
    data_processor = SpecGRDataProcessor(max_length=config['max_history_len'], tokenizer=tokenizer)
    
    return model, semantic_ids, evaluator, data_processor

def setup_trainer(trainer_config, devices, log_path, name):
    logger = TensorBoardLogger(save_dir=log_path, name=name)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    return L.Trainer(
        strategy="ddp" if isinstance(devices, (list, tuple)) and len(devices) > 1 else 'auto',
        accelerator="gpu",
        devices=devices,
        max_epochs=trainer_config['epochs'],
        precision='bf16-mixed',
        check_val_every_n_epoch=1 if name == 'finetune' else 2,
        logger=logger,
        gradient_clip_val=1.0, 
        gradient_clip_algorithm="norm",
        callbacks=[lr_monitor, NICE_PROGRESS_BAR],
    )

def pretrain(config, model, semantic_ids, evaluator, data_processor, paths, devices):
    trainer_config = config["SpecGR"]["pretrain_trainer"]
    
    ltn_model = SpecGRPretrainLightningModule(
        model=model,
        lambda_emb=trainer_config['lambda_emb'],
        lambda_gen=trainer_config['lambda_gen'],
        lr=trainer_config["lr"],
        warmup_steps=trainer_config['warmup_steps'],
        saved_model_path=paths['pretrain_model_path'],
        weight_decay=trainer_config["weight_decay"],
        test_start_index=config["test_start_index"],
        semantic_ids=semantic_ids,
        evaluator=evaluator,
    )
    
    data_module = SpecGRPretrainDataModule(
        domain=config['dataset'],
        splits=['train', 'valid', 'test'],
        data_processor=data_processor, 
        emb_batch_size=trainer_config["train_emb_batch_size"],
        gen_batch_size=trainer_config["train_gen_batch_size"],
        eval_batch_size=trainer_config["eval_batch_size"],
        num_workers=config['num_workers']
    )
    
    trainer = setup_trainer(trainer_config, devices, paths['log_file_path'], 'pretrain')
    
    trainer.fit(ltn_model, data_module)
    
    ltn_model.model.load_state_dict(torch.load(ltn_model.saved_model_path))
    best_val_result = trainer.validate(ltn_model, data_module)
    test_result = trainer.test(ltn_model, data_module)
    
    return ltn_model.model, best_val_result[0], test_result[0]

def finetune(config, model, semantic_ids, evaluator, data_processor, paths, devices):
    trainer_config = config["SpecGR"]["finetune_trainer"]
    
    model.load_state_dict(torch.load(paths['pretrain_model_path']))
    print("Loaded model weights from:", paths['pretrain_model_path'])
    model.train()

    ltn_model = SpecGRFinetuneLightningModule(
        model=model,
        lambda_emb=trainer_config['lambda_emb'],
        lambda_gen=trainer_config['lambda_gen'],
        lr=trainer_config["lr"],
        saved_model_path=paths['finetune_model_path'],
        saved_embedding_path=paths['finetune_emb_path'],
        weight_decay=trainer_config["weight_decay"],
        warmup_steps=trainer_config['warmup_steps'],
        unseen_start_index=config["unseen_start_index"],
        test_start_index=config["test_start_index"],
        semantic_ids=semantic_ids,
        evaluator=evaluator
    )

    data_module = SpecGRFinetuneDataModule(
        domain=config['dataset'],
        splits=['train', 'valid', 'test'],
        data_processor=data_processor,
        train_batch_size=trainer_config['train_batch_size'],
        eval_batch_size=trainer_config["eval_batch_size"],
        num_workers=config['num_workers'],
    )

    trainer = setup_trainer(trainer_config, devices, paths['log_file_path'], 'finetune')

    trainer.fit(ltn_model, data_module)
    
    ltn_model._load_checkpoint()
    best_val_result = trainer.validate(ltn_model, data_module)
    test_result = trainer.test(ltn_model, data_module)
    
    return ltn_model.model, best_val_result[0], test_result[0]

def train(config, devices):
    paths = setup_paths(config)
    model, semantic_ids, evaluator, data_processor = prepare_model_and_data(config, paths, devices)
    
    model, best_val_result_pt, test_result_pt = pretrain(config, model, semantic_ids, evaluator, data_processor, paths, devices)
    model, best_val_result_ft, test_result_ft = finetune(config, model, semantic_ids, evaluator, data_processor, paths, devices)
    return model, best_val_result_ft, test_result_ft

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--devices', type=str, required=True, help='GPU devices to use, e.g., 3 or [1,2,3]')
    args = parser.parse_args()

    config = load_config(args.config)
    devices = parse_devices(args.devices)
    
    train(config, devices)