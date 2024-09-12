import os
import torch
import numpy as np
from torch.optim import Adam, AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_scheduler
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

from utils import get_logfile_path, get_model_ckpt_path

class AbstractTrainer(ABC):
    def __init__(self, config: Dict[str, Any], device: torch.device, model: torch.nn.Module, evaluator: Any):
        self.config = config
        self.device = device
        self.model = model.to(device)
        self.model_name = self.model.__class__.__name__

        self.trainer_config = config[self.model_name]['trainer']
        self.val_metric = self.config['val_metric']
        self.num_epochs = self.trainer_config.get('epochs')
        self.clip_grad_norm_value = self.trainer_config.get('clip_grad_norm')
        self.stopping_epochs = self.trainer_config.get('stopping_epochs')
        self.learning_rate = self.trainer_config['lr']
        self.weight_decay = self.trainer_config['weight_decay']
        self.train_batch_size = self.trainer_config['train_batch_size']
        self.eval_batch_size = self.trainer_config['eval_batch_size']

        self.evaluator = evaluator
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        self.best_metric = float('-inf')
        self.best_epoch = 0

        self.setup_writer()
        self.saved_model_ckpt = get_model_ckpt_path(self.model_name, self.config['dataset'], self.config['exp_id'])

    def _get_optimizer(self) -> torch.optim.Optimizer:
        optimizer_type = self.trainer_config['optimizer']
        
        if optimizer_type == 'Adam':
            return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif optimizer_type == 'AdamW':
            return AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _get_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if 'scheduler' not in self.trainer_config or self.trainer_config['scheduler'] is None:
            return None
        elif self.trainer_config['scheduler'] == 'cosine':
            total_steps = self.trainer_config['total_steps']
            warmup_steps = self.trainer_config['warmup_steps']
            return get_scheduler(
                name="cosine",
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            raise NotImplementedError(f"Unimplemented scheduler type: {self.trainer_config['scheduler']}")

    def setup_writer(self) -> None:
        log_dir = get_logfile_path(self.model_name, self.config['dataset'], self.config['exp_id'])
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def fit(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        model_path = self.saved_model_ckpt
        
        num_epochs = self.num_epochs or int(np.ceil(self.trainer_config['total_steps'] / len(train_dataloader)))
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1} / {num_epochs}]",
            )

            for batch in train_progress_bar:
                self.optimizer.zero_grad()
                batch = {key: value.to(self.device) for key, value in batch.items()}
                loss = self.model.calculate_loss(**batch)
                loss.backward()

                if self.clip_grad_norm_value:
                    clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)

                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            self.writer.add_scalar('training_loss', avg_loss, epoch)

            if (epoch + 1) % 2 == 0:
                eval_results = self.validate(val_dataloader)
                
                for key, value in eval_results.items():
                    self.writer.add_scalar(key, value, epoch)

                if self.val_metric not in eval_results:
                    raise KeyError(f"Validation metric '{self.val_metric}' not found in evaluation results.")
                
                val_score = eval_results[self.val_metric]
                if val_score > self.best_metric:
                    self.best_metric = val_score
                    self.best_epoch = epoch + 1
                    torch.save(self.model.state_dict(), model_path)

                if self.stopping_epochs and epoch + 1 - self.best_epoch >= self.stopping_epochs:
                    break

    def update_best_model(self, eval_results: Dict[str, float], epoch: int, model_path: str) -> None:
        metric_value = eval_results[self.val_metric]
        if metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), model_path)
    
    @abstractmethod
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        pass
    
class UniSRecTrainer(AbstractTrainer):
    def __init__(self, config: Dict[str, Any], device: torch.device, model: torch.nn.Module, evaluator: Any, val_item_embeddings: torch.Tensor):
        super().__init__(config, device, model, evaluator)
        self.val_item_embeddings = val_item_embeddings

    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:  
        return self.evaluator.evaluate(val_dataloader, device=self.device, item_embeddings=self.val_item_embeddings)

class TIGERTrainer(AbstractTrainer):
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:  
        return self.evaluator.evaluate(val_dataloader, device=self.device)