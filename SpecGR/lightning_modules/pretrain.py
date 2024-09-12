import torch
from torch.optim import AdamW
import lightning as L
from transformers import get_scheduler
from models.specGR.specGR_train import SpecGR
from evaluator import SpecGRForTrainEvaluator
from SpecGR.lightning_modules.utils import calculate_optimizer_config_in_distributed_setting

class SpecGRPretrainLightningModule(L.LightningModule):
    def __init__(
        self, 
        model: SpecGR, 
        lambda_emb,
        lambda_gen,
        lr, 
        saved_model_path, 
        weight_decay, 
        warmup_steps, 
        test_start_index,
        semantic_ids, 
        evaluator: SpecGRForTrainEvaluator
    ):
        super().__init__()
        
        self.model = model
        self.lr = lr
        self.saved_model_path = saved_model_path
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.test_start_index = test_start_index
        self.lambda_emb = lambda_emb
        self.lambda_gen = lambda_gen
        
        self.evaluator = evaluator
        self.evaluator.device = self.device
        
        self.best_metrics = 0.0
        self.best_epoch = 0

        self.semantic_id_sequences = self._prepare_semantic_id_sequences(semantic_ids)

        self.train_outputs = []
        self.valid_outputs = []
        self.test_outputs = []
        
    def _prepare_semantic_id_sequences(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat((
            torch.full((semantic_ids.shape[0], 1), self.model.genrec.tokenizer.bos_token_id),
            semantic_ids,
            torch.full((semantic_ids.shape[0], 1), self.model.genrec.tokenizer.eos_token_id),
        ), dim=1).requires_grad_(False)
        
    def setup(self, stage):
        self.model = self.model.to(self.device)
        self.semantic_id_sequences = self.semantic_id_sequences.to(self.device)
        
    def on_train_epoch_start(self):
        self.train_outputs.clear()
    
    def training_step(self, batch, batch_idx):
        embedding_batch = batch['embedding']
        generative_batch = batch['generative']
        
        emb_loss = self.model.calculate_emb_loss(
            embedding_batch['input_ids'], 
            embedding_batch['attention_mask'], 
            embedding_batch['labels'],
            embedding_batch['item_id']
        )
        
        gen_loss = self.model.calculate_gen_loss(
            generative_batch['input_ids'], 
            generative_batch['attention_mask'], 
            generative_batch['labels']
        )
        
        loss = self.lambda_emb * emb_loss + self.lambda_gen * gen_loss
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")
        
        self._log_training_metrics(emb_loss, gen_loss, loss, embedding_batch['input_ids'].shape[0])
        
        return loss

    def _log_training_metrics(self, emb_loss, gen_loss, loss, batch_size):
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=False)
        self.log('emb_loss', emb_loss, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=False)
        self.log('gen_loss', gen_loss, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=False)
        
        avg_loss = torch.stack([x['loss'] for x in self.train_outputs]).mean() if self.train_outputs else -1
        self.log("avg_train_loss", avg_loss, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=False)
        
        self.train_outputs.append({"loss": loss, "emb_loss": emb_loss, "gen_loss": gen_loss})

    def on_train_epoch_end(self):
        avg_losses = {k: torch.stack([x[k].to(self.device) for x in self.train_outputs]).mean() for k in ['loss', 'emb_loss', 'gen_loss']}
        self.log_dict({f"avg_{k}": v for k, v in avg_losses.items()}, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    def on_validation_epoch_start(self):
        self.valid_outputs.clear()
        with torch.no_grad():
            valid_item_sequences = self.semantic_id_sequences[:self.test_start_index+1].to(self.device)
            self.valid_item_embs = self.model.encode(valid_item_sequences).detach().to(self.device)

    def validation_step(self, batch, batch_idx):
        batch_size = batch['input_ids'].shape[0]
        metrics = self.evaluator.evaluation_step(batch, self.device, test_item_embs=self.valid_item_embs)
        metrics = self.evaluator.convert_metrics_to_tensor(metrics, self.device)
        self.valid_outputs.append(metrics)
        
        self._log_validation_metrics(metrics, batch_size)

        return metrics

    def _log_validation_metrics(self, metrics, batch_size):
        avg_recall_h = torch.stack([x['recall_h_50'].to(self.device) for x in self.valid_outputs]).mean()
        avg_recall_b = torch.stack([x['recall_b_50'].to(self.device) for x in self.valid_outputs]).mean()
        avg_valid_loss = torch.stack([x['loss_h'].to(self.device) for x in self.valid_outputs]).mean()
        self.log('valid_Recall_Emb@50', avg_recall_h, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=True)
        self.log('valid_Recall_Gen@50', avg_recall_b, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=True)
        self.log('valid_loss', avg_valid_loss, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=True)

    def on_validation_epoch_end(self):
        avg_metrics = self.evaluator.process_evaluation_result(self.valid_outputs)
        avg_metrics = self.evaluator.convert_metrics_to_tensor(avg_metrics, self.device)
        if avg_metrics["recall_h_50"] > self.best_metrics:
            self.best_metrics = avg_metrics["recall_h_50"]
            self.best_epoch = self.current_epoch
            self._save_checkpoint(self.saved_model_path)

    def on_test_epoch_start(self):
        self.test_outputs.clear()
        self.model.load_state_dict(torch.load(self.saved_model_path))
        
        with torch.no_grad():
            test_item_sequences = self.semantic_id_sequences.to(self.device)
            self.test_item_embs = self.model.encode(test_item_sequences).detach().to(self.device)

    def test_step(self, batch, batch_idx):
        metrics = self.evaluator.evaluation_step(batch, self.device, test_item_embs=self.test_item_embs)
        metrics = self.evaluator.convert_metrics_to_tensor(metrics, self.device)
        self.test_outputs.append(metrics)
        return metrics
    
    def on_test_epoch_end(self):
        avg_metrics = self.evaluator.process_evaluation_result(self.test_outputs)
        avg_metrics = self.evaluator.convert_metrics_to_tensor(avg_metrics, self.device)

    def _save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def configure_optimizers(self):
        total_training_steps, total_warmup_steps, scaled_lr = calculate_optimizer_config_in_distributed_setting(
            self.trainer, self.warmup_steps, self.lr, self.weight_decay
        )

        optimizer = AdamW(self.model.parameters(), lr=scaled_lr, weight_decay=self.weight_decay)
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_training_steps=total_training_steps,
            num_warmup_steps=total_warmup_steps,   
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]