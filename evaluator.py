from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from models.genrec.genrec import AbstractGenRec
from models.specGR.specGR_inference import SpecGRForRec, SpecGRAuxForRec
from models.specGR.specGR_train import SpecGR

class Evaluator(ABC):
    def __init__(self, model: Any, ks: List[int]):
        self.model = model
        self.ks = ks
        self.device: Optional[torch.device] = None
        self.metrics: List[Dict[str, float]] = []

    def calculate_dcg(self, scores) -> float:
        scores = np.asfarray(scores)
        return np.sum((2**scores - 1) / np.log2(np.arange(2, scores.size + 2)))

    def calculate_ndcg_at_k(self, r, k) -> float:
        r = np.asfarray(r)[:k]
        dcg_max = self.calculate_dcg(sorted(r, reverse=True))
        if not dcg_max:
            return 0.0
        return self.calculate_dcg(r) / dcg_max

    def calculate_metrics_at_k(self, matches, k) -> float:
        recall = matches[:k].sum() / 1.0  # Assuming each label has only 1 correct match.
        ndcg = self.calculate_ndcg_at_k(matches, k)
        return recall, ndcg
    
    def log_dict(self, metrics_dict: Dict[str, float], prog_bar: bool = False, logger: bool = True) -> None:
        for key, value in metrics_dict.items():
            print(f"{key}: {value:.6f}")
    
    @property
    def metrics_to_log(self) -> List[str]:
        return []
       
    def set_prog_bar_description(self, metrics_list: List[Dict[str, float]]) -> str:
        if len(self.metrics_to_log) == 0:
            return "Evaluating"
        
        avg_metrics = {}
        for metric in self.metrics_to_log:
            values = [m[metric] for m in metrics_list if metric in m]
            if values:
                avg_metrics[metric] = sum(values) / len(values)
            
        return ", ".join(
            f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {int(value)}"
            for key, value in avg_metrics.items()
        )

    def evaluate(self, dataloader: torch.utils.data.DataLoader, device: torch.device, **kwargs: Any) -> Dict[str, float]:
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.metrics = []

        progress_bar = tqdm(iter(dataloader))

        for batch in progress_bar:
            metrics = self.evaluation_step(batch, device, **kwargs)
            self.metrics.append(metrics)
            progress_bar.set_description(self.set_prog_bar_description(self.metrics))

        progress_bar.close()
        
        avg_metrics = self.process_evaluation_result(self.metrics)
        self.log_dict(avg_metrics)
                 
        return avg_metrics

    def process_evaluation_result(self, outputs: List[Dict[str, float]]) -> Dict[str, float]:
        avg_metrics = {}
        for key in outputs[0]:
            values = [float(x[key]) for x in outputs]
            avg_metrics[key] = round(sum(values) / len(values), 6)
        return avg_metrics

    def convert_metrics_to_tensor(self, metrics: Dict[str, float], device: torch.device) -> Dict[str, Tensor]:
        return {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in metrics.items()}

    @abstractmethod       
    def evaluation_step(self, batch: Dict[str, Tensor], device: torch.device, **kwargs: Any) -> Dict[str, float]:
        pass
    
    def log_metrics_table(self, avg_metrics: Dict[str, Union[float, Tensor]]) -> None:
        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim", width=20)
        table.add_column("Value", justify="right")

        for metric, value in avg_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, float):
                value = f"{value:.4f}"
            table.add_row(metric, str(value))

        console.print(table)

class DrafterEvaluator(Evaluator):
    def calculate_drafter_metrics_at_k(self, outputs: Tensor, labels: Tensor, k: int) -> Tuple[float, float]:
        batch_size, _ = outputs.shape
        recalls_at_k, ndcgs_at_k = [], []

        for i in range(batch_size):
            matches = torch.eq(outputs[i][:k], labels[i]).cpu().numpy()
            recall, ndcg = self.calculate_metrics_at_k(matches, k)
            recalls_at_k.append(recall)
            ndcgs_at_k.append(ndcg)

        return np.mean(recalls_at_k), np.mean(ndcgs_at_k)

class GenerativeEvaluator(Evaluator):
    def calculate_generative_metrics_at_k(self, outputs: Tensor, labels: Tensor, k: int) -> Tuple[float, float]:
        batch_size, _, _ = outputs.shape
        recalls, ndcgs = [], []

        for i in range(batch_size):
            matches = torch.all(torch.eq(outputs[i][:k].unsqueeze(1), labels[i].unsqueeze(0)), dim=2)
            matches = matches.any(dim=1).cpu().numpy()
            recall, ndcg = self.calculate_metrics_at_k(matches, k)
            recalls.append(recall)
            ndcgs.append(ndcg)

        return np.mean(recalls), np.mean(ndcgs)

class UniSRecEvaluator(DrafterEvaluator):
    def __init__(self, model: Any, ks: List[int], item_embeddings: Optional[Tensor] = None):
        super().__init__(model, ks)
        self.item_embeddings = item_embeddings

    def evaluation_step(self, batch: Dict[str, Tensor], device: torch.device, **kwargs: Any) -> Dict[str, float]:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["length"].to(device)
        labels = batch["labels"].to(device)

        logits = self.model.full_sort_predict(input_ids, lengths) if self.item_embeddings is None else \
                 self.model.predict(input_ids, lengths, self.item_embeddings, self.item_embeddings)

        loss = self.model.loss_fct(logits, labels).item()
        metrics = {"loss": loss}

        for k in self.ks:
            _, indices = torch.topk(logits, k, dim=1)
            recall, ndcg = self.calculate_drafter_metrics_at_k(indices, labels, k)
            metrics[f"recall_{k}"] = recall
            metrics[f"ndcg_{k}"] = ndcg

        return metrics

    @property
    def metrics_to_log(self) -> List[str]:
        return [f"recall_{max(self.ks)}", f"ndcg_{max(self.ks)}"]

class TIGEREvaluator(GenerativeEvaluator):
    def evaluation_step(self, batch: Dict[str, Tensor], device: torch.device, **kwargs: Any) -> Dict[str, float]:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.item()
        metrics = {"loss": loss}

        max_k = max(self.ks)
        output_sequences = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, k=max_k
        )
        labels = labels[:, :-1]

        for k in self.ks:
            k_outputs = output_sequences[:, :k, :]
            recall, ndcg = self.calculate_generative_metrics_at_k(k_outputs, labels, k)
            metrics[f"recall_{k}"] = recall
            metrics[f"ndcg_{k}"] = ndcg

        return metrics

    @property
    def metrics_to_log(self) -> List[str]:
        return [f"recall_{max(self.ks)}", f"ndcg_{max(self.ks)}"]

class SpecGRAuxEvaluator(GenerativeEvaluator):
    
    def __init__(self, model: SpecGRAuxForRec, ks: List[int]):
        super().__init__(model, ks)
        self.model = model
        
    def evaluation_step(self, batch: Dict[str, Tensor], device: torch.device, **kwargs: Any) -> Dict[str, float]:
        batch = batch[0]
        batch_draft, batch_target = batch['draft'], batch['generative']
        inputs_draft = batch_draft["input_ids"].to(device)
        lengths_draft = batch_draft["length"].to(device)
        inputs_target = batch_target["input_ids"].to(device)
        attention_mask_target = batch_target["attention_mask"].to(device)
        labels_target = batch_target["labels"].to(device)
        
        constraints = kwargs.get('constraints', None)
        test_item_embs = kwargs['item_embeddings']
        semantic_ids = kwargs['semantic_ids']
        max_k = max(self.ks)

        recommended_items, scores, runtime_info = self.model.recommend(
            input_ids=inputs_target,
            attention_mask=attention_mask_target,
            k=max_k,
            semantic_ids=semantic_ids,
            constraints=constraints,
            return_info=True,
            item_seq=inputs_draft,
            item_seq_len=lengths_draft,
            item_embeddings=test_item_embs,
        )

        metrics = {
            "rounds": runtime_info["exit_rounds"],
            "accepted": runtime_info["num_accepted"]
        }

        for k in self.ks:
            k_outputs = recommended_items[:k, :].unsqueeze(0)
            recall, ndcg = self.calculate_generative_metrics_at_k(k_outputs, labels_target[:, :-1], k)
            metrics[f"recall_{k}"] = recall
            metrics[f"ndcg_{k}"] = ndcg

        return metrics
    
    @property
    def metrics_to_log(self) -> List[str]:
        return [f"recall_{max(self.ks)}", f"ndcg_{max(self.ks)}", "accepted", "rounds"]

class SpecGREvaluator(GenerativeEvaluator):
    def __init__(self, model: SpecGRForRec, ks: List[int], device: torch.device):
        super().__init__(model, ks)
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.device = device

    def evaluation_step(self, batch: Dict[str, Tensor], device: torch.device, **kwargs: Any) -> Dict[str, float]:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        constraints = kwargs.get('constraints', None)
        test_item_embs = kwargs['test_item_embs']
        semantic_ids = kwargs['semantic_ids']
        
        max_k = max(self.ks)

        recommended_items, scores, runtime_info = self.model.recommend(
            input_ids,
            attention_mask,
            max_k,
            constraints=constraints,
            semantic_ids=semantic_ids,
            return_info=True,
            test_item_embs=test_item_embs,
        )

        metrics = {}
        for k in self.ks:
            recall, ndcg = self.calculate_generative_metrics_at_k(
                recommended_items[:, :k].unsqueeze(0), labels[:, :-1], k
            )
            metrics[f"recall_{k}"] = recall
            metrics[f"ndcg_{k}"] = ndcg

        metrics[f"accepted_{max_k}"] = runtime_info["num_accepted"]
        metrics[f"rounds_{max_k}"] = runtime_info["exit_rounds"]

        return metrics
    
    @property
    def metrics_to_log(self) -> List[str]:
        return [f"recall_{max(self.ks)}", f"ndcg_{max(self.ks)}", "accepted", "rounds"]

class SpecGRForTrainEvaluator(GenerativeEvaluator, DrafterEvaluator):
    def __init__(self, model: SpecGR, ks: List[int], device: torch.device):
        super().__init__(model, ks)
        self.device = device

    def evaluation_step(self, batch: Dict[str, Tensor], device: torch.device, **kwargs: Any) -> Dict[str, float]:
        test_item_embs: Tensor = kwargs['test_item_embs']

        model = self.model
        batch_size: int = batch["input_ids"].shape[0]
        input_ids: Tensor = batch["input_ids"].to(device)
        attention_mask: Tensor = batch["attention_mask"].to(device)
        item_ids: Tensor = batch["item_id"].to(device)
        labels: Tensor = batch["labels"].to(device)

        encoder_outputs: Dict[str, Tensor] = model.genrec.encoder_forward(input_ids=input_ids, attention_mask=attention_mask)
        embeddings: Tensor = model.encoder_outputs_to_embedding(encoder_outputs, attention_mask)
        logits: Tensor = torch.matmul(embeddings, test_item_embs.transpose(0, 1))

        emb_loss: float = model.emb_loss_fct(logits, item_ids).item()
        metrics: Dict[str, float] = {"loss_h": emb_loss}

        max_k: int = max(self.ks)
        for k in self.ks:
            _, indices = torch.topk(logits, k, dim=1)
            recall, ndcg = self.calculate_drafter_metrics_at_k(indices, item_ids, k)
            metrics[f"recall_h_{k}"] = recall
            metrics[f"ndcg_h_{k}"] = ndcg

        loss: Tensor = model.genrec(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
        metrics["loss_b"] = loss.item()

        output_sequences = self.model.genrec.generate(
            encoder_outputs=encoder_outputs, attention_mask=attention_mask, k=max_k
        )

        labels = labels[:, :-1]

        for k in self.ks:
            k_outputs = output_sequences[:, :k, :]
            recall, ndcg = self.calculate_generative_metrics_at_k(k_outputs, labels, k)
            metrics[f"recall_b_{k}"] = recall
            metrics[f"ndcg_b_{k}"] = ndcg

        return metrics

    @property
    def metrics_to_log(self) -> List[str]:
        return [f"recall_h_{max(self.ks)}", f"ndcg_h_{max(self.ks)}", f"recall_b_{max(self.ks)}", f"ndcg_b_{max(self.ks)}"]