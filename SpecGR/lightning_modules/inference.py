import torch
import torch.distributed as dist
from torch import nn
import lightning as L
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.table import Table
from models.specGR.specGR_inference import SpecGRForRec
from evaluator import SpecGREvaluator

class SpecGRForRecLightningModule(L.LightningModule):
    def __init__(
        self,
        model: SpecGRForRec,
        config: Dict[str, Any],
        params: Dict[str, Any],
        saved_model_path: str,
        saved_embedding_path: str,
        semantic_ids: torch.Tensor,
        evaluator: SpecGREvaluator
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.params = params
        self.saved_model_path = saved_model_path
        self.saved_embedding_path = saved_embedding_path
        self.unseen_start_index = config['unseen_start_index']
        self.semantic_ids = semantic_ids
        self.semantic_id_sequences = self._prepare_semantic_id_sequences(semantic_ids)
        self.evaluator = evaluator
        self.test_outputs: List[Dict[str, torch.Tensor]] = []
        
    def _prepare_semantic_id_sequences(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat((
            torch.full((semantic_ids.shape[0], 1), self.model.genrec.tokenizer.bos_token_id),
            semantic_ids,
            torch.full((semantic_ids.shape[0], 1), self.model.genrec.tokenizer.eos_token_id),
        ), dim=1).requires_grad_(False)

    def init_item_embeddings(self, embeddings: Optional[torch.Tensor] = None) -> None:
        embedding_dim = (
            self.model.projection.out_features
            if self.model.projection
            else self.model.genrec.config['d_model']
        )
        
        if embeddings is not None:
            self.item_embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True).to(self.device)
            print('Initialized item embeddings from vectors.')
        else:
            self.item_embeddings = nn.Embedding(
                num_embeddings=self.semantic_id_sequences.shape[0],
                embedding_dim=embedding_dim,
            ).to(self.device)
            print('Initialized empty item embeddings.')
        
        self.item_embeddings.eval()
        self.item_embeddings.requires_grad_(False)

    def _load_checkpoint(self) -> None:
        self.model.load_state_dict(torch.load(self.saved_model_path))
        print(f'Loaded model checkpoint from {self.saved_model_path}.')

        self.init_item_embeddings()
        self.item_embeddings.load_state_dict(torch.load(self.saved_embedding_path))
        print(f'Loaded item embeddings from {self.saved_embedding_path}.')

        self.item_embeddings.eval()
        self.item_embeddings.requires_grad_(False)

    def setup(self, stage: str) -> None:
        self.model = self.model.to(self.device)
        self.semantic_ids = self.semantic_ids.to(self.device)
        self.semantic_id_sequences = self.semantic_id_sequences.to(self.device)
        
        self.model._initialize_specGR_inputs(self.params["num_beams"], self.params["draft_size"], self.device)

        self._print_params_to_console()

    def _print_params_to_console(self) -> None:
        if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
            console = Console()
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("SpecGR Params", style="dim")
            table.add_column("Value")

            for key, value in self.params.items():
                table.add_row(key, str(value))

            console.print(table)

    def on_test_epoch_start(self) -> None:
        self.test_outputs.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        metrics = self.evaluator.evaluation_step(batch, self.device, semantic_ids=self.semantic_ids, test_item_embs=self.item_embeddings.weight)
        metrics = self.evaluator.convert_metrics_to_tensor(metrics, self.device)
        self.test_outputs.append(metrics)
        
        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        # if (batch_idx + 1) % 1000 == 0:
        #     avg_metrics = self.evaluator.process_evaluation_result(self.test_outputs)
        #     print(f'Average Recall@50 after {batch_idx + 1} steps: {avg_metrics}')

        return metrics

    def on_test_epoch_end(self) -> None:
        avg_metrics = self.evaluator.process_evaluation_result(self.test_outputs)
        avg_metrics = self.evaluator.convert_metrics_to_tensor(avg_metrics, self.device)
        self.log_dict(avg_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)