import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional, Union
from models.draft.drafter import AbstractDrafter
from models.genrec.genrec import AbstractGenRec
from utils import torch_in, safe_topk
# from models.genrec.beam_search import beam_search_step

class AbstractSpecGR(nn.Module):
    def __init__(self, genrec: AbstractGenRec, config: Dict[str, Any], params: Dict[str, Any]):
        super().__init__()
        self.genrec = genrec
        self.n_digits = self.genrec.tokenizer.n_digits
        self.batch_size = 1  # SpecGR only support batch_size = 1 for now
        self.unseen_start_index = config['unseen_start_index']
        self.draft_size = params['draft_size']
        self.threshold = params['threshold']
        self.num_beams = params['num_beams']
        self._initialize_specGR_inputs(self.num_beams, self.draft_size, self.genrec.device)

    def _initialize_specGR_inputs(self, num_beams: int, draft_size: int, device: torch.device) -> None:
        self.all_decoder_input_ids = torch.zeros(num_beams + draft_size, self.n_digits+1, dtype=torch.long).to(device)
        self.initial_beam_sequences = torch.ones(
            (self.batch_size * num_beams, 1),
            dtype=torch.long,
            device=device
        ) * self.genrec.tokenizer.decoder_start_token_id
        self.initial_beam_scores = torch.zeros(self.batch_size * num_beams, device=device)
        self.initial_beam_scores[1:] = -1e9
        self.beam_idx_offset = (torch.arange(self.batch_size).repeat_interleave(num_beams) * num_beams).to(device=device)

    @torch.no_grad()
    def _specGR_forward(self, encoder_outputs: Dict[str, torch.Tensor], all_attention_mask: torch.Tensor, candidates: torch.Tensor, beam_sequences: torch.Tensor, beam_seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_candidates = candidates.shape[0]
        self.all_decoder_input_ids[:num_candidates, 1:] = candidates
        if beam_seq_length > 0:
            self.all_decoder_input_ids[self.draft_size:, 1:beam_seq_length + 1] = beam_sequences

        all_logits = self.genrec.decoder_forward(
            encoder_outputs=encoder_outputs,
            attention_mask=all_attention_mask,
            decoder_input_ids=self.all_decoder_input_ids,
        ).logits

        candidates_logits = all_logits[:num_candidates, :-1, :]
        beam_search_logits = all_logits[self.draft_size:, :beam_seq_length + 1, :]
        return candidates_logits, beam_search_logits

    def _constrained_draft(self, draft_logits: torch.Tensor, draft_mask: torch.Tensor, k: int) -> torch.Tensor:
        draft_range = torch.nonzero(draft_mask).squeeze()
        if draft_range.numel() < k:
            return draft_range
        else:
            _, indices = torch.topk(draft_logits[0, draft_range], k, dim=-1)
            return draft_range[indices]

    def _verify(self, logits: torch.Tensor, candidates: torch.Tensor, candidates_unseen_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        flat_logits = logits.contiguous().view(-1, logits.size(-1))
        flat_candidates = candidates.view(-1)

        scores = loss_fct(flat_logits, flat_candidates).view_as(candidates)
        scores[:, -1].mul_(~candidates_unseen_mask)
        scores = scores.sum(dim=1)
        scores.div_(1 * candidates_unseen_mask - self.n_digits)
        acceptance_mask = scores > self.threshold
        return acceptance_mask, scores

    def finalize_outputs(self, all_candidates: List[torch.Tensor], all_acceptance_mask: List[torch.Tensor], all_candidates_scores: List[torch.Tensor], num_recommended: int, beam_sequences: torch.Tensor, beam_scores: torch.Tensor, k: int, constraints: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        all_acceptance_mask = torch.cat(all_acceptance_mask, dim=0)
        all_candidates = torch.cat(all_candidates, dim=0)
        all_candidates_scores = torch.cat(all_candidates_scores, dim=0)
        num_accepted = min(num_recommended, k)

        if num_recommended >= k:
            top_k_recommended_items_scores, top_k_indices = torch.topk(all_candidates_scores[all_acceptance_mask], k)
            top_k_recommended_items = all_candidates[all_acceptance_mask][top_k_indices]
        elif constraints is not None:
            top_k_recommended_items_scores, top_k_indices = safe_topk(all_candidates_scores, k)
            top_k_recommended_items = all_candidates[top_k_indices]
        else:
            items_indicies = ~torch_in(beam_sequences, all_candidates)
            items = torch.cat([all_candidates[~all_acceptance_mask], beam_sequences[items_indicies]], dim=0)
            items_scores = torch.cat([all_candidates_scores[~all_acceptance_mask], beam_scores[items_indicies] / 4], dim=0)

            items_scores, items_indices = safe_topk(items_scores, k - num_recommended)
            top_k_recommended_items_scores = torch.cat([all_candidates_scores[all_acceptance_mask], items_scores], dim=0)
            top_k_recommended_items = torch.cat([all_candidates[all_acceptance_mask], items[items_indices]], dim=0)

            top_k_recommended_items_scores, top_k_indices = torch.sort(top_k_recommended_items_scores, descending=True)
            top_k_recommended_items = top_k_recommended_items[top_k_indices]

        return top_k_recommended_items, top_k_recommended_items_scores, num_accepted

    def calculate_draft_logits(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

    @torch.no_grad()
    def recommend(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, k: int, semantic_ids: torch.Tensor, **kwargs) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        return_info = kwargs.get('return_info', None)
        constraints = kwargs.get('constraints', None)

        beam_sequences = self.initial_beam_sequences
        beam_scores = self.initial_beam_scores
        beam_seq_length = num_recommended = round_counter = 0

        semantic_ids_constrained = semantic_ids[constraints] if constraints is not None else semantic_ids
        draft_mask = torch.full((semantic_ids_constrained.shape[0],), True, dtype=bool, device=input_ids.device)

        encoder_outputs = self.genrec.encoder_forward(input_ids=input_ids, attention_mask=attention_mask)
        draft_logits = self.calculate_draft_logits(encoder_outputs=encoder_outputs, attention_mask=attention_mask, **kwargs)

        encoder_outputs['last_hidden_state'] = encoder_outputs['last_hidden_state'].expand(self.num_beams + self.draft_size, -1, -1)
        all_attention_mask = attention_mask.expand(self.num_beams + self.draft_size, -1, -1)

        all_candidates, all_acceptance_mask, all_candidates_scores = [], [], []

        while beam_seq_length < self.n_digits and num_recommended < k:
            if beam_seq_length > 0:
                torch.logical_and(draft_mask, torch_in(semantic_ids_constrained[:, :beam_seq_length], beam_sequences), out=draft_mask)

            draft_indices = self._constrained_draft(draft_logits, draft_mask, self.draft_size)
            draft_mask[draft_indices] = False

            candidates = semantic_ids_constrained[draft_indices].view(-1, self.n_digits)
            candidates_logits, beam_search_logits = self._specGR_forward(encoder_outputs, all_attention_mask, candidates, beam_sequences, beam_seq_length)

            unseen_mask = constraints[draft_indices] > self.unseen_start_index if constraints is not None else draft_indices > self.unseen_start_index
            acceptance_mask, candidates_scores = self._verify(candidates_logits, candidates, unseen_mask)

            all_candidates.append(candidates)
            all_acceptance_mask.append(acceptance_mask)
            all_candidates_scores.append(candidates_scores)
            num_recommended += acceptance_mask.sum().item()

            beam_sequences, beam_scores = self.genrec.beam_search_step(
                beam_search_logits,
                beam_sequences,
                beam_scores,
                self.beam_idx_offset,
                self.batch_size,
                self.num_beams,
            )
            if beam_seq_length == 0:
                beam_sequences = beam_sequences[:, -1].unsqueeze(-1)
            beam_seq_length += 1
            round_counter += 1

        top_k_recommended_items, top_k_recommended_items_scores, num_accepted = self.finalize_outputs(all_candidates, all_acceptance_mask, all_candidates_scores, num_recommended, beam_sequences, beam_scores, k, constraints)

        if return_info:
            runtime_info = {
                'num_accepted': num_accepted,
                'exit_rounds': round_counter
            }
            return top_k_recommended_items, top_k_recommended_items_scores, runtime_info

        return top_k_recommended_items, top_k_recommended_items_scores

class SpecGRAuxForRec(AbstractSpecGR):
    def __init__(self, genrec: AbstractGenRec, draft_model: AbstractDrafter, config: Dict[str, Any], params: Dict[str, Any]):
        super().__init__(genrec, config, params)
        self.draft_model = draft_model
        
    def calculate_draft_logits(self, **kwargs) -> torch.Tensor:
        return self.draft_model.score(**kwargs)

class SpecGRForRec(AbstractSpecGR):
    def __init__(self, model: nn.Module, config: Dict[str, Any], params: Dict[str, Any]):
        super().__init__(model, config, params)
        self.specGR_config = config['SpecGR']
        self.hidden_size = model.config['d_model']
        self.encoder_batch_size = self.specGR_config["encoder_batch_size"]

        projection_dim = self.specGR_config.get("projection", self.hidden_size)
        self.projection = (
            torch.nn.Linear(self.hidden_size, int(projection_dim))
            if projection_dim else None
        )

    def calculate_draft_logits(self, encoder_outputs: Dict[str, torch.Tensor], attention_mask: torch.Tensor, test_item_embs: torch.Tensor, constraints: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        item_embeddings_constrained = test_item_embs[constraints] if constraints is not None else test_item_embs
        sequence_embeddings = self.encoder_outputs_to_embedding(encoder_outputs, attention_mask)
        return torch.matmul(sequence_embeddings, item_embeddings_constrained.transpose(0, 1))

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = self.encoder_batch_size
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.long).to(input_ids.device)
        all_embeddings = []

        for start_index in range(0, len(input_ids), batch_size):
            mini_batch_input_ids = input_ids[start_index : start_index + batch_size]
            mini_batch_attention_mask = attention_mask[start_index : start_index + batch_size]

            encoder_outputs = self.genrec.encoder_forward(input_ids=mini_batch_input_ids, attention_mask=mini_batch_attention_mask)
            embeddings = self.encoder_outputs_to_embedding(encoder_outputs, mini_batch_attention_mask)
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def encoder_outputs_to_embedding(self, encoder_outputs: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        output_hidden_state = encoder_outputs["last_hidden_state"]
        if self.projection:
            output_hidden_state = self.projection(output_hidden_state)
        embeddings = self.mean_pooling(output_hidden_state, attention_mask)
        return F.normalize(embeddings, dim=-1)

    @torch.no_grad()
    def mean_pooling(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        embedding = s / d
        return embedding
