import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional, Union
from models.draft.drafter import AbstractDrafter
from models.genrec.genrec import AbstractGenRec
from utils import torch_in, safe_topk

from models.SpecGR.utils import *
from utils import repeat_interleave_with_expand

class AbstractSpecGR(nn.Module):
    """
    Abstract base class for SpecGR framework (Section 3.2).
    Implements speculative generative recommendation with 4 key components:
    1. Inductive Drafting, 2. Target-aware Verifying, 3. Guided Re-drafting, 4. Adaptive Exiting
    """
    def __init__(self, genrec: AbstractGenRec, config: Dict[str, Any], semantic_ids, params: Dict[str, Any]):
        super().__init__()
        self.genrec = genrec
        self.num_digits = self.genrec.tokenizer.n_digits

        self.unseen_start_index = config['unseen_start_index']
        self.draft_size = params['draft_size']
        self.threshold = params['threshold']
        self.num_beams = params['num_beams']

        # Cache for beam initialization
        self.cached_batch_size = None
        self.cached_beam_sequences = None
        self.cached_beam_scores = None
        self.cached_beam_idx_offset = None
        self.cached_all_decoder_input_ids = None

        all_prefix_lengths = calculate_in_sample_prefix_lengths(
            semantic_ids, 
            self.unseen_start_index
        )
        
        if self.__class__.__name__ == "SpecGRForRec":
            # if self-drafting is enabled, we only ignore the last identifier when calculating the score
            all_prefix_lengths = torch.clamp_min(all_prefix_lengths, self.num_digits - 1)
        else:  # clamp to 2 for SpecGRAuxForRec
            all_prefix_lengths = torch.clamp_min(all_prefix_lengths, 2)  # at least 2 matching
        
        print(all_prefix_lengths)
        
        self.register_buffer(
            "all_prefix_lengths", all_prefix_lengths, persistent=False
        )  # will not be saved or loaded

    def update_params(self, params):
        self.draft_size = params.get('draft_size', self.draft_size)
        self.threshold = params.get('threshold', self.threshold)
        self.num_beams = params.get('num_beams', self.num_beams)
        
        # reset cached tensors as model params have changed
        self.cached_batch_size = None
        self.cached_beam_sequences = None
        self.cached_beam_scores = None
        self.cached_beam_idx_offset = None
        self.cached_all_decoder_input_ids = None

    @torch.no_grad()
    def specGR_forward(
        self,
        all_last_hidden_state: Dict[str, torch.Tensor],
        all_attention_mask: torch.Tensor,
        candidates: torch.Tensor,
        beam_sequences: torch.Tensor,
        beam_seq_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform forward pass for SpecGR with cached decoder inputs.

        Args:
            all_last_hidden_state (Dict[str, torch.Tensor]): Last hidden states from the encoder.
            all_attention_mask (torch.Tensor): Attention mask for the encoder.
            candidates (torch.Tensor): Candidate sequences.
            beam_sequences (torch.Tensor): Beam sequences.
            beam_seq_length (int): Length of the beam sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits for candidates and beam sequences.
        """
        batch_draft_size = self.draft_size * self.cached_batch_size

        self.cached_all_decoder_input_ids[:batch_draft_size, 1:] = candidates
        if beam_seq_length > 0:
            self.cached_all_decoder_input_ids[batch_draft_size:, :beam_seq_length + 1] = beam_sequences

        all_logits = self.genrec.decoder_forward(
            encoder_outputs=[all_last_hidden_state],
            attention_mask=all_attention_mask,
            decoder_input_ids=self.cached_all_decoder_input_ids,
        ).logits

        candidates_logits = all_logits[:batch_draft_size, :-1, :]
        beam_search_logits = all_logits[batch_draft_size:, :beam_seq_length + 1, :]

        return candidates_logits, beam_search_logits

    @torch.no_grad()
    def recommend(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        k: int,
        semantic_ids: torch.Tensor,
        **kwargs
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]],
    ]:
        """
        Main SpecGR inference loop implementing the 4-component framework (Section 3.2):
        1. Inductive Drafting -> 2. Target-aware Verifying -> 3. Guided Re-drafting -> 4. Adaptive Exiting
        """
        batch_size = input_ids.shape[0]
        return_info = kwargs.get('return_info', None)

        # Check if we need to reinitialize beam variables
        if self.cached_batch_size != batch_size:
            # Initialize or reinitialize beam variables if batch size has changed
            self.cached_batch_size = batch_size
            self.cached_beam_sequences, self.cached_beam_scores, self.cached_beam_idx_offset = prepare_beam_search_inputs(self.cached_batch_size, self.num_beams, input_ids.device)
            self.cached_all_decoder_input_ids = torch.zeros(
                self.cached_batch_size * (self.num_beams + self.draft_size),
                self.num_digits + 1,
                dtype=torch.long,
            ).to(input_ids)

        beam_sequences, beam_scores, beam_idx_offset = self.cached_beam_sequences, self.cached_beam_scores, self.cached_beam_idx_offset

        # Encoder forward pass
        encoder_outputs = self.genrec.encoder_forward(input_ids=input_ids, attention_mask=attention_mask)
        # 1. Inductive Drafting (Section 3.2.1): Drafter proposes δ candidates Q = D(X)
        draft_logits = self.calculate_draft_logits(encoder_outputs=encoder_outputs, attention_mask=attention_mask, **kwargs)

        all_last_hidden_state = torch.cat([
            repeat_interleave_with_expand(encoder_outputs["last_hidden_state"], self.draft_size, dim=0),
            repeat_interleave_with_expand(encoder_outputs["last_hidden_state"], self.num_beams, dim=0)
        ], dim=0)

        all_attention_mask = torch.cat([
            repeat_interleave_with_expand(attention_mask, self.draft_size, dim=0),
            repeat_interleave_with_expand(attention_mask, self.num_beams, dim=0)
        ], dim=0)

        # Initialize status tracking variables
        beam_seq_length, iteration_count = 0, 0  # beam_seq_length keep track of length of beam_sequence except for the root token (i.e. 0)
        all_candidate_sequences, all_acceptance_masks, all_candidate_scores = [], [], []
        num_recommended = torch.zeros(batch_size, dtype=int, device=input_ids.device)

        # Main draft-verify loop implementing guided re-drafting and adaptive exiting
        while beam_seq_length < self.num_digits and not torch.all(num_recommended >= k):
            # 3. Guided Re-drafting (Section 3.2.3): Constrain candidates using beam prefixes (Equation 3)
            if beam_seq_length > 1:
                valid_mask = torch_in(semantic_ids[:, :beam_seq_length], beam_sequences[:, 1:])
                draft_logits = torch.where(valid_mask, draft_logits, float('-inf'))  # Set to -inf if not valid

            draft_indices = constrained_draft(draft_logits, self.draft_size).flatten()
            # print('Drafted unseen proportions:', torch.sum((draft_indices > self.unseen_start_index)) / self.draft_size / batch_size)
            candidate_sequences = semantic_ids[draft_indices]  # Shape: (batch_size, draft_size, codebook_length)

            # 2. Target-aware Verifying (Section 3.2.2): GR model verifies candidates
            candidate_logits, beam_logits = self.specGR_forward(
                all_last_hidden_state,
                all_attention_mask,
                candidate_sequences,
                beam_sequences,
                beam_seq_length,
            )

            # Calculate verification scores using target-aware likelihood (Equation 2, Section 3.2)
            candidate_scores = calculate_masked_score(candidate_logits, candidate_sequences, draft_indices, self.all_prefix_lengths)
            acceptance_mask = candidate_scores > self.threshold  # Accept if V(xt, X) > γ

            all_candidate_sequences.append(candidate_sequences.view(batch_size, self.draft_size, self.num_digits))
            all_acceptance_masks.append(acceptance_mask.view(batch_size, self.draft_size))
            all_candidate_scores.append(candidate_scores.view(batch_size, self.draft_size))

            num_recommended += acceptance_mask.view(batch_size, self.draft_size).sum(dim=-1)

            # Generate beam sequences for next iteration's guided re-drafting
            beam_sequences, beam_scores = self.genrec.beam_search_step(
                beam_logits,
                beam_sequences,
                beam_scores,
                beam_idx_offset,
                batch_size,
                self.num_beams,
            )

            beam_seq_length += 1
            iteration_count += 1

        # 4. Adaptive Exiting (Section 3.2.4): Exit when K items accepted or max iterations reached
        # Concatenate inputs across iterations
        all_candidate_sequences = torch.cat(all_candidate_sequences, dim=1)  # Shape: (batch_size, total_beams, codebook_length)
        all_acceptance_masks = torch.cat(all_acceptance_masks, dim=1)  # Shape: (batch_size, total_beams)
        all_candidate_scores = torch.cat(all_candidate_scores, dim=1)  # Shape: (batch_size, total_beams)

        beam_sequences = beam_sequences.view(batch_size, self.num_beams, beam_seq_length + 1)[:, :, 1:]  # Ignore the 0th column
        beam_scores = beam_scores.view(batch_size, self.num_beams) / beam_seq_length

        recommended_items, recommended_scores, num_accepted = finalize_batch_recommendation(
            all_candidate_sequences,
            all_acceptance_masks,
            all_candidate_scores,
            num_recommended,
            beam_sequences,
            beam_scores,
            k,
        )

        if return_info:
            runtime_info = {
                'num_accepted': num_accepted,
                'exit_rounds': iteration_count
            }
            return recommended_items, recommended_scores, runtime_info

        return recommended_items, recommended_scores

    def calculate_draft_logits(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")


class SpecGRAuxForRec(AbstractSpecGR):
    """
    SpecGR with Auxiliary Model as Drafter (Section 3.3).
    Uses external inductive model (e.g., UniSRec) for drafting.
    """
    def __init__(self, genrec: AbstractGenRec, draft_model: AbstractDrafter, config: Dict[str, Any], semantic_ids, params: Dict[str, Any]):
        super().__init__(genrec, config, semantic_ids, params)
        self.draft_model = draft_model
        
    def calculate_draft_logits(self, **kwargs) -> torch.Tensor:
        return self.draft_model.score(**kwargs)

class SpecGRForRec(AbstractSpecGR):
    """
    SpecGR++ with Self-Speculative Generative Recommendation (Section 3.3).
    Reuses GR model encoder for inductive drafting via KNN search.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any], semantic_ids, params: Dict[str, Any]):
        super().__init__(model, config, semantic_ids, params)
        self.specGR_config = config['SpecGR']
        self.hidden_size = model.config['d_model']
        self.encoder_batch_size = self.specGR_config["encoder_batch_size"]

        self.projection_dim = self.specGR_config.get("projection")
        self.projection = (
            nn.Linear(self.hidden_size, self.projection_dim)
            if self.projection_dim is not None
            else None
        )
        # self.projection = (
        #     nn.Sequential(
        #         nn.Linear(self.hidden_size, self.projection_dim),
        #         nn.ReLU(),  # Non-linear activation, SimCLR
        #         nn.Linear(self.projection_dim, self.projection_dim)
        #     ) if self.projection_dim is not None else None
        # )  # non-linear projection, no signficant improvement than linear

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
