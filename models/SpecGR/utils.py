import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional, Union
from models.draft.drafter import AbstractDrafter
from models.genrec.genrec import AbstractGenRec
from utils import torch_in, safe_topk


def prepare_beam_search_inputs(batch_size, num_beams, device):
    decoder_input_ids = torch.ones((batch_size * num_beams, 1), device=device, dtype=torch.long)
    initial_decoder_input_ids = decoder_input_ids * 0

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
    beam_scores[:, 1:] = -1e9
    initial_beam_scores = beam_scores.view((batch_size * num_beams,))

    beam_idx_offset = torch.arange(batch_size, device=device).repeat_interleave(num_beams) * num_beams

    return initial_decoder_input_ids, initial_beam_scores, beam_idx_offset


def constrained_draft(draft_logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Selects top-k candidates from draft_logits for each batch and sets the selected logits to -inf 
    to prevent them from being drafted again.

    Args:
        draft_logits (torch.Tensor): Logits for candidate drafts of shape (batch_size, num_items).
        k (int): Number of candidates to select.

    Returns:
        torch.Tensor: Indices of the top-k selected candidates for each batch, shape (batch_size, k).
    """
    # Get the top-k indices for each batch
    _, top_k_indices = torch.topk(draft_logits, k, dim=1)

    # Scatter -inf into draft_logits at the selected indices
    batch_size = draft_logits.size(0)
    batch_indices = torch.arange(batch_size, device=draft_logits.device).unsqueeze(1)  # Shape: (batch_size, 1)
    draft_logits[batch_indices, top_k_indices] = float('-inf')

    return top_k_indices


def calculate_masked_score(logits, candidates, candidate_idx, all_prefix_lengths):
    """
    Calculates scores considering only up to the number of token positions specified by all_prefix_lengths.

    Args:
        logits (torch.Tensor): Logits of shape (batch_size * K, sequence_length, vocab_size).
        candidates (torch.Tensor): Candidate tokens of shape (batch_size * K, sequence_length).
        candidate_idx (torch.Tensor): Indices of the candidates in semantic_ids.
        all_prefix_lengths (torch.Tensor): Tensor of shape (n_item,) specifying the prefix lengths to consider.

    Returns:
        torch.Tensor: Scores for each candidate, adjusted for the prefix lengths.
    """
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    flat_logits = logits.contiguous().view(-1, logits.size(-1))
    flat_candidates = candidates.view(-1)

    # Compute per-token loss
    token_losses = loss_fct(flat_logits, flat_candidates).view_as(candidates)

    # Mask to select only the tokens up to the prefix length
    batch_size, sequence_length = candidates.shape
    prefix_lengths = all_prefix_lengths[candidate_idx.view(-1)]  # Shape: (batch_size * K,)
    mask = torch.arange(sequence_length, device=logits.device).unsqueeze(0) < prefix_lengths.unsqueeze(1)

    # Apply the mask and compute the mean loss per candidate
    masked_losses = token_losses * mask
    masked_mean_losses = -(masked_losses.sum(dim=1) + 1e-9) / (prefix_lengths + 1e-10)  # if no matching prefix, return -10
    
    return masked_mean_losses


def calculate_in_sample_prefix_lengths(semantic_ids, unseen_start_index):
    
    in_sample_items = semantic_ids[1:unseen_start_index+1] # padding row
    unseen_items = semantic_ids[unseen_start_index+1:]
    
    # Precompute hashes for all prefixes of in-sample items
    prefix_set = set()
    for item in in_sample_items:
        for prefix_length in range(1, item.size(0) + 1):  # Prefix lengths from 1 to codebook_length
            prefix_tuple = tuple(item[:prefix_length].tolist())  # Convert prefix to tuple (hashable)
            prefix_set.add(prefix_tuple)
    
    # Calculate the longest prefix length for each unseen item
    longest_prefix_lengths = torch.zeros(unseen_items.size(0), dtype=torch.long)
    for i, unseen_item in enumerate(unseen_items):
        max_prefix_length = 0
        for prefix_length in range(1, unseen_item.size(0) + 1):  # Prefix lengths from 1 to codebook_length
            prefix_tuple = tuple(unseen_item[:prefix_length].tolist())  # Convert prefix to tuple
            if prefix_tuple in prefix_set:  # Check existence in precomputed prefix set
                max_prefix_length = prefix_length
            else:
                break  # Stop if no match is found for the current prefix length
        longest_prefix_lengths[i] = max_prefix_length
    
    # Result
    all_prefix_lengths = torch.full(size=(unseen_start_index+1,), fill_value=semantic_ids.shape[1], dtype=torch.long)
    all_prefix_lengths = torch.cat([all_prefix_lengths, longest_prefix_lengths]).to(semantic_ids)

    return all_prefix_lengths


def torch_in(query_tensor: torch.Tensor, reference_tensor: torch.Tensor) -> torch.Tensor:
    """
    Check if rows in query_tensor are present in reference_tensor.

    Args:
        query_tensor (torch.Tensor): A tensor of shape (num_items, codebook_length).
        reference_tensor (torch.Tensor): A tensor of shape 
            - (batch_size, num_beams, codebook_length), or
            - (num_beams, codebook_length).

    Returns:
        torch.Tensor: A boolean tensor of size
            - (batch_size, num_items) if reference_tensor is 3D, or
            - (num_items,) if reference_tensor is 2D.
    """
    
    # Handle the case where reference_tensor is 2D
    if reference_tensor.dim() == 2:
        # Expand query_tensor for broadcasting
        query_tensor = query_tensor.unsqueeze(1)  # Shape: (num_items, 1, codebook_length)
        reference_tensor = reference_tensor.unsqueeze(0)  # Shape: (1, num_beams, codebook_length)
        matches = (query_tensor == reference_tensor).all(dim=2)  # Shape: (num_items, num_beams)
        return matches.any(dim=1)  # Shape: (num_items,)

    # Handle the case where reference_tensor is 3D
    elif reference_tensor.dim() == 3:
        # Expand query_tensor for broadcasting
        query_tensor = query_tensor.unsqueeze(0).unsqueeze(2)  # Shape: (1, num_items, 1, codebook_length)
        reference_tensor = reference_tensor.unsqueeze(1)  # Shape: (batch_size, 1, num_beams, codebook_length)
        matches = (query_tensor == reference_tensor).all(dim=3)  # Shape: (batch_size, num_items, num_beams)
        return matches.any(dim=2)  # Shape: (batch_size, num_items)

    else:
        raise ValueError("reference_tensor must be 2D or 3D")


def finalize_recommendation(
    candidates: List[torch.Tensor],
    acceptance_mask: List[torch.Tensor],
    candidate_scores: List[torch.Tensor],
    num_recommended: int,
    beam_sequences: torch.Tensor,
    beam_scores: torch.Tensor,
    top_k: int,
    constraints: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Finalize recommendations by selecting the top-k scoring candidates.

    Args:
        candidate_tensors (List[torch.Tensor]): List of candidate sequences, each of shape (total_candidates, codebook_length).
        acceptance_masks (List[torch.Tensor]): List of boolean masks indicating accepted candidates, shape (total_candidates,).
        candidate_scores (List[torch.Tensor]): List of scores corresponding to candidates, shape (total_candidates,).
        num_recommended (int): Number of candidates already recommended.
        beam_sequences (torch.Tensor): Beam sequences of shape (num_beams, codebook_length).
        beam_scores (torch.Tensor): Scores associated with the beam sequences, shape (num_beams,).
        top_k (int): Number of top items to recommend.
        constraints (Optional[torch.Tensor]): Constraints for recommendation (unused here).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, int]: 
            - Recommended items of shape (top_k, codebook_length),
            - Recommended scores of shape (top_k,),
            - Number of accepted candidates (int).
    """

    # Compute the number of accepted candidates
    accepted_candidates = candidates[acceptance_mask]
    accepted_scores = candidate_scores[acceptance_mask]

    if num_recommended < top_k:
        # Find top scoring additional sequences from rejected and beam sequences
        items_indicies = ~torch_in(beam_sequences, candidates)
        items = torch.cat([candidates[~acceptance_mask], beam_sequences[items_indicies]], dim=0)
        items_scores = torch.cat([candidate_scores[~acceptance_mask], beam_scores[items_indicies]], dim=0)
        
        # Concatenate additional sequences with accepted candidates
        items_scores, items_indices = safe_topk(items_scores, top_k - num_recommended)
        accepted_scores = torch.cat([accepted_scores, items_scores], dim=0)
        accepted_candidates = torch.cat([accepted_candidates, items[items_indices]], dim=0)
        
    # Select the top-k scoring candidates
    # top_k_scores, top_k_indices = torch.topk(accepted_scores, top_k)
    # top_k_candidates = accepted_candidates[top_k_indices]
    
    # Select the first top_k candidates
    top_k_candidates = accepted_candidates[:top_k]
    top_k_scores = accepted_scores[:top_k]
    
    sort_by_scores = True
    if sort_by_scores:
        top_k_scores, top_k_indices = torch.sort(top_k_scores, descending=True)
        top_k_candidates = top_k_candidates[top_k_indices]

    # Compute the number of accepted candidates that were recommended
    num_accepted = min(num_recommended, top_k)

    return top_k_candidates, top_k_scores, num_accepted


def finalize_batch_recommendation(
    all_candidates: List[torch.Tensor],
    all_acceptance_mask: List[torch.Tensor],
    all_candidates_scores: List[torch.Tensor],
    num_recommended: torch.Tensor,
    beam_sequences: torch.Tensor,
    beam_scores: torch.Tensor,
    k: int,
    constraints: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finalize recommendations for all samples in a batch.

    Args:
        all_candidates (List[torch.Tensor]): List of candidate sequences for each batch, shape (batch_size, total_beams, codebook_length).
        all_acceptance_mask (List[torch.Tensor]): List of acceptance masks for each batch, shape (batch_size, total_beams).
        all_candidates_scores (List[torch.Tensor]): List of scores for candidates, shape (batch_size, total_beams).
        num_recommended (torch.Tensor): Number of candidates already recommended for each batch, shape (batch_size,).
        beam_sequences (torch.Tensor): Beam sequences of shape (batch_size, num_beams, codebook_length).
        beam_scores (torch.Tensor): Beam scores of shape (batch_size, num_beams).
        k (int): Number of top items to recommend for each batch.
        constraints (Optional[torch.Tensor]): Constraints for recommendation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            - Recommended items of shape (batch_size, k, codebook_length),
            - Recommended scores of shape (batch_size, k),
            - Number of accepted candidates for each batch, shape (batch_size,).
    """
    # Prepare outputs
    batch_size = all_candidates.size(0)
    recommended_items = []
    recommended_scores = []
    num_accepted_list = []

    # Loop through each sample in the batch
    for i in range(batch_size):
        # print(all_candidates[i].shape, all_acceptance_mask[i].shape, all_candidates_scores[i].shape, beam_sequences[i].shape, beam_scores[i].shape)
        top_k_candidates, top_k_scores, num_accepted = finalize_recommendation(
            candidates=all_candidates[i],
            acceptance_mask=all_acceptance_mask[i],
            candidate_scores=all_candidates_scores[i],
            num_recommended = num_recommended[i],
            beam_sequences=beam_sequences[i],
            beam_scores=beam_scores[i],
            top_k=k,
        )
        recommended_items.append(top_k_candidates)
        recommended_scores.append(top_k_scores)
        num_accepted_list.append(num_accepted)

    # Stack results into batch outputs
    recommended_items = torch.stack(recommended_items, dim=0)  # Shape: (batch_size, k, codebook_length)
    recommended_scores = torch.stack(recommended_scores, dim=0)  # Shape: (batch_size, k)
    num_accepted = torch.tensor(num_accepted_list, device=beam_sequences.device)  # Shape: (batch_size,)

    return recommended_items, recommended_scores, num_accepted
