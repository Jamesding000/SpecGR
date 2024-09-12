import torch
from torch import nn
from typing import Dict, List, Any, Optional
from utils import repeat_interleave_with_expand
from models.genrec.tokenizer import AbstractTokenizer

# GenRecConfig

# Add the beam search logic here with the placeholder encoder_forward and decoder_forward function
class AbstractGenRec(nn.Module):
    
    def __init__(self, config: Any, tokenizer: AbstractTokenizer):
        """
        Initialize the AbstractGenRec.

        Args:
            config (Any): Configuration object or dictionary containing model parameters.
            tokenizer (Any): Tokenizer object for handling text tokenization.
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
    
    @property
    def hidden_size(self):
        return NotImplementedError
    
    def calculate_loss(self, **kwargs):
        raise NotImplementedError
    
    def encoder_forward(self, **kwargs):
        raise NotImplementedError

    def decoder_forward(self, **kwargs):
        raise NotImplementedError
    
    def __call__(self, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def _prepare_beam_search_inputs(self, batch_size, num_beams, device):
        decoder_input_ids = torch.ones((batch_size * num_beams, 1), device=device, dtype=torch.long)
        initial_decoder_input_ids = decoder_input_ids * self.tokenizer.decoder_start_token_id

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        initial_beam_scores = beam_scores.view((batch_size * num_beams,))

        beam_idx_offset = torch.arange(batch_size, device=device).repeat_interleave(num_beams) * num_beams

        return initial_decoder_input_ids, initial_beam_scores, beam_idx_offset

    @torch.no_grad()
    def beam_search_step(self, logits, decoder_input_ids, beam_scores, beam_idx_offset, batch_size, num_beams):
        assert batch_size * num_beams == logits.shape[0]

        vocab_size = logits.shape[-1]
        next_token_logits = logits[:, -1, :]
        next_token_scores = torch.log_softmax(next_token_logits, dim=-1)

        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_scores = next_token_scores[:, :num_beams].reshape(-1)
        beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
        beam_idx = next_indices[:, :num_beams].reshape(-1)

        decoder_input_ids = torch.cat([decoder_input_ids[beam_idx + beam_idx_offset, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        return decoder_input_ids, beam_scores

    @torch.no_grad()
    def beam_search(self, attention_mask: torch.Tensor, max_length: int, num_beams: int = 1, 
                     num_return_sequences: int = 1, return_score: bool = False, 
                     input_ids: Optional[torch.Tensor] = None, 
                     encoder_outputs: Optional[Dict[str, torch.Tensor]] = None):
        """
        Perform beam search to generate sequences.

        Args:
            attention_mask (torch.Tensor): Tensor representing the attention mask.
            max_length (int): Maximum length of the sequence to be generated.
            num_beams (int): Number of beams for beam search.
            num_return_sequences (int): Number of sequences to return.
            return_score (bool): If True, returns scores along with sequences.
            input_ids (torch.Tensor, optional): Tensor of input ids.
            encoder_outputs (Dict[str, torch.Tensor], optional): Pre-computed encoder outputs.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Generated sequences and optionally their scores.
        """
        assert num_return_sequences <= num_beams, "num_return_sequences cannot be larger than num_beams"
        
        batch_size = attention_mask.shape[0]

        decoder_input_ids, beam_scores, beam_idx_offset = self._prepare_beam_search_inputs(
            batch_size, num_beams, attention_mask.device
        )
        
        if encoder_outputs is None:
            assert input_ids is not None, "Either input_ids or encoder_outputs must be provided"
            input_ids = input_ids.repeat_interleave(num_beams, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
            encoder_outputs = self.encoder_forward(input_ids=input_ids, attention_mask=attention_mask)
        else:
            encoder_outputs['last_hidden_state'] = repeat_interleave_with_expand(encoder_outputs['last_hidden_state'], num_beams, dim=0)
            attention_mask = repeat_interleave_with_expand(attention_mask, num_beams, dim=0)

        while decoder_input_ids.shape[1] < max_length:
            outputs = self.decoder_forward(encoder_outputs=encoder_outputs, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
            decoder_input_ids, beam_scores = self.beam_search_step(outputs.logits, decoder_input_ids, beam_scores, beam_idx_offset, batch_size, num_beams)

        selection_mask = torch.zeros(batch_size, num_beams, dtype=bool)
        selection_mask[:, :num_return_sequences] = True

        if return_score:
            return decoder_input_ids[selection_mask.view(-1), :], beam_scores[selection_mask.view(-1)] / (decoder_input_ids.shape[1] - 1)

        return decoder_input_ids[selection_mask.view(-1), :]
    
    
    @torch.no_grad()
    def generate(self, **model_kwargs):
        """
        Generates sequences using beam search algorithm.

        Args:
            **model_kwargs: A dictionary containing generation parameters and model inputs.
                Required:
                    - 'k' (int): The number of sequences to generate per input.
                    - 'attention_mask' (torch.Tensor): Attention mask for input sequences.
                Optional (one of these must be provided):
                    - 'input_ids' (torch.Tensor): Input token ids.
                    - 'encoder_outputs' (Dict[str, torch.Tensor]): Pre-computed encoder outputs.

        Returns:
            torch.Tensor: The generated sequences.

        Raises:
            ValueError: If neither 'input_ids' nor 'encoder_outputs' is provided.
        """
        k = model_kwargs.pop('k')
        attention_mask = model_kwargs.pop('attention_mask')
        input_ids = model_kwargs.pop('input_ids', None)
        encoder_outputs = model_kwargs.pop('encoder_outputs', None)

        if input_ids is None and encoder_outputs is None:
            raise ValueError("Please provide either 'input_ids' or 'encoder_outputs'.")

        n_digits = self.tokenizer.n_digits

        outputs = self.beam_search(
            attention_mask=attention_mask,
            max_length=n_digits+2,     # the complete decoded sequence: [<bos>, <c_1>, ..., <c_l>, <eos>]
            num_beams=k,
            num_return_sequences=k,
            input_ids=input_ids,
            encoder_outputs=encoder_outputs
        )

        # Reshape outputs to [batch_size, k, n_digits]
        outputs = outputs[:, 1:1+n_digits].reshape(-1, k, n_digits)
        return outputs

