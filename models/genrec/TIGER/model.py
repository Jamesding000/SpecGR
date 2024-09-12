from transformers import T5Config, T5ForConditionalGeneration
from models.genrec.TIGER.tokenizer import TIGERTokenizer
from models.genrec.genrec import AbstractGenRec
  
class TIGER(AbstractGenRec):
    def __init__(self, config, tokenizer: TIGERTokenizer):
        
        super().__init__(config, tokenizer)  # Initialize the parent AbstractGenRec class
        
        t5config = T5Config(
            num_layers=config['n_layers'], 
            num_decoder_layers=config['n_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            activation_function=config['activation_function'],
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.decoder_start_token_id,
            feed_forward_proj=config["feed_forward_proj"],
            n_positions=config["n_positions"],
        )
        
        self.t5 = T5ForConditionalGeneration(config=t5config)
        self.config = config
        
    @property
    def hidden_size(self) -> str:
        return self.t5.config.hidden_size
    
    @property
    def device(self) -> str:
        return self.t5.device
    
    def calculate_loss(self, **kwargs):
        """
        Calculates the loss for a given batch of data.

        Args:
            batch (dict): A dictionary containing the input data for the model.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.t5(**kwargs).loss

    def encoder_forward(self, input_ids, attention_mask):
        encoder_outputs = self.t5.get_encoder()(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True
        )
        return encoder_outputs

    def decoder_forward(self, encoder_outputs, attention_mask, decoder_input_ids):
        decoder_outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        return decoder_outputs
    
    def __call__(self, **model_kwargs):
        outputs = self.t5(**model_kwargs)
        return outputs
