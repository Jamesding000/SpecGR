from abc import ABC, abstractmethod

class AbstractTokenizer(ABC):
    def __init__(self, config: dict):
        self.config = config

    @property
    @abstractmethod
    def n_digits(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def code_book_size(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def max_inputs_length(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def pad_token_id(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def bos_token_id(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def eos_token_id(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def decoder_start_token_id(self):
        raise NotImplementedError
    
    @abstractmethod
    def fit(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def transform(self, **kwargs):
        raise NotImplementedError