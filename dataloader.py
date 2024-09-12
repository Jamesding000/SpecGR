from typing import List, Dict, Any, Tuple
import numpy as np
from torch.utils.data import DataLoader
from utils import load_dataset_splits

from abc import ABC, abstractmethod

class AbstractDataProcessor(ABC):
    @abstractmethod
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        pass

class UniSRecDataProcessor(AbstractDataProcessor):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        input_sequence = list(map(int, example['history'].split()))
        label = example['item_id']
        item_seq_length = len(input_sequence)
        input_ids = np.array(input_sequence + [0] * (self.max_length - item_seq_length), dtype=int)
        return {
            'input_ids': input_ids,
            'length': item_seq_length,
            'labels': label
        }

class TIGERDataProcessor(AbstractDataProcessor):
    def __init__(self, max_length: int, tokenizer: Any):
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        input_sequence = list(map(int, example['history'].split()))
        item_id = example['item_id']
        return self.tokenizer.tokenize(input_sequence, item_id)
    
class SpecGRDataProcessor(AbstractDataProcessor):
    def __init__(self, max_length: int, tokenizer: Any):
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        input_sequence = list(map(int, example['history'].split()))
        item_id = example['item_id']
        
        inputs = self.tokenizer.tokenize(input_sequence, item_id)
        inputs['item_id'] = item_id
        return inputs
    
def get_dataloaders(
    domain: str,
    splits: List[str],
    train_batch_size: int,
    eval_batch_size: int,
    data_processor: AbstractDataProcessor,
    num_workers: int = 1
) -> Tuple[DataLoader, ...]:
    
    datasets = load_dataset_splits(domain, splits)

    dataloaders = []
    
    for split in splits:
        result_dataset = datasets[split].map(
            data_processor,
            num_proc=num_workers,
            remove_columns=datasets[split].column_names,  # Automatically remove all columns except the ones returned by the function
        )
        result_dataset.set_format(type="torch")
    
        if split == 'train':
            dataloader = DataLoader(result_dataset, batch_size=train_batch_size, shuffle=True)
        else:
            dataloader = DataLoader(result_dataset, batch_size=eval_batch_size, shuffle=False)
            
        dataloaders.append(dataloader)

    return tuple(dataloaders)
