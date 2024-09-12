from typing import List, Dict
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from utils import load_dataset_splits
from dataloader import TIGERDataProcessor

class SpecGRPretrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        domain: str,
        splits: List[str],
        data_processor: TIGERDataProcessor,
        emb_batch_size: int,
        gen_batch_size: int,
        eval_batch_size: int,
        num_workers: int
    ):
        super().__init__()
        self.domain = domain
        self.splits = splits
        self.data_processor = data_processor
        self.num_workers = num_workers
        self.emb_batch_size = emb_batch_size
        self.gen_batch_size = gen_batch_size
        self.eval_batch_size = eval_batch_size
        self.datasets = {}

    def setup(self, stage: str = None) -> None:
        datasets = load_dataset_splits(self.domain, self.splits)
        
        for split in self.splits:
            result_dataset = datasets[split].map(
                self.data_processor,
                num_proc=self.num_workers,
                remove_columns=datasets[split].column_names,
            )
            result_dataset.set_format(type="torch")
            self.datasets[split] = result_dataset

    def train_dataloader(self) -> CombinedLoader:
        return CombinedLoader(
            {
                "embedding": DataLoader(self.datasets['train'], batch_size=self.emb_batch_size, shuffle=True),
                "generative": DataLoader(self.datasets['train'], batch_size=self.gen_batch_size, shuffle=True)
            },
            mode='max_size_cycle'
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['valid'],
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['test'],
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

class SpecGRFinetuneDataModule(L.LightningDataModule):
    def __init__(
        self,
        domain: str,
        splits: List[str],
        data_processor: TIGERDataProcessor,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int
    ):
        super().__init__()
        self.domain = domain
        self.splits = splits
        self.data_processor = data_processor
        self.num_workers = num_workers
        self.gen_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.datasets = {}

    def setup(self, stage: str = None) -> None:
        datasets = load_dataset_splits(self.domain, self.splits)
        
        for split in self.splits:
            result_dataset = datasets[split].map(
                self.data_processor,
                num_proc=self.num_workers,
                remove_columns=datasets[split].column_names,
            )
            result_dataset.set_format(type="torch")
            self.datasets[split] = result_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['train'],
            batch_size=self.gen_batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['valid'],
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['test'],
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )