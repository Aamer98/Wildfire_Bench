from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from GreeceFire.dataset import GreeceFireDataset
from common_utils.data_utils import collate_fn


class GreeceFireDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """
    def __init__(
            self,
            root_dir: str = None,
            variables: list = None,
            out_variables: list = None,
            predict_range: int = 24,
            hrs_each_step: int = 1,

            batch_size: int = 64,
            num_workers: int = 1,
            pin_memory: bool = False,
            access_mode: str = 'spatial',
            problem_class: str = 'classification',
            nan_fill: float = 0.5,
            sel_dynamic_features=None,
            sel_static_features=None,
            prefetch_factor: int = 2,
            persistent_workers: bool = True,
            clc: str = 'vec'
    ):
        super().__init__()

        self.root_dir = root_dir
        self.variables = variables
        self.out_variables = out_variables
        self.predict_range = predict_range
        self.hrs_each_step = hrs_each_step

        if sel_dynamic_features is None:
            sel_dynamic_features = []
        if sel_static_features is None:
            sel_static_features = []

        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.nan_fill = nan_fill
        self.access_mode = access_mode
        self.problem_class = problem_class
        self.batch_size = batch_size
        self.val_batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sel_dynamic_features = sel_dynamic_features
        self.sel_static_features = sel_static_features

        if not root_dir:
            raise ValueError('dataset_root variable must be set. Check README')

        self.data_train = GreeceFireDataset(dataset_root=root_dir, access_mode=self.access_mode,
                                          problem_class=self.problem_class,
                                          train_val_test='train',
                                          dynamic_features=self.sel_dynamic_features,
                                          static_features=self.sel_static_features,
                                          categorical_features=None, nan_fill=self.nan_fill, clc=clc,
                                          variables=self.variables, out_variables=self.out_variables, 
                                          lead_time=self.predict_range)

        self.data_val = GreeceFireDataset(dataset_root=root_dir, access_mode=self.access_mode,
                                        problem_class=self.problem_class,
                                        train_val_test='val',
                                        dynamic_features=self.sel_dynamic_features,
                                        static_features=self.sel_static_features,
                                        categorical_features=None, nan_fill=self.nan_fill, clc=clc,
                                        variables=self.variables, out_variables=self.out_variables, 
                                        lead_time=self.predict_range)

        self.data_test = GreeceFireDataset(dataset_root=root_dir, access_mode=self.access_mode,
                                         problem_class=self.problem_class,
                                         train_val_test='test',
                                         dynamic_features=self.sel_dynamic_features,
                                         static_features=self.sel_static_features,
                                         categorical_features=None, nan_fill=self.nan_fill, clc=clc,
                                         variables=self.variables, out_variables=self.out_variables, 
                                         lead_time=self.predict_range)                                         

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
            drop_last=False
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
            drop_last=False
        )