from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import numpy as np
import xarray as xr

from SeasFire.dataset import sample_dataset, SeasFireDataset, collate_fn


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class SeasFireDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

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
            root_dir,
            variables,
            positional_vars: list = None,
            target: str = 'gwis_ba',
            target_shift: int = 1,
            task: str = 'classification',
            debug: bool = False,
            predict_range: int = 192,
            hrs_each_step: int = 1,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.variables = list(variables)
        self.target = target
        self.target_shift = target_shift
        self.ds = xr.open_zarr(root_dir, consolidated=True)
        self.ds['sst'] = self.ds['sst'].where(self.ds['sst'] >= 0)
        self.mean_std_dict = None
        self.positional_vars = positional_vars
        if debug:
            self.num_timesteps = 5
        else:
            self.num_timesteps = -1
        self.task = task
        self.predict_range = predict_range
        self.hrs_each_step = hrs_each_step
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            print(self.ds[self.variables])
            # IMPORTANT! Call sample_dataset with ds.copy(). xarray Datasets are mutable
            train_batches, self.mean_std_dict = sample_dataset(self.ds.copy(), input_vars=self.variables,
                                                               target=self.target,
                                                               target_shift=-self.target_shift, split='train',
                                                               num_timesteps=self.num_timesteps)
            
            # Save mean std dict for normalization during inference time
            print(self.mean_std_dict)
            with open(f'mean_std_dict_{self.target_shift}.json', 'w') as f:
                f.write(json.dumps(self.mean_std_dict))

            val_batches, _ = sample_dataset(self.ds.copy(), input_vars=self.variables, target=self.target,
                                            target_shift=-self.target_shift, split='val',
                                            num_timesteps=self.num_timesteps)
            test_batches, _ = sample_dataset(self.ds.copy(), input_vars=self.variables, target=self.target,
                                             target_shift=-self.target_shift, split='test',
                                             num_timesteps=self.num_timesteps)

            self.data_train = SeasFireDataset(train_batches, input_vars=self.variables, positional_vars=self.positional_vars,
                                        target=self.target, lead_time=self.predict_range,
                                        mean_std_dict=self.mean_std_dict, task=self.task)
            self.data_val = SeasFireDataset(val_batches, input_vars=self.variables, positional_vars=self.positional_vars,
                                      target=self.target, lead_time=self.predict_range,
                                      mean_std_dict=self.mean_std_dict, task=self.task)
            self.data_test = SeasFireDataset(test_batches, input_vars=self.variables, positional_vars=self.positional_vars,
                                       target=self.target, lead_time=self.predict_range,
                                       mean_std_dict=self.mean_std_dict, task=self.task)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            drop_last=False
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
            collate_fn=collate_fn,
            drop_last=False
        )