import logging
import numpy as np
import omegaconf
from beartype import beartype

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


@beartype
def get_dataloader(cfg, shared_dict= None,):

   
    from dataset import generate_batch
    from dataset import PlanTDataset as Dataset
    
    if cfg.benchmark == 'lav':
        # we train without T2 and T5

        logging.info(f'LAV training without T2 and T5')
        train_set = Dataset(cfg.data_dir, cfg, shared_dict=shared_dict, split='train')
        val_set = Dataset(cfg.data_dir, cfg, shared_dict=shared_dict, split='val')
    
    elif cfg.benchmark == 'longest6':
        logging.info(f'Longest6 training with all towns')
        dataset = Dataset(
            cfg.data_dir, cfg, shared_dict=shared_dict, split="all"
        )

        # we validate on CARLA closed-loop, so we don't have a proper validation set here
        if cfg.trainset_size >= 1:
            train_set = dataset

            # use a very small subset of the trainset as validation set
            train_length = int(len(dataset) * 0.98)
            val_length = len(dataset) - train_length
            # train_length = int(len(dataset) * 0.08)
            # val_length = int(len(dataset) * 0.01)

            _, val_set = random_split(dataset, [train_length, val_length])

        else:
            train_length = int(len(dataset) * cfg.trainset_size)
            val_length = int(len(dataset) * 0.02)
            test_length = len(dataset) - train_length - val_length

            train_set, val_set, _ = random_split(
                dataset, [train_length, val_length, test_length]
            )
    else: 
        raise ValueError(f"Unknown benchmark: {cfg.benchmark}")
        
    logging.info(f'Train set size: {len(train_set)}')
    logging.info(f'Validation set size: {len(val_set)}')

    
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        # sampler=sampler_train,
        pin_memory=False,
        batch_size=cfg.model.training.batch_size,
        collate_fn=generate_batch,
        num_workers=cfg.model.training.num_workers,
    )

    # part_len2 = len(val_set) // cfg.gpus
    # indices2 = np.arange(
    #     rank * part_len2, min(len(val_set), (1 + rank) * part_len2), 1
    # )
    # sampler_val = SubsetRandomSampler(indices2)
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        # sampler=sampler_val,
        pin_memory=False,
        batch_size=cfg.model.training.batch_size,
        collate_fn=generate_batch,
        num_workers=cfg.model.training.num_workers,
    )
    


    return train_loader, val_loader
