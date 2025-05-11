import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import math
import itertools
import numpy as np

class FastRandomDistributedSampler(Sampler[int]):
    r"""
    A distributed sampler that continuously yields random indices with replacement,
    avoiding frequent iterator recreation overhead for DataLoader.

    Instead of stopping after one pass through the dataset, this sampler's
    iterator yields a specified number of indices (`epoch_steps`) before
    stopping. This significantly reduces the frequency of DataLoader worker
    restarts when the underlying dataset is small.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. Defaults to current world size.
        rank (int, optional): Rank of the current process. Defaults to current rank.
        seed (int): Base seed for the random number generator. Each epoch/rank
                  gets a different derived seed. Defaults to 0.
        epoch_steps (int): The number of indices this sampler should yield per
                         __iter__ call (per replica). Set this to a large number
                         to reduce iterator recreation frequency. If None, it defaults
                         to ceil(len(dataset) / num_replicas).
    """
    def __init__(self, dataset, num_replicas=None, rank=None, seed=0, epoch_steps=None):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Requires distributed package to be available and initialized")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Requires distributed package to be available and initialized")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.dataset_len = len(self.dataset)

        # Determine the number of steps/indices per iterator cycle for this rank
        if epoch_steps is None:
            # Default behavior: roughly one pass over the data
            self.num_samples_per_epoch = math.ceil(self.dataset_len / self.num_replicas)
        else:
            # User-defined length for the iterator cycle
            self.num_samples_per_epoch = epoch_steps

        if not isinstance(self.num_samples_per_epoch, int) or self.num_samples_per_epoch <= 0:
            raise ValueError("epoch_steps must be a positive integer")

    def _infinite_indices(self):
        """A generator that yields random indices indefinitely."""
        g = torch.Generator()
        # Ensure distinct seeds based on rank, epoch, and base seed
        current_seed = self.seed + self.epoch * self.num_replicas + self.rank
        g.manual_seed(current_seed)
        while True:
            yield torch.randint(low=0, high=self.dataset_len, size=(1,), generator=g).item()

    def __iter__(self):
        """
        Returns an iterator that yields 'num_samples_per_epoch' indices.
        It uses itertools.islice to take a finite slice from the
        infinite generator, avoiding expensive list creation.
        """
        # Create the infinite generator and slice it
        # The generator state is preserved across calls to next() by the DataLoader
        # The expensive DataLoader setup only happens when this sliced iterator is exhausted
        return itertools.islice(self._infinite_indices(), self.num_samples_per_epoch)

    def __len__(self):
        """The number of samples produced by the iterator per __iter__ call."""
        return self.num_samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. This is used to vary the random seed sequence
        each time __iter__ is called.
        """
        self.epoch = epoch

class QAMNISTSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            if self.dataset.num_images_range[0] == self.dataset.num_images_range[1]:
                batch_num_digits = self.dataset.num_images_range[0]
            else:
                batch_num_digits = np.random.randint(self.dataset.num_images_range[0], self.dataset.num_images_range[1])

            if self.dataset.num_operations_range[0] == self.dataset.num_operations_range[1]:
                batch_num_operations = self.dataset.num_operations_range[0]
            else:
                batch_num_operations = np.random.randint(self.dataset.num_operations_range[0], self.dataset.num_operations_range[1])

            self.dataset.set_num_digits(batch_num_digits)
            self.dataset.set_num_operations(batch_num_operations)
            
            yield batch_indices

    def __len__(self):
        return self.num_samples // self.batch_size