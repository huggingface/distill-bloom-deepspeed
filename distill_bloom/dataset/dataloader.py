import torch

class DistributedDataset(torch.utils.data.Dataset):
    r"""
        Wrapper for torch.utils.data.Dataset to make it distributed.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to be distributed.
            rank (int): Rank of the current process.
            world_size (int): Number of processes in the distributed group.
    """
    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset

        self.current_dataset_index = 0
        self.current_dataset = dataset[self.current_dataset_index]
        
        self.rank = rank
        self.world_size = world_size
    
    def _update_dataset(self):
        self.current_dataset_index += 1
        if self.current_dataset_index >= len(self.dataset):
            self.current_dataset_index = 0
        self.current_dataset = self.dataset[self.current_dataset_index]
    
    def __getitem__(self, index):
        r"""
            Loads a unique sample from the dataset.
            First tries to load the sample from the current dataset.
            If the current dataset is exhausted, it moves to the next dataset.
        """
        try:
            item = self.current_dataset[(index*self.world_size) + self.rank]
        except IndexError:
            self._update_dataset()
            item = self.current_dataset[(index*self.world_size) + self.rank]
        return item

    def __len__(self):
        r"""
            Returns the length of the dataset. It corresponds to the total
            lenght of all the datasets in the dataset list.
        """
        total_length = list(map(lambda x: len(x), self.dataset))
        return sum(total_length)

class DistributedDataLoader(torch.utils.data.DataLoader):
    r"""
        Wrapper around torch.utils.data.DataLoader to support distributed training.
    """
    def __init__(self, dataset, rank, world_size, **kwargs):
        self.dataset = DistributedDataset(dataset, rank, world_size)
        super().__init__(self.dataset, **kwargs)
    