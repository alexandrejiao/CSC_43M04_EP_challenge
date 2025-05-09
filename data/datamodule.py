from torch.utils.data import DataLoader, random_split
from data.dataset import Dataset
import torch



class DataModule:
    def __init__(
        self,
        dataset_path,
        train_transform,
        test_transform,
        batch_size,
        num_workers,
        metadata=["title"],
        val_ratio=0.2,  # Add validation ratio parameter
        random_seed=42
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.train_set = None  # Will store the train split
        self.val_set = None    # Will store the val split

    def _get_full_train_set(self):
        """Helper method to load full training data"""
        return Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.train_transform,
            metadata=self.metadata,
        )

    def _split_train_val(self):
        """Split the training data into train and validation sets"""
        full_train = self._get_full_train_set()
        val_size = int(len(full_train) * self.val_ratio)
        train_size = len(full_train) - val_size
        
        # Split with fixed random seed for reproducibility
        self.train_set, self.val_set = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.random_seed)
        )
        
        # Apply different transforms if needed
        self.val_set.dataset.transforms = self.test_transform  

    def train_dataloader(self):
        """Train dataloader with automatic train-val split"""
        if self.train_set is None:
            self._split_train_val()
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Validation dataloader"""
        if self.val_set is None:
            self._split_train_val()
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,  # Important: don't shuffle validation
            num_workers=self.num_workers,
        )

    def train_dataloader2(self):
        """Train dataloader."""
        train_set = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.train_transform,
            metadata=self.metadata,
        )
        return DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader2(self):
        """TODO: 
        Implement a strategy to create a validation set from the train set.
        """
        return
    
    def test_dataloader(self):
        """Test dataloader."""
        dataset = Dataset(
            self.dataset_path,
            "test",
            transforms=self.test_transform,
            metadata=self.metadata,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )