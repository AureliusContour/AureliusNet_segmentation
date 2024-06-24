# PyTorch libraries
import lightning as L
from torch.utils.data import DataLoader

# Local libraries
from .transform import transform
from .BreastCTDataset import BreastCTDataset

# Other libraries
from pandas import DataFrame
import os


class BreastCTDataModule(L.LightningDataModule):
    def __init__(self, dataframe:DataFrame, batch_size=32):
        super().__init__()
        self.df = dataframe
        self.batch_size = batch_size

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.trainDS = BreastCTDataset(self.df, transform=transform, set_type="train")
            self.valDS = BreastCTDataset(self.df, transform=transform, set_type="validation")

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.testDS = BreastCTDataset(self.df, transform=transform, set_type="test")

    def train_dataloader(self):
        return DataLoader(self.trainDS, 
                          batch_size=self.batch_size,
                          shuffle=True, 
						  num_workers=os.cpu_count(), 
						  pin_memory=True, 
						  persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valDS, 
                          batch_size=self.batch_size,
                          shuffle=False, 
						  num_workers=os.cpu_count(), 
						  pin_memory=True, 
						  persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.testDS, 
                          batch_size=self.batch_size,
                          shuffle=False, 
						  num_workers=os.cpu_count(), 
						  pin_memory=True, 
						  persistent_workers=True)
