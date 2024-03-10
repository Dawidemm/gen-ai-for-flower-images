from torch.utils.data import DataLoader
from torch.utils.data import random_split
from lightning import LightningDataModule
from torchvision import transforms

from src.utils.flowers_dataset import FlowersDataset

class LightningFlowersDatamodule(LightningDataModule):
    def __init__(
            self, 
            train_directory: str, 
            test_directory: str, 
            batch_size: int,
            train_split: float = 0.2,
            transform=None
    ):
        
        super().__init__()

        self.train_directory = train_directory
        self.test_directory = test_directory
        self.train_split = train_split
        self.batch_size = batch_size

        if transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=30),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def setup(self, stage: str):

        if stage == 'fit':
            dataset = FlowersDataset(root_dir=self.train_directory, transform=self.transform)
            train_dataset_samples = int(len(dataset)* (1-self.train_split))
            val_dataset_samples = int(len(dataset)*self.train_split)
            self.train_dataset, self.val_dataset = random_split(dataset, [train_dataset_samples, val_dataset_samples])

        if stage == 'test':
            self.test_dataset = FlowersDataset(root_dir=self.test_directory)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)