import pytest
from lightning import LightningDataModule
from torchvision import transforms

from flowers_dataset import FlowersDataset
from lit_flowers_datamodule import LightningFlowersDatamodule

@pytest.fixture
def flowers_data_module():
    train_directory = 'flowerdataset/train'
    test_directory = 'flowerdataset/test'
    batch_size = 32
    return LightningFlowersDatamodule(train_directory, test_directory, batch_size)

def test_lightning_flowers_data_module_initialization(flowers_data_module):
    assert flowers_data_module.train_directory == 'flowerdataset/train'
    assert flowers_data_module.test_directory == 'flowerdataset/test'
    assert flowers_data_module.batch_size == 32
    assert flowers_data_module.train_split == 0.2
    assert flowers_data_module.transform is not None

def test_lightning_flowers_data_module_setup_fit(flowers_data_module):
    flowers_data_module.setup(stage='fit')
    assert len(flowers_data_module.train_dataset) > 0
    assert len(flowers_data_module.val_dataset) > 0

def test_lightning_flowers_data_module_setup_test(flowers_data_module):
    flowers_data_module.setup(stage='test')
    assert len(flowers_data_module.test_dataset) > 0

def test_lightning_flowers_data_module_train_dataloader(flowers_data_module):
    flowers_data_module.setup(stage='fit')
    train_dataloader = flowers_data_module.train_dataloader()
    assert train_dataloader.batch_size == 32
    assert len(train_dataloader.dataset) > 0

def test_lightning_flowers_data_module_val_dataloader(flowers_data_module):
    flowers_data_module.setup(stage='fit')
    val_dataloader = flowers_data_module.val_dataloader()
    assert val_dataloader.batch_size == 32
    assert len(val_dataloader.dataset) > 0

def test_lightning_flowers_data_module_test_dataloader(flowers_data_module):
    flowers_data_module.setup(stage='test')
    test_dataloader = flowers_data_module.test_dataloader()
    assert test_dataloader.batch_size == 32
    assert len(test_dataloader.dataset) > 0