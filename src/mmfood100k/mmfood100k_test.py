import pandas as pd
from pathlib import Path
import torch 
from src.mmfood100k.dataset import MMFood100KDataset
from src.models import get_EfficientNet_B0
import random

data_path = Path('data')
df = pd.read_csv(data_path / 'mm-food-100k/mm-food-100k.csv')


def test_dataset_no_errors():
    _, transforms = get_EfficientNet_B0()
    dataset = MMFood100KDataset(df, transform=transforms)

    head_indices = list(range(100))
    tail_indices = list(range(len(dataset) - 100, len(dataset)))
    random_indices = random.sample(range(100, len(dataset)-100), 100) 
    
    for i in head_indices + tail_indices + random_indices:
        img, target = dataset[i]
        assert img.shape == (3, 224, 224)
        assert not torch.isnan(img).any()
        assert target.shape == (3,) 
    
