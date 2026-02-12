import pandas as pd
from pathlib import Path
import torch 
from src.mmfood100k.builder import MMFood100KBuilder
from src.mmfood100k.dataset import MMFood100KDataset
from src.models.efficientnet import get_model
from src.helpers.misc import ids, missing_ids
import random


data_path = Path('data')
builder = MMFood100KBuilder(data_path, 224)
df = pd.read_csv(data_path / 'mm-food-100k/mm-food-100k.csv')

def test_no_missing_is_present():
    ids_missing = missing_ids(builder.imgs_dir, len(builder.df))
    missing_img_paths = df.iloc[ids_missing]['img_path'].tolist()
    assert all(not Path(p).is_file() for p in missing_img_paths)

def test_no_present_is_missing():
    present_ids = ids(builder.imgs_dir)
    present_img_paths = df.iloc[present_ids]['img_path'].tolist()
    assert all(Path(p).is_file() for p in present_img_paths)

def test_mutual_exclusion():
    present = set(ids(builder.imgs_dir))
    missing = set(missing_ids(builder.imgs_dir, len(builder.df)))
    assert len(present.intersection(missing)) == 0

def test_dataset_no_errors():
    _, transforms = get_model()
    dataset = MMFood100KDataset(df, transform=transforms)

    head_indices = list(range(100))
    tail_indices = list(range(len(dataset) - 100, len(dataset)))
    random_indices = random.sample(range(100, len(dataset)-100), 100) 
    
    for i in head_indices + tail_indices + random_indices:
        img, target = dataset[i]
        assert img.shape == (3, 224, 224)
        assert not torch.isnan(img).any()
        assert target.shape == (3,) 
    
