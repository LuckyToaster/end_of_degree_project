import pandas as pd
from pathlib import Path
from src.mmfood100k.builder import MMFood100KBuilder
from src.mmfood100k.dataset import MMFood100KDataset

data_path = Path('data')
builder = MMFood100KBuilder(data_path)
df = pd.read_csv(data_path / 'mm-food-100k/mm-food-100k.csv')

def test_no_missing_is_present():
    missing_img_paths = df.iloc[builder._missing_img_ids()]['img_path'].tolist()
    assert all(not Path(p).is_file() for p in missing_img_paths)

def test_no_present_is_missing():
    present_img_paths = df.iloc[builder._file_img_ids()]['img_path'].tolist()
    assert all(Path(p).is_file() for p in present_img_paths)

def test_mutual_exclusion():
    present = set(builder._file_img_ids())
    missing = set(builder._missing_img_ids())
    assert len(present.intersection(missing)) == 0

def test_dataset_iterable_no_errors():
    dataset = MMFood100KDataset(df)
    [(i, t) for i,t in dataset]
