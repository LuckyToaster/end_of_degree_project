from sys import stderr, stdout
from datasets.mmfood100k import MMFood100K
from pathlib import Path
import pytest

ds = MMFood100K(Path('data'))
missing_img_paths = ds.df.iloc[ds._missing_img_ids()]['img_path'].tolist()
present_img_paths = ds.df.iloc[ds._file_img_ids()]['img_path'].tolist()

def test_no_missing_is_present():
    assert all(not Path(p).is_file() for p in missing_img_paths)

def test_no_present_is_missing():
    assert all(Path(p).is_file() for p in present_img_paths)

def test_mutual_exclusion():
    present = set(ds._file_img_ids())
    missing = set(ds._missing_img_ids())
    assert len(present.intersection(missing)) == 0
