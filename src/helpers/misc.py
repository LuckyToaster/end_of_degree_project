from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
from sys import stderr
from tqdm import tqdm

__all__ = ['ids', 'missing_ids', 'remove_files']

def remove_files(paths: list[str], tqdm_desc='Removing files', tqdm_unit='file') -> None:
    try:
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            list(tqdm(executor.map(_rm_file, paths), total=len(paths), desc=tqdm_desc, unit=tqdm_unit))
    except (RuntimeError, Exception) as e:
        print(f'remove_files(): {e}')


def ids(dir: Path):
    return [int(i.name.split('.')[0]) for i in dir.iterdir()]


def missing_ids(dir: Path, length: int):
    if not any(dir.iterdir()): return sorted(range(0, length))
    return sorted(set(range(0, length)) - set(ids(dir)))


def _rm_file(path: str):
    p = Path(path)
    if p.is_dir(): print(f'rm_file() => {path} is a directory', file=stderr)
    elif not p.is_file(): print(f'rm_file() => {path} file does not exist', file=stderr)
    else: p.unlink()
