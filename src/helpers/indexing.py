from pathlib import Path

__all__ = ['ids', 'missing_ids']

def ids(dir: Path):
    return [int(i.name.split('.')[0]) for i in dir.iterdir()]

def missing_ids(dir: Path, length: int):
    if not any(dir.iterdir()): return sorted(range(0, 100_000))
    return sorted(set(range(0, length)) - set(ids(dir)))
