from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from asyncio import Semaphore, create_task
from aiohttp import ClientSession
from tqdm.asyncio import tqdm
from pathlib import Path
from itertools import repeat
from os import cpu_count

import torch
from torchvision import io
from torchvision.transforms import v2, InterpolationMode

import tempfile, sys, os, aiofiles
from contextlib import contextmanager


__all__ = ['resize_images', 'get_corrupted_images', 'download_images']


def resize_images(src_paths, dst_paths, size=256):
    '''
        Resize images in src_paths to dst_paths to size keeping aspect ratio
    '''
    src_dir = str(Path(src_paths[0]).parent)
    dst_dir = str(Path(dst_paths[0]).parent)
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        list(tqdm(
            executor.map(_resize_img, src_paths, dst_paths, repeat(size)),
            total=len(src_paths),
            desc=f'{src_dir} => Re-Sizing Images to {dst_dir}',
            unit='img',
            file=sys.stdout
        ))


def get_corrupted_images(imgs_dir: Path):
    ''' 
        open images in image dir with tv to verify integrity 
    '''
    paths = list(map(lambda p: str(p), imgs_dir.iterdir()))
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        results = list(tqdm(
            executor.map(_check_corrupted_img, paths),
            total=len(paths),
            desc=f'{str(imgs_dir)} => Verifying Integrity',
            unit='img',
            file=sys.stdout
        ))
    return [r for r in results if r is not None]


async def download_images(urls, paths, limit, tqdm_desc):
    '''
        download a list of images asynchronouly with a limit (semaphore)
    '''
    async with ClientSession() as sesh:         
        sem = Semaphore(limit) if limit else None
        tasks = [create_task(_dwn_img_semaphore(sesh, u, p, sem)) for u, p in zip(urls, paths)]
        await tqdm.gather(*tasks, desc=tqdm_desc, total=len(urls), unit="img", file=sys.stdout)


'''
    PRIVATE FUNCTIONS:
'''

def _resize_img(src_path, dst_path, size):
    try: 
        img = io.read_image(str(src_path), mode=io.ImageReadMode.RGB)
        transform = v2.Resize(size=size, interpolation=InterpolationMode.BICUBIC, antialias=True) # keeps aspect ratio, uses BICUBIC like efficientnet
        img = transform(img)
        io.write_jpeg(img.as_subclass(torch.Tensor).to(torch.uint8).cpu(), dst_path, quality=95)
    except (RuntimeError, Exception) as e: 
        print(f'Skipped {src_path} => {e}')


def _check_corrupted_img(path: str):
    with capture_stderr() as log:
        try:
            t_img = io.read_image(path, mode=io.ImageReadMode.RGB)

            log.seek(0)
            warning = log.read()
            if warning: raise ValueError(f'C-Level Warning Detected: {warning.strip()}')

            if t_img.ndim not in [2, 3] or t_img.numel() == 0: 
                return (path, f"Invalid tensor shape: {t_img.shape}")

        except (RuntimeError, Exception) as e: 
            return (path, e)


@contextmanager
def capture_stderr():
    '''
        Redirects C-level stderr to a temporary file to catch library warnings.
    '''
    with tempfile.NamedTemporaryFile(mode='w+') as tmp: # Create a pipe or temporary file
        stderr_fd = sys.stderr.fileno()
        with os.fdopen(os.dup(stderr_fd), 'w') as old_stderr: # Keep a backup of the original stderr
            sys.stderr.flush()
            os.dup2(tmp.fileno(), stderr_fd) # Redirect FD 2 to temp file
            try: yield tmp
            finally:
                sys.stderr.flush()
                os.dup2(old_stderr.fileno(), stderr_fd) # Restore stderr


async def _dwn_img_semaphore(session, url, path, semaphore):
    if semaphore:
        async with semaphore: 
            await _dwn_img(session, url, path)
    else: await _dwn_img(session, url, path)


async def _dwn_img(session, url, dst_path):
    try:
        async with session.get(url) as res:
            res.raise_for_status()
            data = await res.read()
            async with aiofiles.open(dst_path, mode='wb') as f: 
                await f.write(data)
    except (RuntimeError, Exception) as e: 
        tqdm.write(f'{url} => {e}')
