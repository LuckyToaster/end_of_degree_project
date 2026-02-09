from concurrent.futures import ThreadPoolExecutor
import aiofiles
from sys import stderr, stdout
from asyncio import Semaphore, create_task
from aiohttp import ClientSession
from tqdm.asyncio import tqdm
from pathlib import Path
from itertools import repeat
from os import cpu_count
from torchvision import io
from torchvision.transforms import v2, InterpolationMode
import torch

__all__ = ['resize_images', 'get_corrupted_images', 'download_images']


def resize_images(src_paths, dst_paths, size=256, bicubic=True):
    '''
        Resize images in src_paths to dst_paths to size keeping aspect ratio
    '''
    src_dir = str(Path(src_paths[0]).parent)
    dst_dir = str(Path(dst_paths[0]).parent)
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        list(tqdm(
            executor.map(_resize_img, src_paths, dst_paths, repeat(size), repeat(bicubic)),
            total=len(src_paths),
            desc=f'{src_dir} => Re-Sizing Images to {dst_dir}',
            unit='img',
            file=stdout
        ))


def get_corrupted_images(imgs_dir: Path):
    ''' 
        open images in image dir with tv to verify integrity 
    '''
    paths = list(map(lambda p: str(p), imgs_dir.iterdir()))
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        results = list(tqdm(
            executor.map(_check_corrupted_img, paths),
            total=len(paths),
            desc=f'{str(imgs_dir)} => Verifying Integrity',
            unit='img',
            file=stdout
        ))
    return [r for r in results if r is not None]


async def download_images(urls, paths, limit, tqdm_desc):
    '''
        download a list of images asynchronouly with a limit (semaphore)
    '''
    async with ClientSession() as sesh:         
        sem = Semaphore(limit) if limit else None
        tasks = [create_task(_dwn_img_semaphore(sesh, u, p, sem)) for u, p in zip(urls, paths)]
        await tqdm.gather(*tasks, desc=tqdm_desc, total=len(urls), unit="img", file=stdout)


'''
    PRIVATE FUNCTIONS:
'''

def _resize_img(src_path, dst_path, size, bicubic):
    img = io.read_image(str(src_path), mode=io.ImageReadMode.RGB)
    transform = v2.Resize(size=size, interpolation=InterpolationMode.BICUBIC if bicubic else InterpolationMode.BILINEAR, antialias=True) # keeps aspect ration
    img = transform(img)

    if img.ndim == 4: img = img.squeeze(0)  # [1, 3, H, W] -> [3, H, W]
    elif img.ndim == 2: img = img.unsqueeze(0).repeat(3, 1, 1) # [H, W] -> [3, H, W]
    if img.shape[0] != 3: img = img.repeat(3, 1, 1)[:3, :, :] # If it's something bizarre like [1, H, W] after all that

    io.write_jpeg(img.as_subclass(torch.Tensor).to(torch.uint8).cpu(), dst_path, quality=95)


def _check_corrupted_img(path: str):
    try:
        io.read_image(path, mode=io.ImageReadMode.RGB)
    except (RuntimeError, Exception) as e: 
        return (path, e)


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
        print(f'{url} => {e}', file=stderr)
