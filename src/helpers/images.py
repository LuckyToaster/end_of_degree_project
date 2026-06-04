from concurrent.futures import ProcessPoolExecutor
from tqdm.asyncio import tqdm
from pathlib import Path
from itertools import repeat
from os import cpu_count
from contextlib import contextmanager
import requests

import torch
from torchvision import io
from torchvision.transforms import v2, InterpolationMode
import tempfile, sys, os

__all__ = ['get_corrupted_images', 'download_and_resize_images']


def download_and_resize_images(urls, dst_paths, size=256):
    '''
        Download images in urls, resize them to size (keeping aspect ratio) and save them in dst_paths
    '''
    if not urls or not dst_paths: return []
    dst_dir = str(Path(dst_paths[0]).parent)
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        failed_urls = list(tqdm(
            executor.map(_download_and_resize_img, urls, dst_paths, repeat(size)),
            total=len(urls),
            desc=f'Downloading and Resizing into {dst_dir}',
            unit='img',
            file=sys.stdout
        ))
    return [url for url in failed_urls if url is not None]


def get_corrupted_images(imgs_dir: str):
    ''' 
        open images in image dir with tv to verify integrity 
    '''
    paths = list(map(lambda p: str(p), Path(imgs_dir).iterdir()))
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        results = list(tqdm(
            executor.map(_check_corrupted_img, paths),
            total=len(paths),
            desc=f'{str(imgs_dir)} => Verifying Integrity',
            unit='img',
            file=sys.stdout
        ))
    return [r for r in results if r is not None]


'''
    PRIVATE FUNCTIONS:
'''
def _download_and_resize_img(url, dst_path, size):
    try: 
        bytes = _download_img_bytes(url)
        _resize_and_save_bytes(bytes, dst_path, size)
    except (RuntimeError, Exception) as e: 
        # print(f'Skipping {url} => {e}')
        return url


def _download_img_bytes(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else: raise Exception(f'Failed to Download {url}')


def _resize_and_save_bytes(img_bytes, dst_path, size):
    byte_tensor = torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8)
    with _capture_stderr() as log:
        img = io.decode_image(byte_tensor, mode=io.ImageReadMode.RGB)
        transform = v2.Resize(size=size, interpolation=InterpolationMode.BICUBIC, antialias=True)
        img = transform(img)
        io.write_jpeg(img.as_subclass(torch.Tensor).to(torch.uint8).cpu(), dst_path, quality=95)


def _check_corrupted_img(path: str):
    # with _capture_stderr():
    try:
        t_img = io.read_image(path, mode=io.ImageReadMode.RGB)
        if t_img.ndim not in [2, 3] or t_img.numel() == 0: 
            return (path, f"Invalid tensor shape: {t_img.shape}")
    except (RuntimeError, Exception) as e: 
        return (path, e)


@contextmanager
def _capture_stderr():
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
