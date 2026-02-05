import sys, aiofiles
from asyncio import Semaphore, create_task
from aiohttp import ClientSession
from tqdm.asyncio import tqdm
from requests import get
from PIL import Image 
from sklearn.preprocessing import StandardScaler

__all__ = ['download_images', 'download_images_sync', 'downsample_imgs', 'standardize']


def standardize(train_df, test_df):
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    return scaler.fit_transform(train_df), scaler.transform(test_df)


async def download_images(urls, paths, limit, tqdm_desc):
    '''
        download a list of images asynchronouly with a limit (semaphore)
    '''
    async with ClientSession() as sesh:         
        sem = Semaphore(limit) if limit else None
        tasks = [create_task(_dwn_img_semaphore(sesh, u, p, sem)) for u, p in zip(urls, paths)]
        await tqdm.gather(*tasks, desc=tqdm_desc, unit="img")


def download_images_sync(urls, paths):
    '''
        Download a list of images synchronously
    '''
    for url, path in tqdm(zip(urls, paths) , total=len(urls), desc="Downloading", unit="img"):
        _dwn_img_sync(url, path)


def downsample_imgs(img_dir_path, pixel_limit=89_478_485, max_dim=2000):
    '''
        Downsammple keeping aspect ratio all images in a directory larger than a pixel_limit
    '''
    for img_path in img_dir_path.iterdir():
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if w * h <= pixel_limit: continue

                print(f"Downsizing {img_path} ({w}x{h})")
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                img.save(img_path, quality=95, subsampling=0)
        except Exception as e: print(f"Error processing {img_path}: {e}")


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
    except Exception as e: 
        print(f'{url}: {e}', file=sys.stderr)


def _dwn_img_sync(url, dst_path):
    try:
        r = get(url, timeout=10)
        if r.status_code != 200: print(f"Failed: {url} (Status: {r.status_code})")
        with open(dst_path, 'wb') as f: f.write(r.content)
    except Exception as e: print(f"Error with {url}: {e}")

