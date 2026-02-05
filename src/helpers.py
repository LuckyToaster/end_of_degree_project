import sys, aiofiles
from asyncio import Semaphore, create_task
from aiohttp import ClientSession
from tqdm.asyncio import tqdm
from requests import get
from PIL import Image 
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
from os import scandir

__all__ = ['download_images', 'download_images_sync', 'downsample_imgs', 'check_corrupted_imgs', 'standardize', 'train_loop']


def train_loop(model, epochs, loader, criterion, optimizer, device):
    model.train()
    for epoch in range(epochs): 
        running_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}", leave=True)
        
        for inputs, targets in loop:
            inputs = inputs.to(device)
            targets = targets.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0) # Added this to prevent exploding gradients in regression
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1} Complete - Avg Loss: {running_loss/len(loader):.4f}")


def standardize(train_df, test_df):
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    return scaler.fit_transform(train_df), scaler.transform(test_df)


def check_corrupted_imgs(imgs_dir):
    corrupted = []
    total = sum(1 for entry in scandir(imgs_dir) if entry.is_file())
    loop = tqdm(Path(imgs_dir).iterdir(), total=total, desc='Checking for corrupted images')
    for p in loop:
        try:
            with Image.open(p) as img:
                img.verify() 
                with Image.open(p) as img2: 
                    img2.load() 
        except Exception as e: corrupted.append((str(p), str(e)))
    return corrupted 

def count_files(directory):
    return sum(1 for entry in scandir(directory) if entry.is_file())


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

