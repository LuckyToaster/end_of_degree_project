import sys, aiofiles
from asyncio import Semaphore, create_task
from aiohttp import ClientSession
from tqdm.asyncio import tqdm
from requests import get
from PIL import Image 
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
from os import scandir, path
from pathlib import Path
import torch

__all__ = ['download_images', 'download_images_sync', 'downsample_imgs', 'check_corrupted_imgs', 'standardize', 'train_loop', 'file_count', 'downsample_img']


def train_loop(model, epochs, loader, criterion, optimizer, device):
    model.train()
    epochs_losses = []
    for epoch in range(epochs): 
        running_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}", leave=True, unit='batch')
        
        for inputs, targets in loop:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # Added this to prevent exploding gradients in regression
            clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epochs_losses.append(running_loss / len(loader))
        print(f"Epoch {epoch+1} Complete - Avg Loss: {running_loss/len(loader):.4f}")
    return epochs_losses


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def standardize(train_df, test_df):
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    return scaler.fit_transform(train_df), scaler.transform(test_df)

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from os import cpu_count

def downsample_images(src_paths, dst_paths, size):
    src_dir = str(Path(src_paths[0]).parent)
    dst_dir = str(Path(dst_paths[0]).parent)
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        list(tqdm(
            executor.map(downsample_img, src_paths, dst_paths, repeat(size)),
            total=len(src_paths),
            desc=f'{src_dir} => {dst_dir}: Downsampling Images',
            unit='img'
        ))

import torch
from torchvision import io
from torchvision.transforms import v2, InterpolationMode
from pathlib import Path

def resize_img(src_path, dst_path, size, bicubic):
    img = io.read_image(str(src_path), mode=io.ImageReadMode.RGB)
    interpolation = InterpolationMode.BICUBIC if bicubic else InterpolationMode.BILINEAR
    transform = v2.Resize(size=size, interpolation=interpolation, antialias=True) # keeps aspect ration
    img = transform(img)
    io.write_jpeg(img.cpu(), dst_path, quality=95)


def resize_images(src_paths, dst_paths, size=256, bicubic=True):
    src_dir = str(Path(src_paths[0]).parent)
    dst_dir = str(Path(dst_paths[0]).parent)
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        list(tqdm(
            executor.map(downsample_img, src_paths, dst_paths, repeat(size), repeat(bicubic)),
            total=len(src_paths),
            desc=f'{src_dir} => {dst_dir}: Downsampling Images',
            unit='img'
        ))


def downsample_img(old_path, new_path, size):
    try:
        with Image.open(old_path) as img:
            img = img.convert("RGB")
            img.thumbnail((size,size), resample=Image.LANCZOS)
            img.save(new_path, "JPEG", quality=95)
    except Exception or Error as e:
        print(f"Error processing {old_path}: {e}")


def check_corrupted_imgs(imgs_dir: Path):
    corrupted = []
    paths = list(imgs_dir.iterdir())
    for p in tqdm(paths, total=len(paths), desc='Checking for corrupted images', unit='img'): 
        try:
            with Image.open(p) as img:
                img.verify() 
                with Image.open(p) as img2: 
                    img2.load() 
        except (RuntimeError, Exception) as e: 
            corrupted.append((str(p), str(e)))
    return corrupted 


def check_corrupted_imgs_v2(imgs_dir: Path):
    paths = list(map(lambda p: str(p), imgs_dir.iterdir()))
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        return list(tqdm(
            executor.map(check_corrupted_img, paths),
            total=len(paths),
            desc=f'{str(imgs_dir)} => Verifying Integrity',
            unit='img'
        ))

def check_corrupted_img(path: str):
        try:
            io.read_image(path, mode=io.ImageReadMode.RGB)
        except (RuntimeError, Exception) as e: 
            return (path, e)

def file_count(directory: Path):
    return sum(1 for entry in scandir(directory) if entry.is_file())


async def download_images(urls, paths, limit, tqdm_desc):
    '''
        download a list of images asynchronouly with a limit (semaphore)
    '''
    async with ClientSession() as sesh:         
        sem = Semaphore(limit) if limit else None
        tasks = [create_task(_dwn_img_semaphore(sesh, u, p, sem)) for u, p in zip(urls, paths)]
        await tqdm.gather(*tasks, desc=tqdm_desc, total=len(urls), unit="img")


def download_images_sync(urls, paths):
    '''
        Download a list of images synchronously
    '''
    for url, path in tqdm(zip(urls, paths) , total=len(urls), desc="Downloading", unit="img"):
        _dwn_img_sync(url, path)


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

