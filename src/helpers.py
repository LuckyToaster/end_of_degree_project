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

__all__ = ['download_images', 'download_images_sync', 'downsample_imgs', 'check_corrupted_imgs', 'standardize', 'train_loop', 'file_count']


def train_loop(model, epochs, loader, criterion, optimizer, device):
    model.train()
    epochs_losses = []
    for epoch in range(epochs): 
        running_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}", leave=True)
        
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


def train_model(device, model, dataloaders, epochs, criterion, optimizer):
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(epochs):
        # training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        stopper = EarlyStopper(patience, min_delta) 

        for batchX, batchY in dataloaders['train']:
            batchX, batchY = batchX.to(device), batchY.to(device)
            optimizer.zero_grad()
            outputs = model(batchX)
            loss = criterion(outputs, batchY)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = max(outputs.data, 1)
            train_total += batchY.size(0)
            train_correct += (predicted == batchY).sum().item()

        # validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with no_grad():
            for batchX, batchY in dataloaders['val']:
                batchX, batchY = batchX.to(device), batchY.to(device)
                outputs = model(batchX)
                loss = criterion(outputs, batchY)

                val_loss += loss.item()
                _, predicted = max(outputs.data, 1)
                val_total += batchY.size(0)
                val_correct += (predicted == batchY).sum().item()

        # Calculate averages
        train_loss /= len(dataloaders['train'])
        val_loss /= len(dataloaders['val'])
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        # Store history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        print(f'Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        # stop early if need be
        if stopper.early_stop(val_loss): break
    return history



def standardize(train_df, test_df):
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    return scaler.fit_transform(train_df), scaler.transform(test_df)


def check_corrupted_imgs(imgs_dir: Path):
    corrupted = []
    for p in tqdm(imgs_dir.iterdir(), total=file_count(imgs_dir), desc='Checking for corrupted images'): 
        try:
            with Image.open(p) as img:
                img.verify() 
                with Image.open(p) as img2: 
                    img2.load() 
        except Exception as e: corrupted.append((str(p), str(e)))
    return corrupted 


def file_count(directory: Path):
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

