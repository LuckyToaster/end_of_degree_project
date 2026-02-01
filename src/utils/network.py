import sys, aiofiles
from asyncio import Semaphore, create_task
from aiohttp import ClientSession
from tqdm.asyncio import tqdm
from requests import get

async def download_images(urls, paths, limit, tqdm_desc):
    async with ClientSession() as sesh:         
        sem = Semaphore(limit) if limit else None
        tasks = [create_task(download_image(sesh, u, p, sem)) for u, p in zip(urls, paths)]
        await tqdm.gather(*tasks, desc=tqdm_desc, unit="img")


async def download_image(session, url, path, semaphore):
    if semaphore: 
        async with semaphore: 
            await dwn_img(session, url, path)
    else: await dwn_img(session, url, path)


async def dwn_img(session, url, dst_path):
    try:
        async with session.get(url) as res:
            res.raise_for_status()
            data = await res.read()
            async with aiofiles.open(dst_path, mode='wb') as f: 
                await f.write(data)
    except Exception as e: 
        print(f'{url}: {e}', file=sys.stderr)


def download_images_sync(urls, paths):
    for url, path in tqdm(zip(urls, paths) , total=len(urls), desc="Downloading", unit="img"):
        download_image_sync(url, path)

def download_image_sync(url, dst_path):
    try:
        r = get(url, timeout=10)
        if r.status_code != 200: print(f"Failed: {url} (Status: {r.status_code})")
        with open(dst_path, 'wb') as f: f.write(r.content)
    except Exception as e: print(f"Error with {url}: {e}")
