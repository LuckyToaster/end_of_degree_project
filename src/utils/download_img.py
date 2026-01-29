import sys, aiofiles
from pathlib import Path

async def download_img(session, url, dst_path):
    path = Path(dst_path)
    if path.is_file(): 
        print(f'{dst_path} already exists', file=sys.stderr)
        return
    try:
        async with session.get(url) as res:
            if res.status != 200: 
                print(f'Error downloading: {url}', file=sys.stderr)
                return
            data = await res.read()
            async with aiofiles.open(path, mode='wb') as f: 
                await f.write(data)
    except Exception as e: 
        print(f'session.get() failed for {url}: {e}', file=sys.stderr)
