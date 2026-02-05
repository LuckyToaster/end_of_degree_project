import asyncio
from pathlib import Path
from src.mmfood100k.builder import MMFood100KBuilder 
from src.helpers import check_corrupted_imgs, download_images

import pandas as pd

async def main():
    await MMFood100KBuilder(Path('data')).download_imgs(limit=10)

    corrupted_imgs = check_corrupted_imgs('data/mm-food-100k/imgs')
    df = pd.read_csv('data/mm-food-100k/mm-food-100k.csv')
    paths = [pathstr for pathstr, _ in corrupted_imgs]
    urls = df[ df['img_path'].isin(paths) ]['img_url']
    await download_images(urls, paths, 10, 'Re-Downloading corrupted images')


asyncio.run(main())


     


