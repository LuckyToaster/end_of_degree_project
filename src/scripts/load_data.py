import asyncio
from pathlib import Path
from src.datasets.mmfood100k import MMFood100KBuilder 

async def main():
    await MMFood100KBuilder(Path('data')).download_imgs(limit=10)

asyncio.run(main())
