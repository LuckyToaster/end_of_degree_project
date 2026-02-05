import asyncio
from pathlib import Path
from src.mmfood100k.builder import MMFood100KBuilder 

async def main():
    await MMFood100KBuilder(Path('data')).download_imgs(limit=10)

asyncio.run(main())


