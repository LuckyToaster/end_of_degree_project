import asyncio
from pathlib import Path
from src.mmfood100k.builder import MMFood100KBuilder 

async def main():
    builder = MMFood100KBuilder(Path('data'), 224)
    await builder.download_imgs()
    await builder.fix_corrupted_imgs()
    builder.resize_images()

asyncio.run(main())


     


