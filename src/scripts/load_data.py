import asyncio
from pathlib import Path
from src.mmfood100k.builder import MMFood100KBuilder 

async def main():
    builder = MMFood100KBuilder(Path('data'))
    await builder.download_imgs()
    builder.downsample_imgs(224)
    await builder.fix_corrupted_imgs()

asyncio.run(main())


     


