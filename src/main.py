from pathlib import Path
from datasets.mmfood100k import MMFood100K
import asyncio

async def main():
    await MMFood100K(Path('data')).download_imgs(limit=10)

if __name__ == "__main__":
    asyncio.run(main())

