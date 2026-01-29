from pathlib import Path
from datasets.mmfood100k import MMFood100K
import asyncio

async def main():
    data_path = Path('data')
    data_path.mkdir(parents=True, exist_ok=True)
    await MMFood100K(data_path).download_imgs()

if __name__ == "__main__":
    asyncio.run(main())
