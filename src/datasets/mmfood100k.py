import pandas as pd
from pathlib import Path
from aiohttp import ClientSession
from asyncio import Semaphore, gather, create_task
from utils.download_img import download_img


class MMFood100K:
    # https://huggingface.co/datasets/Codatta/MM-Food-100K
    url = 'hf://datasets/Codatta/MM-Food-100K/MM-Food-100K.csv' 

    def __init__(self, data_path):
        self.dir = data_path / 'mm-food-100k'
        self.csv_path = self.dir / 'mm-food-100k.csv'
        self.imgs_dir = self.dir / 'imgs'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.df = pd.read_csv(self.csv_path) if self.csv_path.is_file() else self.__download_csv()

    def __download_csv(self):
        print(f'{str(self.csv_path)} not present, downloading ...')
        df = pd.read_csv(self.url)
        df = df.rename(columns={'image_url': 'img_url'})
        df['img_suffix'] = df['img_url'].apply(lambda x: Path(x).suffix)
        df['img_path'] = str(self.imgs_dir) + '/' + df.index.astype(str) + df['img_suffix']
        df = df.drop(columns=['img_suffix'])
        df.to_csv(self.csv_path, index=False)
        return df

    def __get_missing_imgs_ids(self):
        if not self.imgs_dir.exists() or not any(self.imgs_dir.iterdir()): 
            return sorted(range(0, 100_000))
        ids_present = [int(i.name.split('.')[0]) for i in self.imgs_dir.iterdir()]
        return sorted(set(range(0, 100_000)) - set(ids_present))

    @staticmethod
    async def __throttle(semaphore, session, url, path):
        async with semaphore:
            return await download_img(session, url, path)

    async def download_imgs(self, limit=25):
        self.imgs_dir.mkdir(parents=True, exist_ok=True)

        while True:
            ids_missing = self.__get_missing_imgs_ids()
            if not ids_missing:
                print(f'{self.imgs_dir} finished downloading')
                break

            print(f'Downloading {len(ids_missing)} missing images')
            urls = self.df.iloc[ids_missing]['img_url']
            paths = self.df.iloc[ids_missing]['img_path']
            async with ClientSession() as s:         
                sem = Semaphore(limit)
                tasks = [create_task(self.__throttle(sem, s, u, p)) for u, p in zip(urls, paths)]
                await gather(*tasks)

        # if self.imgs_dir.exists() and any(self.imgs_dir.iterdir()): 
        #     return self

        # if self.imgs_dir.exists():
        #     ids_present = [int(i.name.split('.')[0]) for i in self.imgs_dir.iterdir()]
        #     ids_missing = sorted(set(range(0, 100_000)) - set(ids_present))
        #
        #     urls = self.df.iloc[ids_missing]['img_url']
        #     paths = self.df.iloc[ids_missing]['img_path']
        #
        #     async with ClientSession() as s:         
        #         sem = Semaphore(limit)
        #         tasks = [create_task(self.__throttle(sem, s, u, p)) for u, p in zip(urls, paths)]
        #         await gather(*tasks)
        #
        # else:
        #     print(f'{str(self.imgs_dir)} not present, downloading ...')
        #     self.imgs_dir.mkdir(parents=True)
        #
        #     urls = self.df['img_url'].tolist()
        #     paths = self.df['img_path'].tolist()
        #
        #     async with ClientSession() as s:         
        #         sem = Semaphore(limit)
        #         tasks = [create_task(self.__throttle(sem, s, u, p)) for u, p in zip(urls, paths)]
        #         await gather(*tasks)
        #
        #     return self


