import pandas as pd
from pathlib import Path
from sys import stderr, stdout
from utils.network import download_images


class MMFood100K:
    # https://huggingface.co/datasets/Codatta/MM-Food-100K
    url = 'hf://datasets/Codatta/MM-Food-100K/MM-Food-100K.csv' 

    def __init__(self, data_path):
        self.dir = data_path / 'mm-food-100k'
        self.csv_path = self.dir / 'mm-food-100k.csv'
        self.imgs_dir = self.dir / 'imgs'

        self.dir.mkdir(parents=True, exist_ok=True)
        self.imgs_dir.mkdir(parents=True, exist_ok=True)

        self.df = pd.read_csv(self.csv_path) if self.csv_path.is_file() else pd.read_csv(self.url)
        
        if not self.csv_path.is_file():
            self.df = self.df.rename(columns={'image_url': 'img_url'})
            self.df['img_suffix'] = self.df['img_url'].apply(lambda x: Path(x).suffix)
            self.df['img_path'] = str(self.imgs_dir) + '/' + self.df.index.astype(str) + self.df['img_suffix']
            self.df = self.df.drop(columns=['img_suffix'])
            self.df.to_csv(self.csv_path, index=False)

    def _file_img_ids(self):
        return [int(i.name.split('.')[0]) for i in self.imgs_dir.iterdir()]

    def _missing_img_ids(self):
        if not any(self.imgs_dir.iterdir()): return sorted(range(0, 100_000))
        return sorted(set(range(0, 100_000)) - set(self._file_img_ids()))

    async def download_imgs(self, limit=10):
        while True:
            missing = self._missing_img_ids()
            if not missing: break
            await download_images(
                urls = self.df.iloc[missing]['img_url'], 
                paths = self.df.iloc[missing]['img_path'], 
                limit = limit, 
                tqdm_desc = f'{self.imgs_dir}: Downloading {len(missing)} missing images'
            )
        return self 


            # desc = f'{self.imgs_dir}: Downloading {len(missing)} missing images'
            # urls = self.df.iloc[missing]['img_url']
            # paths = self.df.iloc[missing]['img_path']
