import pandas as pd
from pathlib import Path
import json
from src.helpers import download_images, check_corrupted_imgs, resize_images_pil, resize_images_tv, check_corrupted_imgs_v2
# from os import path


class MMFood100KBuilder:
    URL = 'hf://datasets/Codatta/MM-Food-100K/MM-Food-100K.csv' # https://huggingface.co/datasets/Codatta/MM-Food-100K

    def __init__(self, data_path, img_size):
        self.dir = data_path / 'mm-food-100k'
        self.csv_path = self.dir / 'mm-food-100k.csv'
        self.imgs_dir = self.dir / 'imgs'
        self.resized_imgs_dir = self.dir / 'resized_imgs'
        self.img_size = img_size

        self.dir.mkdir(parents=True, exist_ok=True)
        self.imgs_dir.mkdir(parents=True, exist_ok=True)
        self.resized_imgs_dir.mkdir(parents=True, exist_ok=True)

        if self.csv_path.is_file(): 
            self.df = pd.read_csv(self.csv_path)
            return 

        print(f'Downloading {self.csv_path}')
        self.df = pd.read_csv(self.URL)
        self.df = self.df.rename(columns={'image_url': 'img_url'})
        img_suffixes = self.df['img_url'].apply(lambda x: Path(x).suffix).tolist()
        self.df['img_path'] = str(self.imgs_dir) + '/' + self.df.index.astype(str) + img_suffixes
        self.df['resized_img_path'] = str(self.resized_imgs_dir) + '/' + self.df.index.astype(str) + '.jpeg'
        target_cols = ['fat_g', 'carb_g', 'protein_g', 'kcal']
        target_names = ['fat_g', 'carbohydrate_g', 'protein_g', 'calories_kcal']
        self.df[target_cols] = self.df['nutritional_profile'].apply(lambda x: pd.Series(json.loads(x)))[target_names]
        self.df[target_cols] = self.df[target_cols].astype(float)
        self.df = self.df.drop(columns=['camera_or_phone_prob', 'nutritional_profile'])

        print(f'Saving wrangled CSV')
        self.df.to_csv(self.csv_path, index=False)


    def _file_img_ids(self, dir):
        return [int(i.name.split('.')[0]) for i in dir.iterdir()]

    def _missing_img_ids(self, dir):
        if not any(dir.iterdir()): return sorted(range(0, 100_000))
        return sorted(set(range(0, 100_000)) - set(self._file_img_ids(dir)))


    async def download_imgs(self, limit=10):
        while True:
            missing = self._missing_img_ids(self.imgs_dir)
            if not missing: break
            urls = self.df.iloc[missing]['img_url'].tolist()
            paths = self.df.iloc[missing]['img_path'].tolist()
            await download_images(urls, paths, limit, f'{self.imgs_dir} => Downloading any missing images')
        return self 


    async def fix_corrupted_imgs(self):
        corrupted_paths = [path for path,_ in check_corrupted_imgs_v2(self.imgs_dir)]
        if not corrupted_paths: 
            print('No Corrupted Images ✅')
            return 

        urls = self.df[self.df['img_path'].isin(corrupted_paths)]['img_url']
        await download_images(urls, corrupted_paths, 10, f'{self.imgs_dir} => Re-Downloading corrupted images')


    def resize_images(self):
        resize_images_tv(self.df['img_path'].tolist(), self.df['resized_img_path'].tolist(), self.img_size)
