from src.helpers.images import download_images, resize_images, get_corrupted_images
from src.helpers.misc import remove_files, missing_paths
import pandas as pd
from pathlib import Path
import json


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


    async def download_imgs(self, limit=10):
        while True:
            missing = missing_paths(self.df['img_path'].tolist())
            if not missing: break
            mask = self.df['img_path'].isin(missing)
            urls = self.df.loc[mask, 'img_url'].tolist()
            paths = self.df.loc[mask, 'img_path'].tolist()
            await download_images(urls, paths, limit, f'{self.imgs_dir} => Downloading any missing images')


    async def drop_corrupted_imgs(self):
        corrupted_paths = [path for path,_ in get_corrupted_images(self.imgs_dir)]
        if not corrupted_paths:
            print('No Corrupted Images ✅')
            return 

        remove_files(corrupted_paths, f'{self.imgs_dir} => Removing corrupted images', 'img') 
        self.df = self.df[ ~self.df['img_path'].isin(corrupted_paths) ]
        self.df.to_csv(self.csv_path, index=False)


    def resize_images(self):
        missing = missing_paths(self.df['resized_img_path'].tolist())
        if not missing: return
        mask = self.df['resized_img_path'].isin(missing)
        src_paths = self.df.loc[mask, 'img_path'].tolist()
        dst_paths = self.df.loc[mask, 'resized_img_path'].tolist()
        resize_images(src_paths, dst_paths, self.img_size)
