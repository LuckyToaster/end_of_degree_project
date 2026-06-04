from src.helpers.images import get_corrupted_images, download_and_resize_images
from src.helpers.files import remove_files, missing_paths
import pandas as pd
from pathlib import Path
import json
import os


class MMFood100KBuilder:
    URL = 'hf://datasets/Codatta/MM-Food-100K/MM-Food-100K.csv' # https://huggingface.co/datasets/Codatta/MM-Food-100K
    COLS_TO_KEEP = ['img_url', 'dish_name', 'ingredient', 'cooking_method', 'img_path', 'fat_g', 'carb_g', 'protein_g', 'kcal'] 

    def __init__(self, img_size):
        self.dir = Path('data')
        self.csv_path = self.dir / 'data.csv'
        self.imgs_dir = self.dir / 'imgs'

        self.dir.mkdir(parents=True, exist_ok=True)
        self.imgs_dir.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            print(f'Downloading {self.csv_path}')
            self.df = pd.read_csv(self.URL)

            self.df = self.df.rename(columns={'image_url': 'img_url'})
            img_suffixes = self.df['img_url'].apply(lambda x: Path(x).suffix).tolist()
            self.df['img_path'] = str(self.imgs_dir) + '/' + self.df.index.astype(str) + img_suffixes
            # self.df['resized_img_path'] = str(self.resized_imgs_dir) + '/' + self.df.index.astype(str) + '.jpeg'

            target_cols = ['fat_g', 'carb_g', 'protein_g', 'kcal']
            target_names = ['fat_g', 'carbohydrate_g', 'protein_g', 'calories_kcal']
            self.df[target_cols] = self.df['nutritional_profile'].apply(lambda x: pd.Series(json.loads(x)))[target_names]
            self.df[target_cols] = self.df[target_cols].astype(float)

            cols_to_drop = list(set(self.df.columns) - set(self.COLS_TO_KEEP))
            self.df = self.df.drop(columns=cols_to_drop)

            print(f'Saving wrangled CSV')
            self.df.to_csv(self.csv_path, index=False)
        else:
            print(f'Loading existing CSV from {self.csv_path}')
            self.df = pd.read_csv(self.csv_path)


    def _get_missing_paths(self):
        return [p for p in self.df['img_path'] if not Path(p).exists()]


    async def drop_corrupted_imgs(self):
        corrupted_paths = [path for path,_ in get_corrupted_images(self.imgs_dir)]
        if not corrupted_paths:
            print('No Corrupted Images ✅')
            return 

        remove_files(corrupted_paths, f'{self.imgs_dir} => Removing corrupted images', 'img') 
        self.sync_csv_with_files(check_resized=False)


    def download_and_resize_images(self):
        missing = missing_paths(self.df['resized_img_path'].tolist())
        if not missing: return
        mask = self.df['resized_img_path'].isin(missing)
        urls = self.df.loc[mask, 'img_url'].tolist()
        dst_paths = self.df.loc[mask, 'resized_img_path'].tolist()
        download_and_resize_images(urls, dst_paths, size=self.img_size)


    def sync_csv_with_files(self, check_resized=True):
        '''
            Removes rows from the CSV if the corresponding image files don't exist.
        '''
        initial_len = len(self.df)
        
        # Check source images
        self.df = self.df[self.df['img_path'].apply(os.path.exists)]
        # Check resized images if requested
        if check_resized:
            self.df = self.df[self.df['resized_img_path'].apply(os.path.exists)]
            
        dropped = initial_len - len(self.df)
        if dropped > 0:
            print(f'Syncing CSV: Dropped {dropped} rows due to missing files.')
            self.df.to_csv(self.csv_path, index=False)


