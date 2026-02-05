import fireducks.pandas as pd
from pathlib import Path
import json
from helpers import download_images


class MMFood100KBuilder:
    URL = 'hf://datasets/Codatta/MM-Food-100K/MM-Food-100K.csv' # https://huggingface.co/datasets/Codatta/MM-Food-100K
    PIXEL_LIMIT = 89_478_485 # matches PIL's
    MAX_DIM = 2000  

    def __init__(self, data_path):
        self.dir = data_path / 'mm-food-100k'
        self.csv_path = self.dir / 'mm-food-100k.csv'
        self.imgs_dir = self.dir / 'imgs'

        self.dir.mkdir(parents=True, exist_ok=True)
        self.imgs_dir.mkdir(parents=True, exist_ok=True)
        if self.csv_path.is_file(): return 

        print(f'Downloading {self.csv_path}')
        df = pd.read_csv(self.URL)
        df = df.rename(columns={'image_url': 'img_url'})
        df['img_suffix'] = df['img_url'].apply(lambda x: Path(x).suffix)
        df['img_path'] = str(self.imgs_dir) + '/' + df.index.astype(str) + df['img_suffix']
        target_cols = ['fat_g', 'carb_g', 'protein_g', 'kcal']
        target_names = ['fat_g', 'carbohydrate_g', 'protein_g', 'calories_kcal']
        df[target_cols] = df['nutritional_profile'].apply(lambda x: pd.Series(json.loads(x)))[target_names]
        df[target_cols] = df[target_cols].astype(float)
        df = df.drop(columns=['img_suffix', 'camera_or_phone_prob', 'nutritional_profile'])
        df.to_csv(self.csv_path, index=False)

    def _file_img_ids(self):
        return [int(i.name.split('.')[0]) for i in self.imgs_dir.iterdir()]

    def _missing_img_ids(self):
        if not any(self.imgs_dir.iterdir()): return sorted(range(0, 100_000))
        return sorted(set(range(0, 100_000)) - set(self._file_img_ids()))

    async def download_imgs(self, limit=10):
        df = pd.read_csv(self.csv_path)
        while True:
            missing = self._missing_img_ids()
            if not missing: break
            await download_images(
                df.iloc[missing]['img_url'], 
                df.iloc[missing]['img_path'], 
                limit, 
                f'{self.imgs_dir}: Downloading {len(missing)} missing images'
            )
        return self 
