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

    def __get_present_imgs_ids(self):
        return [int(i.name.split('.')[0]) for i in self.imgs_dir.iterdir()]

    def __get_missing_imgs_ids(self):
        if not any(self.imgs_dir.iterdir()): return sorted(range(0, 100_000))
        return sorted(set(range(0, 100_000)) - set(self.__get_present_imgs_ids()))

    def __log_test(self, testnum, passed):
        msg = f'{str(self.dir)} TEST {testnum} {'PASSED ✅' if passed else ' FAILED ❌'}'
        print(msg, file=stdout if passed else stderr)

    async def download_imgs(self, limit=100):
        while True:
            missing = self.__get_missing_imgs_ids()
            if not missing: break
            desc = f'{self.imgs_dir}: Downloading {len(missing)} missing images'
            urls = self.df.iloc[missing]['img_url']
            paths = self.df.iloc[missing]['img_path']
            await download_images(urls, paths, limit, desc)
        print(f'{self.imgs_dir} finished downloading')
        return self 

    def tests(self):
        for path in self.df.iloc[self.__get_missing_imgs_ids()]['img_path']:
            if Path(path).is_file(): 
                self.__log_test(1, False)
                break
        self.__log_test(1, True)
        present = set(self.__get_present_imgs_ids())
        missing = set(self.__get_missing_imgs_ids())
        if len(present.intersection(missing)) > 0: self.__log_test(2, False)
        else: self.__log_test(2, True)
        return self
