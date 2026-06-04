from pathlib import Path
import pandas as pd
from pandas import DataFrame
import json
from src.helpers.images import get_corrupted_images, download_and_resize_images
from src.helpers.files import remove_files 

URL = 'hf://datasets/Codatta/MM-Food-100K/MM-Food-100K.csv' # https://huggingface.co/datasets/Codatta/MM-Food-100K
COLS_TO_KEEP = ['img_url', 'dish_name', 'ingredient', 'cooking_method', 'img_path', 'fat_g', 'carb_g', 'protein_g', 'kcal'] 
DATA_DIR = 'data'
IMG_DIR = 'data/imgs'
CSV_PATH = 'data/food_dataset.csv'
IMG_SIZE = 512


def main():
    Path(DATA_DIR).mkdir(exist_ok=True)
    Path(IMG_DIR).mkdir(exist_ok=True, parents=True)

    if Path(CSV_PATH).exists(): 
        df = pd.read_csv(CSV_PATH)
    else: 
        df = pd.read_csv(URL)
        df = reshape(df)
        df.to_csv(CSV_PATH, index=False)

    missing_urls, missing_paths = get_missing_urls_and_paths(df)
    download_and_resize_images(missing_urls, missing_paths, size=IMG_SIZE)

    # removed corrupted images from IMG_DIR and from .csv 
    corrupted = [ path for path, _ in get_corrupted_images(IMG_DIR)]
    if corrupted: 
        remove_files(corrupted)
        df = df[~df['img_path'].isin(corrupted)] 
        df.to_csv(CSV_PATH, index=False)


def reshape(df: DataFrame):
    df = df.rename(columns={'image_url': 'img_url'})
    # create and populate img_path column
    df['img_path'] = IMG_DIR + '/' + df.index.astype(str) + '.jpeg'
    # extract the targets from nutritional_profile column into their own columns
    new_targets = [ 'fat_g', 'carb_g', 'prot_g', 'kcal' ]
    old_targets = [ 'fat_g', 'carbohydrate_g', 'protein_g', 'calories_kcal']
    df[new_targets] = df['nutritional_profile'].apply(lambda x: pd.Series(json.loads(x)))[old_targets].astype(float)
    # drop unneeded columns
    to_drop = list(set(df.columns) - set(COLS_TO_KEEP))
    df = df.drop(columns=to_drop)
    return df


def get_missing_urls_and_paths(df: DataFrame):
    missing_imgs = [p for p in df['img_path'] if not Path(p).exists()]
    if not missing_imgs: return [], []
    mask = df['img_path'].isin(missing_imgs)
    missing_urls = df.loc[mask, 'img_url'].tolist()
    missing_paths = df.loc[mask, 'img_path'].tolist()
    return missing_urls, missing_paths
    

if __name__ == '__main__':
    main()


