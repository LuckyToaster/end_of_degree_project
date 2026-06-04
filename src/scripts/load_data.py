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
        print('Downloading .csv')
        df = pd.read_csv(URL)
        df = reshape(df)
        df.to_csv(CSV_PATH, index=False)

    # download missing images but remove from csv failed downloads / reshapes
    missing_urls, missing_paths = get_missing_urls_and_paths(df)
    failed_urls = download_and_resize_images(missing_urls, missing_paths, size=IMG_SIZE)
    if failed_urls: df = df.loc[~df['img_url'].isin(failed_urls)]

    # check for corrupted images and remove them
    corrupted_paths = [path for path, _ in get_corrupted_images(IMG_DIR)]
    if corrupted_paths: 
        remove_files(corrupted_paths)
        df = df[~df['img_path'].isin(corrupted_paths)] 

    print(f'Removing {len(failed_urls)} failed downloads/resizes and {len(corrupted_paths)} corrupted images')
    
    # save the cleaned csv
    df.to_csv(CSV_PATH, index=False)
    print(f"Dataset successfully cleaned and saved. Total valid images: {len(df)}")


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
