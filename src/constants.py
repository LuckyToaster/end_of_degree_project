DATASET_URL = 'hf://datasets/Codatta/MM-Food-100K/MM-Food-100K.csv' # https://huggingface.co/datasets/Codatta/MM-Food-100K
DF_COLS_TO_KEEP = ['img_url', 'dish_name', 'ingredient', 'cooking_method', 'img_path', 'fat_g', 'carb_g', 'prot_g', 'kcal'] 
IMG_DISK_SIZE = 512

CSV_PATH = 'data/food_dataset.csv'
IMGS_DIR = 'data/imgs'
OPTUNA_DB_PATH = 'data/optuna.db'
MODEL_CHECKPOINTS_DIR = 'data/checkpoints'

FIGURES_DIR = 'docs/figures'
ONNX_DIR = 'docs/model_graphs'
