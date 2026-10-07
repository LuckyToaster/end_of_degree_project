import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from src.scripts.platemate_agent import build_platemate, read_image_base64
from src.helpers.ml import three_way_split, get_scaler_on_train
from src.constants import CSV_PATH

TARGETS = ['fat_g', 'carb_g', 'prot_g']
SEED = 1

def validate_platemate():
    # Load data and scaler
    _, _, hidden_df = three_way_split(CSV_PATH, TARGETS, SEED)
    scaler = get_scaler_on_train(CSV_PATH, TARGETS, SEED)

    # Inverse transform to get grams
    hidden_grams = scaler.inverse_transform(hidden_df[TARGETS])
    hidden_df_grams = hidden_df.copy()
    hidden_df_grams[TARGETS] = hidden_grams

    # Use first 100 images
    hidden_df_grams = hidden_df_grams.iloc[:100]

    model = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')
    pipeline = build_platemate(model)

    errors = []

    for _, row in tqdm(hidden_df_grams.iterrows(), total=len(hidden_df_grams), desc="Validating PlateMate"):
        img_path = row['img_path']
        # Ground truth in grams
        gt_fat = row['fat_g']
        gt_carb = row['carb_g']
        gt_prot = row['prot_g']

        # Input
        input_data = {
            'image_b64': read_image_base64(img_path),
            'datetime': 'current time not available',
            'coordinates': 'location not available'
        }

        try:
            # Call agent
            out_str = pipeline.invoke(input_data)
            out = json.loads(out_str)
            # Parse output
            # Keys expected: protein_g, carbohydrate_g, fat_g
            pred_fat = out.get('fat_g', 0)
            pred_carb = out.get('carbohydrate_g', 0)
            pred_prot = out.get('protein_g', 0)
        except Exception as e:
            # If JSON parsing fails or other issue, treat as 0
            pred_fat, pred_carb, pred_prot = 0, 0, 0

        errors.append([
            abs(pred_fat - gt_fat),
            abs(pred_carb - gt_carb),
            abs(pred_prot - gt_prot)
        ])

    errors = np.array(errors)
    mae = np.mean(errors, axis=0)

    print("\nPlateMate Validation Results (100 images):")
    print(f"Fat MAE: {mae[0]:.2f}g")
    print(f"Carbs MAE: {mae[1]:.2f}g")
    print(f"Protein MAE: {mae[2]:.2f}g")

if __name__ == "__main__":
    validate_platemate()
