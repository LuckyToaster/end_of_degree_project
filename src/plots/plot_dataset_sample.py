import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import numpy as np
import os

def plot_sample():
    # Load dataset
    df = pd.read_csv('data/food_dataset.csv')
    
    # Sample 12 images
    sample = df.sample(12, random_state=42)

    # Create subplots without titles, with reduced spacing
    fig = make_subplots(
        rows=3, cols=4,
        horizontal_spacing=0.01,
        vertical_spacing=0.01
    )

    for i, (idx, row) in enumerate(sample.iterrows()):
        img_path = row['img_path']
        
        # Load and convert image to numpy array
        try:
            img = Image.open(img_path)
            # Ensure it's in a format plotly handles (e.g., RGB)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")
            img_array = np.zeros((100, 100, 3), dtype=np.uint8) # Placeholder

        row_idx = (i // 4) + 1
        col_idx = (i % 4) + 1
        
        fig.add_trace(go.Image(z=img_array), row=row_idx, col=col_idx)

    # Further reduce margins to make it compact
    fig.update_layout(
        height=800, width=1200, 
        title_text="Sample Dataset Images", 
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Hide axes
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Save to file
    os.makedirs('data/plots', exist_ok=True)
    fig.write_image('data/plots/dataset_sample.png')
    print("Sample plot saved to data/plots/dataset_sample.png")

if __name__ == "__main__":
    plot_sample()
