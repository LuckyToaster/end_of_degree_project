import argparse
import sys
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

from src.constants import CSV_PATH, FIGURES_DIR

def parse_args():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis (EDA) tools for the food dataset.")
    parser.add_argument("--distribution", action="store_true", help="Plot the distribution of nutritional targets.")
    parser.add_argument("--sample", action="store_true", help="Plot a grid sample of dataset images and their targets.")
    parser.add_argument("--n-samples", type=int, default=9, help="Number of samples to plot (default: 9). Must be a perfect square.")
    
    args = parser.parse_args()
    if not any([args.distribution, args.sample]):
        parser.print_help()
        print("\nPlease specify at least one plot flag (e.g., --distribution or --sample).")
        sys.exit(1)
    return args

def load_dataset():
    if not Path(CSV_PATH).exists():
        print(f"Error: Dataset not found at {CSV_PATH}. Run 'python src/scripts/load_data.py' first.")
        sys.exit(1)
    return pd.read_csv(CSV_PATH)

def plot_distribution(df):
    print("Generating distribution plots...")
    targets = ['fat_g', 'carb_g', 'prot_g', 'kcal']
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=targets)
    
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for target, pos in zip(targets, positions):
        # Using a histogram with marginal boxplot for comprehensive view
        trace = go.Histogram(x=df[target], name=target, nbinsx=50)
        fig.add_trace(trace, row=pos[0], col=pos[1])
        
    fig.update_layout(title_text="Nutritional Target Distributions", height=800, showlegend=False)
    
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    dst = f"{FIGURES_DIR}/target_distributions.png"
    fig.write_image(dst)
    print(f"Distribution plot saved to {dst}")

def plot_dataset_sample(df, n_samples):
    import math
    print(f"Generating dataset sample grid ({n_samples} images)...")
    
    grid_size = int(math.sqrt(n_samples))
    if grid_size * grid_size != n_samples:
         print(f"Warning: n-samples ({n_samples}) is not a perfect square. Using {grid_size*grid_size} instead.")
         n_samples = grid_size * grid_size
         
    sample_df = df.sample(n=n_samples, random_state=42)
    
    fig = make_subplots(
        rows=grid_size, cols=grid_size, 
        subplot_titles=[f"Kcal: {r['kcal']:.1f}<br>F:{r['fat_g']:.1f} C:{r['carb_g']:.1f} P:{r['prot_g']:.1f}" for _, r in sample_df.iterrows()],
        horizontal_spacing=0.05, vertical_spacing=0.1
    )
    
    for i, (_, row) in enumerate(sample_df.iterrows()):
        r = (i // grid_size) + 1
        c = (i % grid_size) + 1
        
        try:
            img = Image.open(row['img_path']).convert('RGB')
            fig.add_trace(go.Image(z=img), row=r, col=c)
        except Exception as e:
            print(f"Error loading image {row['img_path']}: {e}")
            
    fig.update_layout(height=300 * grid_size, width=300 * grid_size, title_text="Dataset Samples")
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    dst = f"{FIGURES_DIR}/dataset_sample.png"
    fig.write_image(dst)
    print(f"Sample grid saved to {dst}")

def main():
    args = parse_args()
    df = load_dataset()
    
    if args.distribution:
        plot_distribution(df)
        
    if args.sample:
        plot_dataset_sample(df, args.n_samples)

if __name__ == "__main__":
    main()
