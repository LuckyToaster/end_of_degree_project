import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def plot_distributions():
    # Load dataset
    df = pd.read_csv('data/food_dataset.csv')

    # Calculate total portion
    df['portion_g'] = df['fat_g'] + df['carb_g'] + df['prot_g']

    # Define plots
    metrics = [
        ('kcal', 'Energy (kcal) Distribution', 'Energy (kcal)'),
        ('carb_g', 'Carbs (g) Distribution', 'Carbs (g)'),
        ('fat_g', 'Fat (g) Distribution', 'Fat (g)'),
        ('prot_g', 'Protein (g) Distribution', 'Protein (g)'),
        ('portion_g', 'Overall Portion (g) Distribution', 'Portion (g)')
    ]

    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[m[1] for m in metrics]
    )

    # Add traces
    for i, (col, title, x_label) in enumerate(metrics):
        row = (i // 3) + 1
        col_idx = (i % 3) + 1
        
        median = df[col].median()
        
        fig.add_trace(
            go.Histogram(x=df[col], name=x_label, marker_color='lightblue', nbinsx=50),
            row=row, col=col_idx
        )
        # Add vline (no annotation text here)
        fig.add_vline(x=median, line_dash="dot", line_color="black", 
                      row=row, col=col_idx)

        # Add median as an annotation in the top-right corner
        fig.add_annotation(
            xref="x domain", yref="y domain",
            x=0.95, y=0.95,
            text=f"Median: {median:.1f}",
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="white",
            opacity=0.8,
            row=row, col=col_idx
        )
        
        fig.update_xaxes(title_text=x_label, row=row, col=col_idx)
        fig.update_yaxes(title_text="Frequency", row=row, col=col_idx)

    fig.update_layout(height=800, width=1200, title_text="Nutritional Distribution", showlegend=False)

    # Save to file
    os.makedirs('data/plots', exist_ok=True)
    fig.write_image('data/plots/food_distribution.png')
    print("Plot saved to data/plots/food_distribution.png")

if __name__ == "__main__":
    plot_distributions()
