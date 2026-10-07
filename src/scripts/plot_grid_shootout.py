from PIL import Image
import os

def create_grid():
    image_paths = [
        'data/plots/shootout_EfficientNet_B3.png',
        'data/plots/shootout_EfficientNet_V2_S.png',
        'data/plots/shootout_MobileNet_V3_L.png',
        'data/plots/shootout_Swin_V2_S.png'
    ]

    # Open all images
    images = [Image.open(p) for p in image_paths]
    
    # Assuming all images are the same size, get dimensions
    width, height = images[0].size
    
    # Create a new image for the grid (2x2)
    grid_width = width * 2
    grid_height = height * 2
    grid_image = Image.new('RGB', (grid_width, grid_height))
    
    # Paste images into the grid
    grid_image.paste(images[0], (0, 0))
    grid_image.paste(images[1], (width, 0))
    grid_image.paste(images[2], (0, height))
    grid_image.paste(images[3], (width, height))
    
    # Save the result
    grid_image.save('data/plots/model_shootout_grid.png')
    print("Grid plot saved to data/plots/model_shootout_grid.png")

if __name__ == "__main__":
    create_grid()
