from torch.utils.data import Dataset
from PIL import Image 
from torch import from_numpy
from random import randint

class FoodDataset(Dataset):
    def __init__(self, df, transform, input='img_path', targets=['fat_g', 'protein_g', 'carb_g']):
        self.transform = transform
        self.paths = df[input].tolist() # python list for fast indexing
        self.targets = df[targets].values.astype('float32') # numpy array for speed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        input = Image.open(self.paths[idx]).convert('RGB') 
        input = self.transform(input) if self.transform else input
        targets = from_numpy(self.targets[idx])
        return input, targets
