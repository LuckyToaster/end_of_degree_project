from torch.utils.data import Dataset
from torch import tensor
from PIL import Image, ImageFile
from sys import stderr
#import random


class MMFood100KDataset(Dataset):
    def __init__(self, df, transform=None, input='img_path', targets=['fat_g', 'protein_g', 'carb_g']):
        self.df = df[ targets + [input] ] 
        self.transform = transform
        self.targets = targets
        self.input = input

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input = Image.open(self.df.iloc[idx][self.input]).convert('RGB')
        targets = tensor(self.df.iloc[idx][self.targets].values.astype('float32'))

        if self.transform: 
            input = self.transform(input)
        return input, targets
