from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torch import tensor, from_numpy
from random import randint
# from torchvision import tv_tensors as tv
# Ignore truncated file error
# ImageFile.LOAD_TRUNCATED_IMAGES = True

class MMFood100KDataset(Dataset):
    def __init__(self, df, transform, input='img_path', targets=['fat_g', 'protein_g', 'carb_g']):
        self.transform = transform
        self.paths = df[input].tolist() # python list for fast indexing
        self.targets = df[targets].values.astype('float32') # numpy array for speed

    def _get_rand_item(self):
        idx = randint(0, len(self))
        input = Image.open(self.paths[idx]).convert('RGB') 
        input = self.transform(input) if self.transform else input
        targets = from_numpy(self.targets[idx])
        return input, targets

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            input = Image.open(self.paths[idx]).convert('RGB') 
            input = self.transform(input) if self.transform else input
            targets = from_numpy(self.targets[idx])
            return input, targets
        except Exception as e:
            print(f'DataLoader: {e}')
            return self._get_rand_item()


            
