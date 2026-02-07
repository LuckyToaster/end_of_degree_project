from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torch import tensor, from_numpy
# from torchvision import tv_tensors as tv

# Ignore truncated file error
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MMFood100KDataset(Dataset):
    def __init__(self, df, transform, input='img_path', targets=['fat_g', 'protein_g', 'carb_g']):
        self.transform = transform
        self.paths = df[input].tolist() # python list for fast indexing
        self.targets = df[targets].values.astype('float32') # numpy array for speed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        input = Image.open(self.paths[idx]).convert('RGB') 
        targets = from_numpy(self.targets[idx])

        if self.transform:
            input = self.transform(input)
        return input, targets
