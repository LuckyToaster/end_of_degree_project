from torch.utils.data import Dataset
from torch import tensor
from PIL import Image
from torchvision import tv_tensors as tv
from PIL import ImageFile

# Ignore truncated file error
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MMFood100KDataset(Dataset):
    def __init__(self, df, transform, device, input='img_path', targets=['fat_g', 'protein_g', 'carb_g']):
        self.df = df[ targets + [input] ] 
        self.transform = transform
        self.targets = targets
        self.input = input
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #input = tv.Image(Image.open(self.df.iloc[idx][self.input]).convert('RGB'), device=self.device)
        input = Image.open(self.df.iloc[idx][self.input]).convert('RGB') 
        targets = tensor(self.df.iloc[idx][self.targets].values.astype('float32'))

        if self.transform: 
            input = self.transform(input)
        return input, targets
