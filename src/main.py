from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from torch.nn import Linear
from src.datasets.mmfood100k import MMFood100KDataset
# import torchvision.models as models
# print(models.list_models())

weights = EfficientNet_B0_Weights.DEFAULT
preprocess = weights.transforms() # img_transformed = preprocess(img)
model = efficientnet_b0(weights=weights)

dataset = MMFood100KDataset(csv_path='data/mm-food-100k/mm-food-100k.csv', transform=preprocess)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

'''
change output layer from Linear(hidden, classes) to Linear(hideen, 3)
change the final activation from softmax/logsoftmax to None
change criterion/loss from Cross-Entropy (classification) to MSE or MAE for regression
'''

in_features = model.classifier[1].in_features
model.classifier[1] = Linear(in_features, 3)


