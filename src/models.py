from torchviz import make_dot
import torch
from torch.nn.init import xavier_uniform_
from torch.nn import Linear
from torchvision.transforms import v2
from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    swin_v2_s, Swin_V2_S_Weights
)

__all__ = [
    'get_EfficientNet_B3', 
    'get_EfficientNet_V2_S', 
    'get_MobileNet_V3_L', 
    'get_Swin_V2_S',
]


def get_EfficientNet_B3(n_targets: int, verbose: bool =True):
    weights = EfficientNet_B3_Weights.DEFAULT
    model = efficientnet_b3(weights=weights)
    preprocess = weights.transforms()
    model.classifier[1] = Linear(model.classifier[1].in_features, n_targets) # adapt the head for regression
    # xavier_uniform_(model.classifier[1].weight)
    if verbose: print(f'EfficientNet B3: {preprocess}')
    return model, preprocess


def get_EfficientNet_V2_S(n_targets: int, verbose=True):
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    preprocess = weights.transforms()
    model.classifier[1] = Linear(model.classifier[1].in_features, n_targets) # adapt the head for regression
    # xavier_uniform_(model.classifier[1].weight)
    if verbose: print(f'EfficientNet V2 Small: {preprocess}')
    return model, preprocess


def get_MobileNet_V3_L(n_targets: int, verbose=True):
    weights = MobileNet_V3_Large_Weights.DEFAULT
    preprocess = weights.transforms()
    model = mobilenet_v3_large(weights=weights)
    model.classifier[3] = Linear(model.classifier[3].in_features, n_targets) # adapt the head for regression
    if verbose: print(f'MobileNet V3 Large: {preprocess}')
    return model, preprocess


def get_Swin_V2_S(n_targets: int, verbose=False):
    weights = Swin_V2_S_Weights.DEFAULT
    val_transforms = weights.transforms() # default validation / inference transforms
    model = swin_v2_s(weights=weights)
    model.head = Linear(model.head.in_features, n_targets) # adapt the head for regression

    # if verbose: print(model)
    # if visualize: # https://stackoverflow.com/questions/52468956/how-do-i-visualize-a-net-in-pytorch
    #     device = torch.device('cuda')
    #     model.to(device)
    #     yhat = model(torch.randn(1, 3, 256, 256, device=device))
    #     make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")

    # transforms with data augmentation for training
    train_transforms = v2.Compose([
        v2.RandomResizedCrop(size=val_transforms.crop_size[0], scale=(0.8, 1.0), interpolation=v2.InterpolationMode.BICUBIC),
        v2.TrivialAugmentWide(interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=val_transforms.mean, std=val_transforms.std)
    ])
    return model, train_transforms, val_transforms

    # train_transforms = v2.Compose([
    #     v2.RandomResizedCrop(size=256, scale=(0.8, 1.0), interpolation=v2.InterpolationMode.BICUBIC),
    #     v2.TrivialAugmentWide(interpolation=v2.InterpolationMode.BICUBIC), # flips, rotations and colorjitters
    #     # ImageNet and swin default transforms
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

