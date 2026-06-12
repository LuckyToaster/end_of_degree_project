from torch.nn.init import xavier_uniform_
from torch.nn import Linear
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
    'freeze',
    'unfreeze'
]


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

def get_EfficientNet_B3(feature_extraction=False, verbose=True):
    weights = EfficientNet_B3_Weights.DEFAULT
    model = efficientnet_b3(weights=weights)
    preprocess = weights.transforms()
    if feature_extraction: freeze(model)
    # adapt the head for regression
    in_features = model.classifier[1].in_features
    model.classifier[1] = Linear(in_features, 3)
    xavier_uniform_(model.classifier[1].weight)
    if verbose: print(f'EfficientNet B3: {preprocess}')
    return model, preprocess


def get_EfficientNet_V2_S(feature_extraction=False, verbose=True):
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    preprocess = weights.transforms()
    if feature_extraction: freeze(model)
    # adapt the head for regression
    in_features = model.classifier[1].in_features
    model.classifier[1] = Linear(in_features, 3)
    xavier_uniform_(model.classifier[1].weight)
    if verbose: print(f'EfficientNet V2 Small: {preprocess}')
    return model, preprocess


def get_MobileNet_V3_L(feature_extraction=False, verbose=True):
    weights = MobileNet_V3_Large_Weights.DEFAULT
    preprocess = weights.transforms()
    model = mobilenet_v3_large(weights=weights)
    if feature_extraction: freeze(model)
    # adapt the head for regression
    last_channel = model.classifier[3].in_features
    model.classifier[3] = Linear(last_channel, 3)
    if verbose: print(f'MobileNet V3 Large: {preprocess}')
    return model, preprocess


def get_Swin_V2_S(feature_extraction=False, verbose=True, modify_head=True):
    weights = Swin_V2_S_Weights.DEFAULT
    preprocess = weights.transforms()
    model = swin_v2_s(weights=weights)
    if feature_extraction: freeze(model)
    if modify_head: model.head = Linear(model.head.in_features, 3)
    if verbose: print(f'Swin V2 Small: {preprocess}')
    return model, preprocess
