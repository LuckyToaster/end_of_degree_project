from torchvision.models import \
    efficientnet_b0, EfficientNet_B0_Weights, \
    efficientnet_b3, EfficientNet_B3_Weights, \
    efficientnet_v2_s, EfficientNet_V2_S_Weights, \
    mobilenet_v3_large, MobileNet_V3_Large_Weights, \
    shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights

from torch.nn.init import xavier_uniform_
from torch.nn import Linear


def get_EfficentNet_B0():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    preprocess = weights.transforms()
    # adapt the head for regression
    in_features = model.classifier[1].in_features
    model.classifier[1] = Linear(in_features, 3)
    xavier_uniform_(model.classifier[1].weight)
    print(f'MobileNet V3 Large: {preprocess}')
    return model, preprocess


def get_EfficentNet_B3():
    weights = EfficientNet_B3_Weights.DEFAULT
    model = efficientnet_b3(weights=weights)
    preprocess = weights.transforms()
    # adapt the head for regression
    in_features = model.classifier[1].in_features
    model.classifier[1] = Linear(in_features, 3)
    xavier_uniform_(model.classifier[1].weight)
    print(f'EfficientNet B3: {preprocess}')
    return model, preprocess


def get_EfficientNet_V2_S():
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    preprocess = weights.transforms()
    # adapt the head for regression
    in_features = model.classifier[1].in_features
    model.classifier[1] = Linear(in_features, 3)
    xavier_uniform_(model.classifier[1].weight)
    print(f'EfficientNet V2 Small: {preprocess}')
    return model, preprocess


def get_MobileNet_V3_L():
    weights = MobileNet_V3_Large_Weights.DEFAULT
    preprocess = weights.transforms()
    model = mobilenet_v3_large(weights=weights)
    # adapt the head for regression
    last_channel = model.classifier[3].in_features
    model.classifier[3] = Linear(last_channel, 3)
    print(f'MobileNet V3 Large: {preprocess}')
    return model, preprocess


def get_ShuffleNet_V2_X2_0_Weights():
    '''
    torchvision MobileNet source code => https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py#L193
    '''
    weights = ShuffleNet_V2_X2_0_Weights.DEFAULT
    preprocess = weights.transforms()
    model = shufflenet_v2_x2_0(weights=weights, num_classes=3)
    print(f'ShuffleNet V2 2: {preprocess}')
    return model, preprocess

