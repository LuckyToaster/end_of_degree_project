from torch.nn.init import xavier_uniform_
from torch.nn import Linear
from torchvision.models import \
    efficientnet_b3, EfficientNet_B3_Weights, \
    efficientnet_v2_s, EfficientNet_V2_S_Weights, \
    efficientnet_v2_m, EfficientNet_V2_M_Weights, \
    efficientnet_v2_l, EfficientNet_V2_L_Weights, \
    mobilenet_v3_large, MobileNet_V3_Large_Weights, \
    shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights, \
    swin_v2_t, Swin_V2_T_Weights, \
    swin_v2_s, Swin_V2_S_Weights, \
    swin_v2_b, Swin_V2_B_Weights \

__all__ = [
    'get_EfficientNet_B3', 'get_EfficientNet_V2_S', 'get_EfficientNet_V2_M', 'get_EfficientNet_V2_L',
    'get_MobileNet_V3_L', 
    'get_ShuffleNet_V2_X2_0',
    'get_Swin_V2_T', 'get_Swin_V2_S', 'get_Swin_V2_B'
]


def get_EfficientNet_B3():
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


def get_EfficientNet_V2_M():
    weights = EfficientNet_V2_M_Weights.DEFAULT
    model = efficientnet_v2_m(weights=weights)
    preprocess = weights.transforms()
    # adapt the head for regression
    in_features = model.classifier[1].in_features
    model.classifier[1] = Linear(in_features, 3)
    xavier_uniform_(model.classifier[1].weight)
    print(f'EfficientNet V2 Medium: {preprocess}')
    return model, preprocess


def get_EfficientNet_V2_L():
    weights = EfficientNet_V2_L_Weights.DEFAULT
    model = efficientnet_v2_l(weights=weights)
    preprocess = weights.transforms()
    # adapt the head for regression
    in_features = model.classifier[1].in_features
    model.classifier[1] = Linear(in_features, 3)
    xavier_uniform_(model.classifier[1].weight)
    print(f'EfficientNet V2 Large: {preprocess}')
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


def get_ShuffleNet_V2_X2_0():
    # https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py#L193
    weights = ShuffleNet_V2_X2_0_Weights.DEFAULT
    preprocess = weights.transforms()
    model = shufflenet_v2_x2_0(weights=weights, num_classes=3)
    print(f'ShuffleNet V2 2: {preprocess}')
    return model, preprocess


def get_Swin_V2_T():
    # https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py
    weights = Swin_V2_T_Weights.DEFAULT
    preprocess = weights.transforms()
    model = swin_v2_t(weights=weights)
    # modify the head
    model.head = Linear(model.head.in_features, 3)
    print(f'Swin V2 Tiny: {preprocess}')
    return model, preprocess


def get_Swin_V2_S():
    weights = Swin_V2_S_Weights.DEFAULT
    preprocess = weights.transforms()
    model = swin_v2_s(weights=weights)
    # modify the head
    model.head = Linear(model.head.in_features, 3)
    print(f'Swin V2 Tiny: {preprocess}')
    return model, preprocess


def get_Swin_V2_B():
    weights = Swin_V2_B_Weights.DEFAULT
    preprocess = weights.transforms()
    model = swin_v2_b(weights=weights)
    # modify the head
    model.head = Linear(model.head.in_features, 3)
    print(f'Swin V2 Tiny: {preprocess}')
    return model, preprocess

