from torchvision.models import shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights

'''
torchvision MobileNet source code => https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py#L193
'''

def get_model():
    weights = ShuffleNet_V2_X2_0_Weights.DEFAULT
    preprocess = weights.transforms()
    model = shufflenet_v2_x2_0(weights=weights, num_classes=3)
    print(f'ShuffleNet V2 2: {preprocess}')
    return model, preprocess
