from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch.nn.init import xavier_uniform_
from torch.nn import Linear 

'''
torchvision MobileNet source code => https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py#L191

self.classifier = nn.Sequential(
    nn.Linear(lastconv_output_channels, last_channel),
    nn.Hardswish(inplace=True),
    nn.Dropout(p=dropout, inplace=True),
    nn.Linear(last_channel, num_classes),
)
'''

def get_model():
    weights = MobileNet_V3_Large_Weights.DEFAULT
    preprocess = weights.transforms()
    model = mobilenet_v3_large(weights=weights)

    # adapt the head for regression
    last_channel = model.classifier[3].in_features
    model.classifier[3] = Linear(last_channel, 3)

    print(f'MobileNet V3 Large: {preprocess}')
    return model, preprocess
    #xavier_uniform_(model.classifier[3].weight)
