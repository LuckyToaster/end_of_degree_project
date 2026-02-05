from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.nn.init import xavier_uniform_
from torch.nn import Linear 


def get_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # adapt the head for regression
    in_features = model.classifier[1].in_features
    model.classifier[1] = Linear(in_features, 3)
    xavier_uniform_(model.classifier[1].weight)

    return model, weights.transforms()
