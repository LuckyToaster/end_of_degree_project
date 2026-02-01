from torvision.models import efficientnet_b0, EfficientNet_B0_Weights.
 
weights = EfficientNet_B0_Weights.DEFAULT
preprocess = weights.transforms() # img_transformed = preprocess(img)
model = efficientnet_b0(weights=weights)
