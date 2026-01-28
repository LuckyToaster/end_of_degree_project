import torch

# Example: Loading a YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

print('model loaded')
