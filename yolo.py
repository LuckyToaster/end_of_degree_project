from ultralytics import YOLO

# Load a COCO-pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# Run inference with the YOLO26n model on the 'bus.jpg' image
results = model("/home/lucky/Pictures/wallpaper/food/breakfast_fruits.jpg")
results[0].show()

