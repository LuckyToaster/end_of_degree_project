from ultralytics import FastSAM
import cv2

# Define an inference source

# Create a FastSAM model
model = FastSAM("FastSAM-x.pt")  # or FastSAM-x.pt

# Run inference on an image
source = "/home/lucky/Pictures/wallpaper/food/breakfast_fruits.jpg"
results = model(
    source, 
    device='cpu', 
    retina_masks=True, 
    imgsz=1024, 
    conf=0.8, # 0.4
    iou=0.3 # 0.9
)

results[0].show()

import matplotlib.pyplot as plt

plot_img = results[0].plot(masks=True, boxes=False, labels=False)
plot_rgb = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(plot_rgb)
plt.axis('off')
plt.tight_layout()
plt.show() # This window is fully resizable and has a functional UI

# display using open cv
# plot_img = results[0].plot(masks=True, boxes=False, labels=False)
# cv2.imshow("FastSAM Segments", plot_img)
#
# key = cv2.waitKey(0) & 0xFF
# if key == ord('q') or key == 27:
#     cv2.destroyAllWindows()
