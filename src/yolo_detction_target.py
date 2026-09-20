from ultralytics import YOLO
import torch

model = YOLO("yolo11n.pt")

x = torch.randn(1, 3, 640, 640)

with torch.no_grad():
    y = model.model(x)

print(type(y))

if isinstance(y, (list, tuple)):
    print(len(y))
    for i, item in enumerate(y):
        if hasattr(item, "shape"):
            print(i, item.shape)
        else:
            print(i, type(item))
else:
    print(y.shape)