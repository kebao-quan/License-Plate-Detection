from ultralytics import YOLO


# Transfer learning from COCO
model = YOLO('yolov8n.pt')
results = model.train(data="alpr.yaml", epochs=20)