import cv2
from sort import *
from ultralytics import YOLO
from helper import identify_vehicle, get_license_plate_text, save_results


# Initialize SORT tracker
tracker = Sort()

# Load YOLO models
vehicle_detector = YOLO('vehicle_detector.pt')
license_plate_detector = YOLO('license_plate_detector.pt')
# license_plate_detector = YOLO('best.pt')
# Use best.pt for model trained on UFPR-ALPR dataset

# Define class IDs for vehicles
vehicle_class_ids = [2, 3, 5, 7]

# Open video file
cap = cv2.VideoCapture('./sample_1.mp4')

# Initialize dictionary to store results
results = dict()

# Process video frames
frame_number = -1
while True:
    frame_number += 1
    ret, frame = cap.read()
    if not ret:
        break

    # Detect vehicles
    vehicle_detections = vehicle_detector(frame)[0]
    vehicle_bboxes = []
    for box in vehicle_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        if int(class_id) in vehicle_class_ids:
            vehicle_bboxes.append([x1, y1, x2, y2, score])

    # Track vehicles
    tracked_ids = tracker.update(np.asarray(vehicle_bboxes))

    # Detect license plates
    license_plate_detections = license_plate_detector(frame)[0]

    results[frame_number] = dict()
    # Process license plates
    for bbox in license_plate_detections.boxes.data.tolist():
        x1_lp, y1_lp, x2_lp, y2_lp, score_lp, class_id = bbox

        # Get the associated vehicle for the license plate
        vehicle_bbox, vehicle_id = identify_vehicle(bbox, tracked_ids)

        if vehicle_id != -1:
            # Crop and process license plate
            cropped_license_plate_image = frame[int(y1_lp):int(y2_lp), int(x1_lp): int(x2_lp), :]

            cropped_license_plate_image = cv2.cvtColor(cropped_license_plate_image, cv2.COLOR_BGR2GRAY)
            _, cropped_license_plate_image = cv2.threshold(cropped_license_plate_image, 64, 127, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

            license_plate_text, text_score = get_license_plate_text(cropped_license_plate_image)

            if license_plate_text is not None:
                results[frame_number][vehicle_id] = {
                    'vehicle': {'bbox': vehicle_bbox},
                    'license_plate': {'bbox': bbox, 'text': license_plate_text,
                                      'bbox_score': score_lp, 'text_score': text_score}
                }

# write results
save_results(results, './results.csv')