import ast
import cv2
import numpy as np
import pandas as pd


def process_video(video_path, output_path, results_path):
    """
    Processes a video to detect and display license plates.

    Args:
        video_path: The path to the input video.
        output_path: The path to the output video.
        results_path: The path to the CSV file containing detection results.
    """

    # Load results
    results = pd.read_csv(results_path)

    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Extract license plate information
    license_plates = dict()
    for vehicle_id in np.unique(results['vehicle_id']):
        max_score = np.amax(results[results['vehicle_id'] == vehicle_id]['license_number_score'])
        license_plates[vehicle_id] = {
            'license_plate_image': None,
            'license_plate_number': results[(results['vehicle_id'] == vehicle_id) &
                                            (results['license_number_score'] == max_score)]['license_number'].iloc[0]
        }

        # Extract license plate crop
        frame_number = results[(results['vehicle_id'] == vehicle_id) &
                               (results['license_number_score'] == max_score)]['frame_number'].iloc[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        lp_x1, lp_y1, lp_x2, lp_y2 = ast.literal_eval(results[(results['vehicle_id'] == vehicle_id) &
                                                  (results['license_number_score'] == max_score)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        license_plates[vehicle_id]['license_plate_image'] = frame[int(lp_y1):int(lp_y2), int(lp_x1):int(lp_x2), :]
        license_plates[vehicle_id]['license_plate_image'] = cv2.resize(license_plates[vehicle_id]['license_plate_image'], (int((lp_x2 - lp_x1) * 400 / (lp_y2 - lp_y1)), 400))

    # Process frames
    frame_number = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        frame_number += 1

        if not ret:
            break

        df = results[results['frame_number'] == frame_number]

        for index, row in df.iterrows():
            # Draw vehicle rectangle
            vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = ast.literal_eval(row['vehicle_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(vehicle_x1), int(vehicle_y1)), (int(vehicle_x2), int(vehicle_y2)), (0, 255, 0), 12)

            # Draw license plate rectangle
            lp_x1, lp_y1, lp_x2, lp_y2 = ast.literal_eval(row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(lp_x1), int(lp_y1)), (int(lp_x2), int(lp_y2)), (0, 255, 255), 8)

            # Display license plate number
            license_crop = license_plates[row['vehicle_id']]['license_plate_image']
            height, width, _ = license_crop.shape

            text = license_plates[row['vehicle_id']]['license_plate_number']
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 10)
            text_x = int(lp_x1 + (lp_x2 - lp_x1 - text_size[0]) / 2)  # Center horizontally
            text_y = int(lp_y1 - 30)  # Adjusted to be above the license plate rectangle
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 10)

        video_writer.write(frame)

    video_writer.release()
    cap.release()


if __name__ == '__main__':
    video_path = 'sample_1.mp4'
    output_path = 'sample_1_output.mp4'
    results_path = 'results_interpolated.csv'
    process_video(video_path, output_path, results_path)