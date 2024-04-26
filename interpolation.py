import csv
import numpy as np
from scipy.interpolate import interp1d


def interpolate_bounding_boxes(result_data):
    frame_numbers = np.array([int(row['frame_number']) for row in result_data])
    vehicle_ids = np.array([int(float(row['vehicle_id'])) for row in result_data])
    vehicle_bboxes = np.array([list(map(float, row['vehicle_bbox'][1:-1].split())) for row in result_data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in result_data])

    results = []
    unique_vehicle_ids = np.unique(vehicle_ids)
    for vehicle_id in unique_vehicle_ids:
        vehicle_frame_numbers = [row['frame_number'] for row in result_data if int(float(row['vehicle_id'])) == int(float(vehicle_id))]

        vehicle_mask = vehicle_ids == vehicle_id
        vehicle_frames = frame_numbers[vehicle_mask]
        interpolated_vehicle_results = []
        interpolated_results = []

        frame_1 = vehicle_frames[0]

        for i, vehicle_bbox in enumerate(vehicle_bboxes[vehicle_mask]):
            frame_number = vehicle_frames[i]
            license_plate_bbox = license_plate_bboxes[vehicle_mask][i]

            if i > 0:
                prev_frame = vehicle_frames[i-1]
                prev_vehicle_bbox = interpolated_vehicle_results[-1]
                prev_license_plate_bbox = interpolated_results[-1]

                if frame_number - prev_frame > 1:
                    frames_gap = frame_number - prev_frame
                    data_1 = np.array([prev_frame, frame_number])
                    data_2 = np.linspace(prev_frame, frame_number, num=frames_gap, endpoint=False)
                    method = interp1d(data_1, np.vstack((prev_vehicle_bbox, vehicle_bbox)), axis=0, kind='linear')
                    interpolated_vehicle_bboxes = method(data_2)
                    method = interp1d(data_1, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = method(data_2)

                    interpolated_vehicle_results.extend(interpolated_vehicle_bboxes[1:])
                    interpolated_results.extend(interpolated_license_plate_bboxes[1:])

            interpolated_vehicle_results.append(vehicle_bbox)
            interpolated_results.append(license_plate_bbox)

        for i in range(len(interpolated_vehicle_results)):
            vehicle_bbox = interpolated_vehicle_results[i]
            license_plate_bbox = interpolated_results[i]
            frame_number = frame_1 + i
            row = {
                'frame_number': str(frame_number),
                'vehicle_id': str(vehicle_id),
                'vehicle_bbox': ' '.join(map(str, vehicle_bbox)),
                'license_plate_bbox': ' '.join(map(str, license_plate_bbox)),
                'license_plate_bbox_score': '0',
                'license_number': '0',
                'license_number_score': '0'
            }

            if str(frame_number) in vehicle_frame_numbers:
                original_data = [row for row in result_data if int(row['frame_number']) == frame_number and int(float(row['vehicle_id'])) == int(float(vehicle_id))][0]
                row.update({
                    'license_plate_bbox_score': original_data.get('license_plate_bbox_score', '0'),
                    'license_number': original_data.get('license_number', '0'),
                    'license_number_score': original_data.get('license_number_score', '0')
                })
            results.append(row)

    return results


def load_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        return list(reader)


def save_data(data, output_path):
    header = ['frame_number', 'vehicle_id', 'vehicle_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
    with open(output_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)


def main():
    input_file_path = 'results.csv'
    output_file_path = 'results_interpolated.csv'

    data = load_data(input_file_path)
    interpolated_data = interpolate_bounding_boxes(data)
    save_data(interpolated_data, output_file_path)


if __name__ == "__main__":
    main()
