# This file was used to change the format of the data from the original format to the yolov8n format

import os
import shutil

def convert_to_yolo_format(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            img_width = 1920
            img_height = 1080
            
            for line in lines:
                if line.startswith('corners:'):
                    corners = line.split(':')[1].strip()
                    points = corners.split()
                    xs = [int(point.split(',')[0]) for point in points]
                    ys = [int(point.split(',')[1]) for point in points]
                    
                    xmin = min(xs)
                    xmax = max(xs)
                    ymin = min(ys)
                    ymax = max(ys)
                    
                    x_center = (xmin + xmax) / 2.0
                    y_center = (ymin + ymax) / 2.0
                    bbox_width = xmax - xmin
                    bbox_height = ymax - ymin
                    
                    # Normalize coordinates
                    x_center /= img_width
                    y_center /= img_height
                    bbox_width /= img_width
                    bbox_height /= img_height
                    
                    yolo_format = f'0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n'
                    
                    output_file_path = os.path.join(output_folder, file_name)
                    
                    with open(output_file_path, 'w') as output_file:
                        output_file.write(yolo_format)

def copy_images(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".png"):
            output_file_path = os.path.join(output_folder, file_name)
            shutil.copy(file_path, output_file_path)

train_folder = '/home/kbquan/dev/BU/CS585Final/UFPR-ALPR/UFPR-ALPR dataset/training/'
test_folder = '/home/kbquan/dev/BU/CS585Final/UFPR-ALPR/UFPR-ALPR dataset/testing/'
validation_folder = '/home/kbquan/dev/BU/CS585Final/UFPR-ALPR/UFPR-ALPR dataset/validation/'


outputlabel_folder_train = '/home/kbquan/dev/BU/CS585Final/data/labels/train'
outputlabel_folder_test = '/home/kbquan/dev/BU/CS585Final/data/labels/test'
outputlabel_folder_validation = '/home/kbquan/dev/BU/CS585Final/data/labels/val'

outputimage_folder_train = '/home/kbquan/dev/BU/CS585Final/data/images/train'
outputimage_folder_test = '/home/kbquan/dev/BU/CS585Final/data/images/test'
outputimage_folder_validation = '/home/kbquan/dev/BU/CS585Final/data/images/val'

for dirpath, dirnames, filenames in os.walk(train_folder):
    if not dirnames:
        input_folder = dirpath
        copy_images(input_folder, outputimage_folder_train)
        convert_to_yolo_format(input_folder, outputlabel_folder_train)
        

for dirpath, dirnames, filenames in os.walk(test_folder):
    if not dirnames:
        input_folder = dirpath
        copy_images(input_folder, outputimage_folder_test)
        convert_to_yolo_format(input_folder, outputlabel_folder_test)
        

for dirpath, dirnames, filenames in os.walk(validation_folder):
    if not dirnames:
        input_folder = dirpath
        copy_images(input_folder, outputimage_folder_validation)
        convert_to_yolo_format(input_folder, outputlabel_folder_validation)


