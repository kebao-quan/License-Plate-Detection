# For only detecting Massachusetts license plates, we can use the following code:


import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
int_to_char = {v: k for k, v in char_to_int.items()}


def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_number', 'vehicle_id', 'vehicle_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_number in results.keys():
            for vehicle_id in results[frame_number].keys():
                print(results[frame_number][vehicle_id])
                f.write('{},{},{},{},{},{},{}\n'.format(frame_number,
                                                        vehicle_id,
                                                        '[{} {} {} {}]'.format(
                                                            results[frame_number][vehicle_id]['vehicle']['bbox'][0],
                                                            results[frame_number][vehicle_id]['vehicle']['bbox'][1],
                                                            results[frame_number][vehicle_id]['vehicle']['bbox'][2],
                                                            results[frame_number][vehicle_id]['vehicle']['bbox'][3]),
                                                        '[{} {} {} {}]'.format(
                                                            results[frame_number][vehicle_id]['license_plate']['bbox'][0],
                                                            results[frame_number][vehicle_id]['license_plate']['bbox'][1],
                                                            results[frame_number][vehicle_id]['license_plate']['bbox'][2],
                                                            results[frame_number][vehicle_id]['license_plate']['bbox'][3]),
                                                        results[frame_number][vehicle_id]['license_plate']['bbox_score'],
                                                        results[frame_number][vehicle_id]['license_plate']['text'],
                                                        results[frame_number][vehicle_id]['license_plate']['text_score'])
                        )
        f.close()


def check_format(text):
    if len(text) != 6:
        return False

    conditions = [
        text[0] in string.digits or text[0] in char_to_int.keys(),  # First character: digit
        text[1] in string.ascii_uppercase or text[1] in int_to_char.keys(),  # Second character: letter
        text[2] in string.ascii_uppercase or text[2] in int_to_char.keys(),  # Third character: letter
        text[3] in string.ascii_uppercase or text[3] in int_to_char.keys(),  # Fourth character: letter
        text[4] in string.digits or text[4] in char_to_int.keys(),  # Fifth character: digit
        text[5] in string.digits or text[5] in char_to_int.keys(),  # Sixth character: digit
    ]

    return all(conditions)


def output_formatted_text(text):
    formatted_text = ''
    for i, char in enumerate(text):
        # In Massachusetts license plates, the second, third, and fourth characters are letters
        if i in [1, 2, 3]:
            formatted_text += int_to_char.get(char, text[i])
        else:
            formatted_text += char_to_int.get(char, text[i])

    return formatted_text


def get_license_plate_text(cropped_license_plate_image):
    text_results = reader.readtext(cropped_license_plate_image)

    for bbox, text, score in text_results:
        text = text.upper().replace(' ', '')
        
        if (check_format(text)):
            return output_formatted_text(text), score

    return None, None


def identify_vehicle(license_plate_bbox, vehicle_track_ids):
    x1_lp, y1_lp, x2_lp, y2_lp, _, _ = license_plate_bbox

    for vehicle_bbox in vehicle_track_ids:
        x1_v, y1_v, x2_v, y2_v, vehicle_id = vehicle_bbox

        if x1_lp > x1_v and y1_lp > y1_v and x2_lp < x2_v and y2_lp < y2_v:
            return (x1_v, y1_v, x2_v, y2_v), vehicle_id

    return (-1, -1, -1, -1), -1
