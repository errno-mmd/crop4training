from human_detection import detect_people, get_detectron2_predictor, draw_people_bbox
from face_detection import detect_faces, draw_face_bbox
import argparse
import cv2
import mediapipe as mp
import pathlib
import re
from mimetypes import guess_type

def locate_crop_box(people_bbox, face_bbox, width, height):
    px, py, pw, ph = people_bbox
    if face_bbox is None:
        center_x = px + pw / 2
        center_y = py + ph / 2
    else:
        fx, fy, fw, fh = face_bbox
        center_x = fx + fw / 2
        center_y = fy + fh / 2
    left = int(center_x - width / 2)
    top = int(center_y - height / 2)

    if left < px:
        if width < pw:
            left = px
        elif left + width < px + pw:
            left = px + pw - width
    elif left + width > px + pw:
        if width < pw:
            left = px + pw - width
        elif left > px:
            left = px

    if top < py:
        if height < ph:
            top = py
        elif top + height < py + ph:
            top = py + ph - height
    elif top + height > py + ph:
        if height < ph:
            top = py + ph - height
        elif top > py:
            top = py

    return left, top

def calc_resize_ratio(width, height, pw, ph, src_w, src_h):
    if ph / height > pw / width:
        if src_w * height / ph < width:
            return width / src_w
        else:
            return height / ph
    else:
        if src_h * width / pw < height:
            return height / src_h
        else:
            return width / pw

def resize_image(image, ratio, width, height):
    image_height, image_width, _ = image.shape
    w = int(image_width * ratio)
    h = int(image_height * ratio)
    if w < width:
        w = width
    if h < height:
        h = height
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image

def resize_bbox(bbox, ratio):
    if bbox is None:
        return None
    x, y, w, h = bbox
    return int(x * ratio), int(y * ratio), int(w * ratio), int(h * ratio)

def crop_image(image, left, top, width, height):
    original_height, original_width, _ = image.shape
    left = int(left)
    top = int(top)
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if left + width > original_width:
        left = original_width - width
    if top + height > original_height:
        top = original_height - height
    cropped_image = image[top : top + height, left : left + width]
    return cropped_image

def bbox_add_mergin(bbox, margin, image):
    x, y, w, h = bbox
    image_height, image_width, _ = image.shape
    w2 = w * (1.0 + margin)
    h2 = h * (1.0 + margin)
    x2 = x - (w2 - w) / 2
    y2 = y - (h2 - h) / 2
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0
    if x2 + w2 > image_width:
        w2 = image_width - x2
    if y2 + h2 > image_height:
        h2 = image_height - y2
    return x2, y2, w2, h2

def resize_and_crop_image(image, people_bbox, face_bbox, width, height):
    people_bbox_margin = 0.05
    people_bbox = bbox_add_mergin(people_bbox, people_bbox_margin, image)
    px, py, pw, ph = people_bbox
    image_height, image_width, _ = image.shape

    if pw > width or ph > height:
        ratio = calc_resize_ratio(width, height , pw, ph, image_width, image_height)
        image = resize_image(image, ratio, width, height)
        people_bbox = resize_bbox(people_bbox, ratio)
        face_bbox = resize_bbox(face_bbox, ratio)

    left, top = locate_crop_box(people_bbox, face_bbox, width, height)
    return crop_image(image, left, top, width, height)

def make_dummy_bbox(image):
    image_height, image_width, _ = image.shape
    x = image_width / 2 - 1
    y = image_height / 2 - 1
    w = 2
    h = 2
    return x, y, w, h

def enlarge_bbox(bbox, image, width, height):
    image_height, image_width, _ = image.shape
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    if (width / height > image_width / image_height):
        w2 = image_width
        h2 = image_width * height / width
    else:
        w2 = image_height * width / height
        h2 = image_height
    x2 = center_x - w2 / 2
    y2 = center_y - h2 / 2
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0
    if x2 + w2 > image_width:
        x2 = image_width - w2
    if y2 + h2 > image_height:
        y2 = image_height - h2
    return x2, y2, w2, h2

def save_detection_result(file_path, image, people_bbox, face_bbox):
    result_image = image.copy()
    result_image = draw_people_bbox(result_image, [people_bbox])
    result_image = draw_face_bbox(result_image, [face_bbox])
    cv2.imwrite(str(file_path), result_image)

def is_image_file(filename):
    (file_type, tmp) = guess_type(filename)
    if (re.match(r'image', file_type)):
        return True
    return False

def batch_crop_images(input_dir, output_dir, width, height, no_focus, detection_save_dir):
    input_dir_path = pathlib.Path(input_dir).resolve()
    output_dir_path = pathlib.Path(output_dir).resolve()
    mp_face_detection = mp.solutions.face_detection
    dt2 = get_detectron2_predictor()
    if detection_save_dir:
        detection_dir_path = pathlib.Path(detection_save_dir).resolve()
        detection_dir_path.mkdir(parents=True, exist_ok=True)

    for input_file_path in input_dir_path.glob("*"):
        if not is_image_file(input_file_path):
            continue
        print(input_file_path)
        image = cv2.imread(str(input_file_path))
        people_bbox_list = detect_people(dt2, image)
        if len(people_bbox_list) == 0:
            print("people not detected")
            if no_focus:
                dummy_bbox = make_dummy_bbox(image)
                people_bbox_list = [dummy_bbox]
            else:
                continue
        face_bbox_list = detect_faces(mp_face_detection, image)

        # first person/face only
        people_bbox = people_bbox_list[0]
        face_bbox = face_bbox_list[0]

        if detection_save_dir:
            detection_file_path = detection_dir_path / f"{input_file_path.stem}.png"
            save_detection_result(detection_file_path, image, people_bbox, face_bbox)

        if no_focus:
            people_bbox = enlarge_bbox(people_bbox, image, width, height)

        result_image = resize_and_crop_image(image, people_bbox, face_bbox, width, height)
        output_file_path = output_dir_path / f"{input_file_path.stem}.png"
        cv2.imwrite(str(output_file_path), result_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--no_focus', action='store_true', default=False)
    parser.add_argument('--detection_save_dir', type=str, default=None)
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    arg = parser.parse_args()
    batch_crop_images(arg.input_dir, arg.output_dir, arg.width, arg.height, arg.no_focus, arg.detection_save_dir)
