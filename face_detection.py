import argparse
import cv2
import mediapipe as mp

def draw_face_bbox(img, bbox_list):
    for bbox in bbox_list:
        if bbox is None:
            continue
        x0, y0, w, h = bbox
        cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), (0, 255, 0), thickness=2)
    return img

def detect_faces(detector, image):
    height, width, _ = image.shape
    with detector.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections is None:
            return [None]
        bbox_list = [] 
        for detection in results.detections:
            rbbox = detection.location_data.relative_bounding_box
            bbox = (rbbox.xmin * width, rbbox.ymin * height, rbbox.width * width, rbbox.height * height)
            bbox_list.append(bbox)
    return bbox_list

def face_detection_test(input_filename, output_filename):
    mp_face_detection = mp.solutions.face_detection
    image = cv2.imread(input_filename)
    bbox_list = detect_faces(mp_face_detection, image)
    annotated_image = image.copy()
    draw_face_bbox(annotated_image, bbox_list)
    cv2.imwrite(output_filename, annotated_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', type=str)
    parser.add_argument('output_image', type=str)
    arg = parser.parse_args()
    face_detection_test(arg.input_image, arg.output_image)
