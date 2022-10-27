import argparse
import cv2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def draw_people_bbox(img, bbox_list):
    for bbox in bbox_list:
        x0, y0, w, h = bbox
        cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), (0, 0, 255), thickness=2)
    return img

def get_detectron2_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor

def detect_people(predictor, img):
    outputs = predictor(img)
    class_array = outputs["instances"].pred_classes.to('cpu').numpy()
    bbox_array = outputs["instances"].pred_boxes.to('cpu').tensor.numpy()
    people_bbox = []
    for i in range(class_array.shape[0]):
        if class_array[i] == 0:
            left, top, right, bottom = bbox_array[i]
            box = (left, top, right - left, bottom - top)
            people_bbox.append(box)
    return people_bbox

def human_detection_test(input_filename, output_filename):
    dt2 = get_detectron2_predictor()
    image = cv2.imread(input_filename)
    bbox_list = detect_people(dt2, image)
    image = draw_people_bbox(image, bbox_list)
    cv2.imwrite(output_filename, image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', type=str)
    parser.add_argument('output_image', type=str)
    arg = parser.parse_args()
    human_detection_test(arg.input_image, arg.output_image)
