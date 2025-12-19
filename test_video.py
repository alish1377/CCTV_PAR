import argparse
import json
import os
import pickle
import time

from ultralytics import YOLO

from dataset.augmentation import get_transform
from metrics.pedestrian_metrics import get_pedestrian_metrics
from models.model_factory import build_loss, build_backbone, build_classifier

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import shutil

from configs import cfg, update_config
from dataset.pedes_attr.pedes import PedesAttr, PedesAttrUPAR, PedesAttrUPARInfer, PedesAttrUPARInferTestPhase
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from models.base_block import FeatClassifier

from tools.function import get_model_log_path, get_reload_weight, seperate_weight_decay
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, str2bool, gen_code_archive
from models.backbone import swin_transformer2
from losses import bceloss, scaledbceloss
import matplotlib.pyplot as plt
import pickle
import cv2
from ultralytics import YOLO
from IPython.display import display, Image, clear_output
import time
import torchvision.transforms as T
from PIL import Image

import matplotlib.image as mpimg


set_seed(605)


# This function gets the frame of video and returns 
# the cropped of detected person in frame 
def generate_box(frame, model, conf_threshold=0.25):
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, classes=0, tracker="bytetrack.yaml", persist=True, conf=conf_threshold)
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    print("annotated_frame\n", annotated_frame.shape)
    # Get the box of detetced person
    boxes = results[0].boxes
    ids = boxes.id
    confs = boxes.conf
    box = boxes.xyxy.tolist()
    is_box = False
    pose_list, cropped_image_list = [], []
    if(len(box)!=0):
        is_box = True
        for b in range(len(box)):
            x0 = int(box[b][0])
            y0 = int(box[b][1])
            x1 = int(box[b][2])
            y1 = int(box[b][3])
            # Crop the image to classify attributes in the next section
            cropped_image_list.append(frame[y0:y1, x0:x1])
            pose_list.append((x0-100,y0-25))

    return annotated_frame, cropped_image_list, is_box, pose_list, ids, confs

# This function get the box of frame(person) and return attributes 
# of that image
def predict(main_img, model, attr_names):
    rgb_image = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(rgb_image)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        normalize
    ])
    transform_img = test_transform(PIL_image)
    transform_img = transform_img[None, :, :, :]
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        transform_img = transform_img.to(device)
        
        gt_label = None
        valid_logits, attns = model(transform_img, gt_label)
        valid_probs = torch.sigmoid(valid_logits[0])

        attr_dict = {key: value for key, value in zip(attr_names, valid_probs[0])}
        print("mapped attribute with index..")
        print(attr_dict)
        attr_list = select_attributes(attr_dict, thresh=0.5)

        return attr_dict, attr_list

# Define some types of function to select attributes
# Binary attribute
def binary_attribute(attr_dict, __attr_lists__, attr1, attr2, thresh=0.5):
    if attr_dict[attr1] > thresh:
        __attr_lists__.append(attr1)
    else:
        __attr_lists__.append(attr2)

    return __attr_lists__


def nothing_or_some_attributes(attr_dict, __attr_lists__, attrs, thresh=0.5):
    temp_attr = {i : attr_dict[i] for i
                 in attrs}
    # If "type attribute" "nothing" value is the most, append this value
    # else append the maximum "type attribute" score
    if temp_attr[attrs[0]] == max(temp_attr.values()):
        __attr_lists__.append(attrs[0])
    else:
        for key in list(temp_attr.keys())[1:]:
            if temp_attr[key] > thresh:
                __attr_lists__.append(key)
    return __attr_lists__


def one_between_some_attributes(attr_dict, __attr_lists__, attrs, thresh=0.5):
    temp_attr = {i : attr_dict[i] for i
                 in attrs}

    __attr_lists__.append(max(temp_attr, key=lambda k: temp_attr[k]))
    return __attr_lists__


# This function is used to separate more conceptual attribute
def select_attributes(attr_dict, thresh=0.5):
    __attr_lists__ = []
    
    # Hat
    __attr_lists__ = binary_attribute(attr_dict, __attr_lists__, 
                                      "Hat", "Not_Hat")
    
    # Glasses
    __attr_lists__ = binary_attribute(attr_dict, __attr_lists__, 
                                      "Glasses", "Not-Glasses")

    # SunGlasses
    __attr_lists__ = binary_attribute(attr_dict, __attr_lists__, 
                                      "Sunglasses", "Not-Sunglasses")

    # Upper body color
    attrs = ["U_black", "U_blue", "U_brown", "U_green", "U_grey", "U_orange", "U_pink", "U_purple", "U_red", "U_white", "U_yellow", "////U_Others////"]
    __attr_lists__ = one_between_some_attributes(attr_dict, __attr_lists__, attrs)

    # Upper body short or long sleeves
    __attr_lists__ = binary_attribute(attr_dict, __attr_lists__, 
                                      "Upper_short", "Not-Upper_short")


    # Lower body color
    attrs = ["L_black", "L_blue", "L_brown", "L_green", "L_grey", "L_orange", "L_pink", "L_purple", "L_red", "L_white", "L_yellow", "////L_Others////"]
    __attr_lists__ = one_between_some_attributes(attr_dict, __attr_lists__, attrs)


    # Bag
    __attr_lists__ = binary_attribute(attr_dict, __attr_lists__, 
                                      "Bag", "Not-Bag")

    # Backpack
    __attr_lists__ = binary_attribute(attr_dict, __attr_lists__, 
                                      "Backpack", "Not-Backpack")
    
    # Age
    attrs = {"Age_young": attr_dict['Age_young'],
             "Age_adult": attr_dict['Age_adult'],
             "Age_old": attr_dict['Age_old']}
    
    __attr_lists__.append(max(attrs, key=lambda k: attrs[k]))

    # Gender
    __attr_lists__ = binary_attribute(attr_dict, __attr_lists__, 
                                      "Gender_female", "Gender_male")
    
    # Hair
    attrs = {"Hair_short": attr_dict['Hair_short'],
             "Hair_long": attr_dict['Hair_long'],
             "Hair_bald": attr_dict['Hair_bald']}
    __attr_lists__.append(max(attrs, key=lambda k: attrs[k]))
    
    return __attr_lists__


# The function to adjust dimensions and new frame with a suitable size
def set_dimensions(old_frame):
    w, h = old_frame.shape[0], old_frame.shape[1]
    scale = h / w
    new_frame = None
    if h > w:
        new_frame = cv2.resize(old_frame, ((int(scale*540)), 540))
    else:
        new_frame = cv2.resize(old_frame, (540, int((1/scale)*540)))

    return new_frame

def main(cfg):
    # Reload the model
    attr_names = cfg.DATASET.ATTRS
    print(cfg)
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, _ = get_model_log_path(exp_dir, cfg.TEST.MODEL_FOLDER_NAME)
    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)
    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=len(attr_names),
        c_in=2*c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale =cfg.CLASSIFIER.SCALE
    )
    model = FeatClassifier(backbone, classifier)

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(DEVICE)
    print("Device: ", DEVICE)

    model_data = None
    if cfg.DATASET.ZERO_SHOT:
        model_data = f"{cfg.DATASET.NAME}ZS_{cfg.BACKBONE.TYPE}"
    else:
        model_data = f"{cfg.DATASET.NAME}_{cfg.BACKBONE.TYPE}"
    attr_recognition_model = get_reload_weight(
        model_dir, model, pth=cfg.TEST.PRETRAINED_WEIGHTS)

    # Open the video file
    video_path = f"{cfg.TEST.VIDEO_PATH}/{cfg.TEST.VIDEO_NAME}.mp4"

    cap = cv2.VideoCapture(video_path)


    if cap.isOpened():
        print("VideoCapture is open")
        # Get the fps of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        raise IOError("Cannot open video file")

    # Read first frame to set the dimensions
    # Save video file
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    _, frame = cap.read()
    frame = set_dimensions(frame)
    
    os.makedirs(cfg.TEST.OUTPUT_VIDEO_PATH, exist_ok=True)
    out = cv2.VideoWriter(f'{cfg.TEST.OUTPUT_VIDEO_PATH}/{model_data}_{cfg.TEST.VIDEO_NAME}.avi', 
                             fourcc, fps, (frame.shape[1], frame.shape[0]), isColor=True)
    # Load the YOLO model
    yolo_model = YOLO(cfg.TEST.YOLO_MODEL)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Flag for checking each 10 frames 
    frame_flag_check = 0
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            frame = set_dimensions(frame)
            # gets frame and return the detected image and the box of person
            annotated_frame, cropped_img_list, is_box, pose_list, ids, confs= generate_box(frame, yolo_model, conf_threshold=cfg.TEST.YOLO_CONF)
            # gets frame and return the attributes of the frame
            attributes = 0
            # frame_flag_check identifies the model execution step
            if is_box and (frame_flag_check % cfg.TEST.CHECK_PER_FRAME) == 0: 
                all_attributes, all_attr_list = [], []
                for f in range(len(cropped_img_list)):
                    attributes, attr_list= predict(cropped_img_list[f], 
                                                   attr_recognition_model, 
                                                   attr_names)
                    all_attributes.append(attributes)
                    all_attr_list.append(attr_list)

            frame_flag_check += 1
            # Save the image temporarily
            if is_box:
                idx = min(len(pose_list), len(all_attr_list))
                for k in range(idx):               
                    x0, y0 = pose_list[k][0], pose_list[k][1]
                    x_copy = x0
                    # In this test, just get outputs from women
                    for i, attr in enumerate(all_attr_list[k]):
                        cv2.putText(annotated_frame, "{ " + attr + " }", (x0, y0), 
                                    font, 0.4, (54, 255, 240), 1, cv2.LINE_AA)
                        x0+=200
                        if (i + 1) % 3 == 0:
                            y0 -= 18
                            x0 = x_copy
            annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], 
                                                           frame.shape[0]))

            out.write(annotated_frame)
            cv2.imshow("annotated image", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
            
    out.release()
    cap.release()
    cv2.destroyAllWindows()

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
    )
    parser.add_argument("--debug", type=str2bool, default="true")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)
    main(cfg)
