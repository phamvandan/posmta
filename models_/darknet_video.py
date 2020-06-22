from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import config as cf


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    number_box = 0
    for detection in detections:
        if "person" in detection[0].decode():
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
            # cv2.putText(img,
            #             detection[0].decode() +
            #             " [" + str(round(detection[1] * 100, 2)) + "]",
            #             (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             [0, 255, 0], 2)
            number_box = number_box + 1
    return img,number_box


cf.netMain = None
cf.metaMain = None
cf.altNames = None

def load_model():
    # global cf.metaMain, cf.netMain, cf.altNames
    configPath = "./yolo/yolov3-tiny.cfg"
    weightPath = "./yolo/yolov3-tiny.weights"
    metaPath = "./yolo/sign.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if cf.netMain is None:
        cf.netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if cf.metaMain is None:
        cf.metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if cf.altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            cf.altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(cf.netMain),darknet.network_height(cf.netMain),3)
    return darknet_image

def sign_detect(image,darknet_image):
    thresh = 0.25
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ## 416 416
    frame_resized = cv2.resize(frame_rgb,
                            (darknet.network_width(cf.netMain),
                                darknet.network_height(cf.netMain)),
                            interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
    ## name tag, confidence, box
    detections = darknet.detect_image(cf.netMain, cf.metaMain, darknet_image, thresh=thresh)
    # image,box_count = cvDrawBoxes(detections, frame_resized)
    boxes = []
    print('detections', detections)
    for detection in detections:
        # if "person" in detection[0].decode():
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            boxes.append((int(xmin),int(ymin),int(xmax),int(ymax)))
            print('boxes', boxes)
    return boxes

if __name__ == "__main__":
        frame_read = cv2.imread('../output/detect_sign_area.png')

        # frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        # frame_resized = cv2.resize(frame_rgb,
        #                            (darknet.network_width(netMain),
        #                             darknet.network_height(netMain)),
        #                            interpolation=cv2.INTER_LINEAR)

        # darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        # detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        # image = cvDrawBoxes(detections, frame_resized)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(1/(time.time()-prev_time))
        # cv2.imshow('Demo', image)

        darknet_image = load_model()
        boxes = sign_detect(frame_read, darknet_image)
        cv2.imshow('frame_read', frame_read)

        cv2.waitKey(0)

