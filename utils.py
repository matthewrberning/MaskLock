import cv2
import torch
import numpy as np
from pathlib import Path

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def load_image_and_preprocess(image_filename, output_image_size, face_detector, face_detector_type="frontal"):
    image = cv2.imread(image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    cropped_image = get_face_crop(face_detector, image, face_detector_type)
    if cropped_image is None:
        return None
    
    resized_image = cv2.resize(cropped_image, (output_image_size, output_image_size))
    return resized_image


def get_face_crop(face_detector, image, face_detector_type):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray, 1)
    
    height, width = image.shape[:2]
    
    if len(faces) == 0:
        return None
    else:
        face = faces[0]
        if face_detector_type == "cnn":
            x, y, size = get_boundingbox_cnn(face, width, height)
        else:
            x, y, size = get_boundingbox(face, width, height)

        cropped_face = image[y:y + size, x:x + size]
        return cropped_face
    
