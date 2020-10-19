import cv2
import torch
import numpy as np
from pathlib import Path



def load_image_and_preprocess(image_filename, output_image_size, face_detector, face_detector_type="frontal"):
    # read in image with cv2

    image_raw = cv2.imread(image_filename)

    #this might not be needed to correct for color space
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    

    # #use first face detected and check if there were none
    result = face_detector.detect(image_raw)
   ## cv2.imshow(image_raw)
    # print(str(image_filename) +" >> " + str(type(result)) + ":" + str(result))

    # if len(result) == 0 or len(result) > 1:
    #     # print("TOO MANY FACES!!! ..or not enough!? :O")
    #     # sys.exit(0)
    #     return None

    # bounding_box = result[0]["box"]
    bounding_box = result[0][0]
    #collect image height and width
    height, width = image.shape[:2]

    #set scaling factor
    scale=0.98

    #collect coordinates of bounding box
    x1 = bounding_box[0] #dib face.left
    y1 = bounding_box[1] #dib face.top
    x2 = bounding_box[0]+bounding_box[2] #dib face.right
    y2 = bounding_box[1]+bounding_box[3] #dib face.bottom

    #scale bounding box?
    size_bb = int(max(x2 - x1, y2 - y1) * scale)

    #control for out of bounds, x-y top left corner
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)

    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    # set up for crop with slicing
    cropped_face = image[y1:y1 + size_bb, x1:x1 + size_bb]

    resized_image = cv2.resize(cropped_face, (output_image_size, output_image_size))
    return resized_image










# def get_boundingbox(face, width, height, scale=1.3, minsize=None):
#     x1 = face.left()
#     y1 = face.top()
#     x2 = face.right()
#     y2 = face.bottom()
#     size_bb = int(max(x2 - x1, y2 - y1) * scale)
#     if minsize:
#         if size_bb < minsize:
#             size_bb = minsize
#     center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

#     # Check for out of bounds, x-y top left corner
#     x1 = max(int(center_x - size_bb // 2), 0)
#     y1 = max(int(center_y - size_bb // 2), 0)
#     # Check for too big bb size for given x, y
#     size_bb = min(width - x1, size_bb)
#     size_bb = min(height - y1, size_bb)

#     return x1, y1, size_bb

# def get_face_crop(face_detector, image, face_detector_type):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     faces = face_detector(gray, 1)
    
#     height, width = image.shape[:2]
    
#     if len(faces) == 0:
#         return None
#     else:
#         face = faces[0]
#         if face_detector_type == "cnn":
#             x, y, size = get_boundingbox_cnn(face, width, height)
#         else:
#             x, y, size = get_boundingbox(face, width, height)

#         cropped_face = image[y:y + size, x:x + size]
#         return cropped_face
    
