#!/usr/bin/env python
# coding: utf-8

# ## Process for taking an image and getting outputs from the model

# In[ ]:





# In[1]:


#imports
import cv2
import numpy as np
import torch
from PIL import ImageDraw, Image
from facenet_pytorch import MTCNN
from torchvision import transforms

import torchvision.models as models
import torch.nn as nn


# In[2]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# ### this part is preliminary: using it to simulate a face coming from MTCNN and scaled to fit 224x224

# In[3]:




def load_image(img_path, output_image_size, face_detector):
    image_raw = cv2.imread(img_path)

    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

    boxes, confidence = face_detector.detect(image_raw)
    
    if boxes is None:
        return None
    bounding_box = boxes[0]
    
    #collect image height and width
    height, width = image.shape[:2]

    #set scaling factor
    scale=0.98

    #collect coordinates of bounding box
    x1 = bounding_box[0] #dib face.left
    y1 = bounding_box[1] #dib face.top
    x2 = bounding_box[2] #dib face.right
    y2 = bounding_box[3] #dib face.bottom

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


# In[4]:


#set up MTCNN

imagefactor = 1.05
minsize = 35  # minimum size of face
threshold = [0.7, 0.8, 0.8]  # three steps's threshold

mtcnn = MTCNN(keep_all=True, device=device,min_face_size = minsize,thresholds= threshold)


# ### This part takes in a model checkpoint file and an already processed image (224x224, face pulled out by MTCNN)

# In[8]:



def load_model(checkpoint_path):
    '''
    function to load the baselie model 
    simmilar to the training process
    sets the model into eval mode as well and
    populates the trained weights from the checkpoint file
    '''
    dropout=0.5
    model = models.resnet18(pretrained=True)
    
    in_features = model.fc.in_features
#     print(f'Input feature dimensions: {in_features}')

    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, in_features // 2),
        nn.ReLU(),
        nn.BatchNorm1d(in_features // 2),
        nn.Dropout(dropout),
        nn.Linear(in_features // 2, 2)
    )
    
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    return model

def gettransforms():
    '''
    returns the transforms the normalize the input image
    '''
    pre_trained_mean, pre_trained_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    com_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std)
    ])
    return  com_transforms



def process_image_through_model(input_image_of_face, model, composed_transforms):
    '''
    function takes in a face already sized to 224x224 and passes 
    it to the composed transforms then sends the transformed image to
    the model which gives an output, which is then maxed and returned as
    the predicted class (1 for Mask, 0 for No-Mask) output is a tensor
    '''
    normalized_image = composed_transforms(input_image_of_face)
    normalized_image_plus_batch_dim = normalized_image.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(normalized_image_plus_batch_dim)
    _, predicted_class = torch.max(outputs.data, 1)
    
    print("classes: Mask=1, No_Mask=0")
    
    print("predicted class: ", predicted_class.numpy()[0])
    
    return predicted_class
    


# In[ ]:





# In[ ]:





# In[9]:


#get the checkpoint path
checkpoint_path = r"C:/Users/OI/Desktop/data/GWU/GWU_2020_FALL_CSCI6011_PROJECT/MaskLock\ML-dev/checkpoints/2020-10-18-15_05_52--best_model.pth"

#load the model
model = load_model(checkpoint_path)

#get the dummy image
test_img_path = "C:/Users/OI/Desktop/test_1.png"
resized_image_of_face = load_image(test_img_path, 224, mtcnn)

#compose the transforms
transform = gettransforms()

#get output from the model
process_image_through_model(resized_image_of_face, model, transform)


# In[ ]:




