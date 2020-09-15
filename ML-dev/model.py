import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

filename = "test.png"
print("Test file: ", filename)

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)

#turn image into batch by changing tensor shape
input_batch = input_tensor.unsqueeze(0) 
# print(input_batch.shape())

#send to GPU if possible
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)


#image net scores
print("Output: ",output[0], "\n")

#softmax for proba
print("Probabilites")
print(torch.nn.functional.softmax(output[0], dim=0))

