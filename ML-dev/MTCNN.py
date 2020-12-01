import cv2
import torch
import torchvision.models as models
import torch.nn as nn
from facenet_pytorch import MTCNN
from torchvision import transforms
import socket
import time
import goto


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

    return predicted_class.numpy()[0].astype(int)


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



def load_model(checkpoint_path):
    '''
    function to load the baselie model
    simmilar to the training process
    sets the model into eval mode as well and
    populates the trained weights from the checkpoint file
    '''
    dropout = 0.5
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

    checkpoint = torch.load(checkpoint_path,lambda storage,loc:storage)

    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return model


def transposeimage(image, output_image_size, bounding_box):

    # collect image height and width
    height, width = image.shape[:2]

    # set scaling factor
    scale = 0.98

    # collect coordinates of bounding box
    x1 = bounding_box[0]  # dib face.left
    y1 = bounding_box[1]  # dib face.top
    x2 = bounding_box[2]  # dib face.right
    y2 = bounding_box[3]  # dib face.bottom

    # scale bounding box?
    size_bb = int(max(x2 - x1, y2 - y1) * scale)

    # control for out of bounds, x-y top left corner
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

#### Mattthew The code here is key  We receive Numpay array
def runmtcnnc(frames):
    boxes , confidence = mtcnn.detect(frames)
    results = []
    if boxes is not None:


        for singleface in boxes:
            ## Each of these is a detected Face. With this coordinate I will crop image to model specs.

            croppedimage = transposeimage(frames,244,singleface)

            output = process_image_through_model(croppedimage,model,transform)
            results.append(output)


        #frames.show()


        print("SENT BACK:" + str(output))
        return results
    return [0]



timefordoor = 3
imagefactor = 1.05
minsize = 35  # minimum size of face
threshold = [0.7, 0.6, 0.6]  # three steps's threshold
learnedpath = r"C:\Users\luisc\Documents\MaskLock\ML-dev\checkpoints\2020-11-30-15_24_14--best_model.pth"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(str(device))
mtcnn = MTCNN(keep_all=True, device=device,min_face_size = minsize,thresholds= threshold)
model = load_model(learnedpath)
transform = gettransforms()






#import facenet
# Create face detector
##################################################################

HOST = '10.0.0.213'  # The server's hostname or IP address
PORT = 12345        # The port used by the server
s = None
def connecttoserver(HOST,PORT,s):

    while s is None:
       try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((HOST, PORT))
       except:
            print("Host not found trying again in 5 seconds" )
            time.sleep(5)
            s = None
    return s

s = connecttoserver(HOST,PORT,s)
videosample = cv2.VideoCapture(0)
testingsample = []
while(videosample.isOpened()):

    ret, frame = videosample.read()

    #every minute capture frame
    if frame is  None: # escape end of frame capture
        break



    result = runmtcnnc(frame)
    #print("RESULT:" + str(type(result)))
    testingsample.append(result)


    if(len(testingsample) > 5):
        print("popping")
        testingsample.pop(0)
        print(str(testingsample))
        signal = 1
        for y in testingsample:
            for x in y:
                if x != 1:
                    signal = 0
                    break
        strcall = str(signal) + "|" + str(timefordoor)

        lockmechanism = 0
        try:
            s.sendall(strcall.encode())
            lockmechanism = s.recv(1024)
            print('Received',repr(lockmechanism))
        except ConnectionAbortedError:
            print("SERVER HAS BEEN SHUT")
            s = None
            s = connecttoserver(HOST,PORT,s)
        except ConnectionResetError:
            print("SERVER HAS BEEN SHUT")
            s = None
            s = connecttoserver(HOST,PORT,s)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

videosample.release()
cv2.destroyAllWindows()




