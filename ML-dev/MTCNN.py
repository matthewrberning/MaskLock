import cv2
import numpy as np
import torch
from PIL import ImageDraw, Image
from facenet_pytorch import MTCNN


learnedpath = r""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device,min_face_size = minsize,thresholds= threshold)
imagefactor = 1.05
minsize = 35  # minimum size of face
threshold = [0.7, 0.8, 0.8]  # three steps's threshold

model = torch.load(learnedpath)
transforms = gettransforms()
model.eval()

#### Mattthew The code here is key  We receive Numpay array
def runmtcnnc(frames):
    boxes , confidence = mtcnn.detect(frames)
    if boxes is not None:
        frames = Image.fromarray(frames)

        for singleface in boxes:
            ## Each of these is a detected Face. With this coordinate I will crop image to model specs.
            singleface[0] = (imagefactor/2)*(singleface[0]-singelface[2])
            singleface[1] = (imagefactor/2)*(singleface[1]-singelface[3])
            singleface[2] = (imagefactor/2)*(singleface[2]-singelface[0])
            singleface[3]= (imagefactor/2)*(singleface[3]-singelface[1])
            croppedimage = frames.crop(singleface)

            output = runpicturetomodel(croppedimage)


        #frames.show()

        finalframe = np.asarray(frames)

        return finalframe
    return frames


def gettransforms():
    pre_trained_mean, pre_trained_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std)
    ])
    return  val_transforms
def runpicturetomodel(cv2image)

#import facenet
# Create face detector
##################################################################


videosample = cv2.VideoCapture()


while(videosample.isOpened()):

    ret, frame = videosample.read()
    print(type(frame[0]))
    #every minute capture frame
    if frame is  None: # escape end of frame capture
        break



    runmtcnnc(frame)

    videosample.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break


videosample.release()
cv2.destroyAllWindows()



def runpicturetomodel(cv2image):
    cv2image = cv2.resize(cv2image, (224, 224))
    cv2image = transforms(cv2image)
    cv2image = cv2image.unsqueeze(0)
    cv2image = cv2image.to(device)

    with torch.no_grad:
        outputs = model(cv2image)

    return outputs



