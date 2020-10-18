import cv2
import numpy as np
import torch
from PIL import ImageDraw, Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import subprocess
from subprocess import Popen



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
exedirectory = "'E:\One Drive\OneDrive\Mainproject\MaskLock\ML-dev\C++ Relay development.exe"


def openrelay(argument1, argument2): ## OPENS FOR ARUGMENT1 = SECONDS INTEGER  ## eventually Argument 2 will allow close and open commanmds
    ## argument 3 will evnetually be a a add functionality
    exedirectory = 'E:\One Drive\OneDrive\Mainproject\MaskLock\ML-dev\C++ Relay development.exe' + ' ' + str(argument1)

    Popen(exedirectory, shell=False)


def is_list_empty(list):
    if len(list) == 0 :
        return True
    return False

#### Mattthew The code here is key  We receive Numpay array
def runmtcnnc(frames):
    boxes , confidence = mtcnn.detect(frames)
    if boxes is not None:
        frames = Image.fromarray(frames)
        draw = ImageDraw.Draw(frames)

        for singleface in boxes:
            ## Each of these is a detected Face. With this coordinate I will crop image to model specs.
            print("Page #:" + str(len(boxes)) + " : " + str(confidence))


            draw.rectangle(singleface.tolist(), outline=(255, 0, 0), width=10)

        #frames.show()

        finalframe = np.asarray(frames)
        openrelay(1,2)
        return finalframe
    return frames








#import facenet
# Create face detector

minsize = 30  # minimum size of face
threshold = [0.7, 0.8, 0.8]  # three steps's threshold


resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=True, device=device,min_face_size = minsize,thresholds= threshold)




resnet = InceptionResnetV1(pretrained='vggface2').eval()




videosample = cv2.VideoCapture("Guardians.of.the.Galaxy.2014.1080p.BluRay.x264.YIFY.mp4")

start_frame_number = 24
while(videosample.isOpened()):

    ret, frame = videosample.read()
    print(type(frame[0]))
    #every minute capture frame
    if frame is  None: # escape end of frame capture
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    cv2.imshow('image', runmtcnnc(frame))

    videosample.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    start_frame_number = start_frame_number + 50  # 1440 = 1 minute every captuer new frame


    if cv2.waitKey(100) & 0xFF == ord('q'):
        break


videosample.release()
cv2.destroyAllWindows()







    #
    # cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
    # cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
    # cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
    # cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
    # cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)





