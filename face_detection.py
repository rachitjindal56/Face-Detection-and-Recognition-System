import cv2
import numpy as np
import pickle
from PIL import Image
import os
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Path-> directory for all the files  
# path = "C:/Users/Rachit/Onedrive/Desktop/Files/Final"
path = os.path.dirname(__file__)

cascade_classifier = cv2.CascadeClassifier(os.path.join(path,"hcascade.xml"))

# LBPHFaceRecogniser classifies the detected faces
recogniser = cv2.face.LBPHFaceRecognizer_create()
# Loading trained model
recogniser.read(os.path.join(path,"train.yml"))

names = []
date_time = []
label_dict = {}


with open(os.path.join(path,"label.pickle"),'rb') as f:
    # Loading the dictionary 
    label_dict = pickle.load(f)

# Function to extract key from value
def key_from_label(label):
    for key,value in label_dict.items():
        if value==label:
            return key

cap = cv2.VideoCapture(0)

while True:
    count = 0
    ret,frame = cap.read()
    
    # Convderting BGR to Gray frame
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = cascade_classifier.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=2
    )

    for x,y,w,h in faces:
        # ROI-> Region Of Interest in image
        roi_img = gray_frame[y:y+h, x:x+w]

        # Making prediction from trained model
        id_prediction,confidence = recogniser.predict(roi_img)

        # confidenct >= 90
        if(confidence >= 90):
            text = key_from_label(id_prediction)
            count = 1

            if text not in names:
                names.append(str(text))
                date_time.append(datetime.today())

        else:
            text = "Unknown"

        # Making rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),1,cv2.LINE_AA)

        # Putting text around face
        cv2.putText(frame,str(text),(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,225,0),2,cv2.LINE_AA)

    # Displaying frame
    cv2.imshow('The attendance system',frame)

    # Break if identified 
    if cv2.waitKey(1) & count==1:
        break

cap.release()
cv2.destroyAllWindows()

# Checking if attendance.csv exists
data_csv = os.path.isfile(os.path.join(path,"attendance.csv"))

# If file exist append the attendance_csv
if data_csv==True:

    # Opening File
    data = pd.read_csv(os.path.join(path,"attendance.csv"))
    # Appending csv
    x = data.append({'Names':names[0],'Date-Time':date_time[0]},ignore_index=True)
    data = x

    # Droping unnecessary columns
    data.dropna(axis=1,inplace=True)

# If file do not exist creating file attendance.csv
else:
    # Dataset contains Name and Date-Time column
    data = pd.DataFrame(columns=["Names","Date-Time"])
    data['Names'] = names
    data['Date-Time'] = date_time

# Saving attendance
data.to_csv(os.path.join(path,"attendance.csv"))