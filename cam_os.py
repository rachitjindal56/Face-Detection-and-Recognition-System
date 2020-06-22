import cv2
import numpy as np
import pickle
from PIL import Image

def key_from_value(dicti,k):
    for key,value in dicti.items():
        if value == k:
            return key

cascade = cv2.CascadeClassifier("C:/Users/Rachit/Onedrive/Desktop/Files/VS/hcascade.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:/Users/Rachit/Onedrive/Desktop/Files/VS/Trained.yml")
cap = cv2.VideoCapture(0)

label_dict = {}
with open('C:/Users/Rachit/Onedrive/Desktop/Files/VS/dictionary.pickle','rb') as f:
    label_dict = pickle.load(f)

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for x,y,w,h in faces:
        roi = gray_frame[y:y+h,x:x+w]
        id_predicted,conf = recognizer.predict(roi)

        text = key_from_value(label_dict,id_predicted)
        Font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
        cv2.putText(frame,str(text), (x,y),Font,1, (0,255,0),1,cv2.LINE_AA)

    cv2.imshow("The Recognizer",frame)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()