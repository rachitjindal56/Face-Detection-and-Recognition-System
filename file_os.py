import os
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_PATH,"im")
cascade = cv2.CascadeClassifier('C:/Users/Rachit/Onedrive/Desktop/Files/VS/hcascade.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

x_img = []
y_label = []
label_dic = {}
id_current = 0

for root,dirs,files in os.walk(image_path):
    for file in files:
        if file.endswith("png") or file.endswith('jpg'):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path))

        if label not in label_dic.keys():
            label_dic[label] = id_current
            id_current += 1

        _id = label_dic[label]

        pil_img = Image.open(path).convert("L")
        np_pil_img = np.array(pil_img)
        faces = cascade.detectMultiScale(
            np_pil_img,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for x,y,w,h in faces:
            roi_img = np_pil_img[y:y+h, x:x+w]
            x_img.append(roi_img)
            y_label.append(_id)

recognizer.train(x_img,np.array(y_label))
recognizer.save("C:/Users/Rachit/Onedrive/Desktop/Files/VS/Train.yml")

with open("C:/Users/Rachit/Onedrive/Desktop/Files/VS/dict.pickle",'wb') as f:
    pickle.dump(label_dic,f)


