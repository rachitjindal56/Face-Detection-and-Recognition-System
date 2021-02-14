import os
import numpy as np
from PIL import Image
import cv2
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

#                     Preparing data for training

x_img = []
y_label = []
label_dict = {}
curr_id = 0

# Path-> path of directory containing all the traind data
path = os.path.dirname(__file__)
# Loading harcascade classifier
cascade_classifier = cv2.CascadeClassifier(os.path.join(path,"hcascade.xml"))

# Im is the dir containing images
for root,dirs,files in os.walk(os.path.join(path,"Im")):

    # Iterating over all the images to train the face recogniser
    for file in files:

        # Selection of only .png and .jpg files
        if file.endswith('png') or file.endswith('jpg'):

            # Storing path of the image file
            img_path = os.path.join(root,file)
            # Storing label
            label = os.path.basename(img_path)
            label = label.split('.')[0]
            
        # Storing the label in dictionary
        if label not in label_dict.keys():
            label_dict[label] = curr_id
            curr_id += 1
            
        _id = label_dict[label]

        # Converting the image to Gray scale and converting into numpy array
        img = Image.open(img_path).convert("L")
        np_img = np.array(img)

        # Face detection bias
        faces = cascade_classifier.detectMultiScale(
            np_img,
            scaleFactor=1.3,
            minNeighbors=2
        )
            
        for x,y,w,h in faces:

            # Region of interest of the face 
            roi_img = np_img[y:y+h, x:x+w]
            # Appending array with image data
            x_img.append(roi_img)
            y_label.append(_id)


# Creating face classifer
recogniser = cv2.face.LBPHFaceRecognizer_create()
# Training the classifer 
recogniser.train(x_img,np.array(y_label))
# Saving the model as Train.yml
recogniser.save(os.path.join(path,"Train.yml"))

# Saving dictionary containg the labels as label.pickle
with open(os.path.join(path,"label.pickle"),'wb') as f:
    pickle.dump(label_dict,f)