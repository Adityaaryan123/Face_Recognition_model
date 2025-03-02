#pylint:disable=no-member

import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = r'C:/Users/Aditya151/Downloads/coding/Python/python opencv/RESUME_PROJ1/Model/Face_recognition_model/faces'

haar_cascade = cv.CascadeClassifier('C:/Users/Aditya151/Downloads/coding/Python/python opencv/RESUME_PROJ1/Model/Face_recognition_model/haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('C:/Users/Aditya151/Downloads/coding/Python/python opencv/RESUME_PROJ1/Model/Face_recognition_model/face_trained.yml')
np.save('C:/Users/Aditya151/Downloads/coding/Python/python opencv/RESUME_PROJ1/Model/Face_recognition_model/features.npy', features)
np.save('C:/Users/Aditya151/Downloads/coding/Python/python opencv/RESUME_PROJ1/Model/Face_recognition_model/labels.npy', labels)