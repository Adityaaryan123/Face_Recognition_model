import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('C:/Users/Aditya151/Downloads/coding/Python/python opencv/RESUME_PROJ1/Model/Face_recognition_model/haar_face.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/Aditya151/Downloads/coding/Python/python opencv/RESUME_PROJ1/Model/Face_recognition_model/face_trained.yml')

img = cv.imread(r'C:/Users/Aditya151/Downloads/coding/Python/python opencv/RESUME_PROJ1/Model/Face_recognition_model/val/Elton John/1.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]),(20,20), cv.FONT_HERSHEY_COMPLEX,0.8, (0,0,255), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)