# WORK IN PROGRESS. THE EYE DETECTION IS SCUFFED

import numpy as np
import cv2

filename = input("Input file name: ")

img = cv2.imread(filename)
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

for (x,y,w,h) in faces:
    roi = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.3, minNeighbors=5)

    """
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
    """


    if len(eyes) != 2:
        print('Detected face somehow doesn\'t have two eyes. Not drawing a box')
    else:
        # eyes is an array of boxes defined by [x,y,w,h]

        points = [np.array([ex+ew/2, ey+eh/2]) for (ex,ey,ew,eh) in eyes]
        p1, p2 = points
        # The coordinates of the center of each box

        thickness = sum([ew + eh for (ex,ey,ew,eh) in eyes]) / 8
        # The thickness of our line should be half the average side length of the boxes.

        diff = np.subtract(p2, p1)
        unit = diff / np.linalg.norm(diff) * thickness
        rot_unit = np.array([-unit[1], unit[0]])

        unit_scale = 1.5
        rot_unit_scale = 0.7

        unit *= unit_scale
        rot_unit *= rot_unit_scale

        rectangle = [p1 - unit + rot_unit, p1 - unit - rot_unit,
                     p2 + unit - rot_unit, p2 + unit + rot_unit]
        rectangle = np.array([[int(a), int(b)] for [a, b] in rectangle])
        cv2.fillPoly(roi, np.int32([rectangle]), (0,0,0))



cv2.imshow('Eyes detected', img)


cv2.waitKey(0)
cv2.destroyAllWindows()