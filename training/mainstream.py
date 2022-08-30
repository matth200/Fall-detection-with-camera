#!/usr/bin/python3
import cv2
import numpy as np
from PIL import Image
import time
from datetime import date

#cap = cv2.VideoCapture("http://192.168.1.68:8080/video/mjpeg")
cap = cv2.VideoCapture(0)

#delay
DELAY = 5.0
counter = 0
start = time.time()

decompte_time = time.time()
decompte=0
print(int(DELAY))

if not cap:
    print("Erreur au niveau de la video")
else:
    while(True):
        ret, current_frame = cap.read()
        if ret == False:
            print("Erreur")
            break
        current_frame = np.array(current_frame)
        current_frame = np.flip(current_frame,axis=1)
        if time.time()-decompte_time>1.0:
            decompte+=1
            decompte_time = time.time()
            print(int(DELAY)-decompte%int(DELAY))

        if time.time()-start>=DELAY:
            start = time.time()
            name = "normal"
            #si impair c'est une personne qui tombe
            if counter%2==1:
                name = "falling"
            im = Image.fromarray(np.flip(current_frame, axis=2))
            im.save("./data/data_perso/{}-{}.png".format(counter, name))
            print("CHEEESSEEEE")
            counter+=1
        cv2.imshow("frame", current_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break