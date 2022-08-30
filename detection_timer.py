#!/usr/bin/python3

import numpy as np
import time
import cv2
import os

from PIL import Image

import tensorflow as tf


#jouer des sons
import simpleaudio as sa

wave_object = sa.WaveObject.from_wave_file('audio/alarme.wav')
print('playing sound using simpleaudio')
play_file = None


#variable pour la détection de plusieurs personnes
#condition sur le temps maximal de chute
ALARM_TIMEOUT = 5
MAX_DISTANCE = 60

start_falls = []
start_timer = []


#fonction à activer quand une chute est détectée
def alarme():
    global play_file
    print("ALARMEEEEEE")
    if play_file == None or not play_file.is_playing():
        play_file = wave_object.play()


#fonctionn de calcul de différence entre 2 rectangles de détection
def distanceBox(box1,box2):
    #on récupére la position et la taille des rectangles
    x1,y1,xw,yh = box1
    w1,h1 = xw-x1,yh-y1
    x2,y2,xw,yh = box2
    w2,h2 = xw-x2,yh-y2

    #distance de manhattan sur la position
    sum = abs(x1-x2)+abs(y1-y2)

    #distance de manhattan sur la taille
    sum += abs(w1-w2) + abs(h1-h2)

    #on divise par 4 car on a l'addition de la distance sur 4 dimensions (x,y,w,h)
    return sum/4.0

#fonction qui renvoie l'indice de la valeur minimum dans une liste ainsi que la valeur du min
def getMinIndiceAndValueOfMin(liste):
    min_i = 0
    min_liste = liste[0]
    n = len(liste)
    for i in range(1,n):
        elt = liste[i]
        if min_liste>elt:
            min_i = i
            min_liste = elt
    return min_i,min_liste


#taille de la matrice d'entrée de notre réseau de neurones de détection de chute
WIDTH=150
HEIGHT=150

#creation du detecteur
print("creation du modele")
model = tf.keras.models.load_model('./trained_network/weights')

#préparation de l'image pour qu'elle puisse entrer dans le réseau de neurones de détection de chute
def add_padding(im, box):
    #manipulation de couleur, de taille ....
    im = im.convert("L")
    im=im.crop(box)
    h,w = np.shape(np.array(im))
    if w>h:
        a,b = 1, h/w
    else:
        a,b = w/h, 1
    im=im.resize((int(a*WIDTH),int(b*HEIGHT)))
    im=np.array(im)
    dtype = im.dtype
    #on ajoute le padding si besoin
    padding_w = int((WIDTH*(1-a)))//2
    padding_h = int((HEIGHT*(1-b)))//2

    #height padding
    if padding_h!=0:
        im = np.vstack([np.zeros((padding_h,WIDTH), dtype),im,np.zeros((padding_h,WIDTH),dtype)])
        h,w = im.shape
        if h!=HEIGHT:
            im = np.vstack([im, np.zeros((HEIGHT-h, WIDTH), dtype)])
    
    if padding_w!=0:
        im = np.column_stack([np.zeros((HEIGHT,padding_w), dtype),im,np.zeros((HEIGHT,padding_w), dtype)])
        h,w = im.shape
        if w!=WIDTH:
            im = np.column_stack([im, np.zeros((HEIGHT, WIDTH-w), dtype)])

    #correction pixel par pixel
    while im.shape != (HEIGHT,WIDTH):
        h,w = im.shape
        if w!=WIDTH:
            im = np.column_stack([im, np.zeros((HEIGHT, 1), dtype)])
        if h!=HEIGHT:
            im = np.vstack([im, np.zeros((1, WIDTH), dtype)])
    if im.shape != (HEIGHT,WIDTH):
        print("ERRREEEUUURRRR")

    return im

#fonction qui fait passer l'image demande dans notre algorithme et qui renvoit la prédiction de chute ou non
def get_prediction(im, box):
    im = add_padding(im,box)
    
    #prediction
    im = np.array([np.expand_dims(im, axis=2)])
    predictions = model.predict(im)
    fallen = np.argmax(predictions[0])
    #print("prediction:",predictions)
    return int(fallen)

#on recupere la caméra de l'ordinateur
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://192.168.1.68:8080/video/mjpeg")

#chemin d'accès à l'algorithme YOLO
DARKNET_PATH = 'darknet'

#objet reconnaissable par YOLO
labels = open(os.path.join(DARKNET_PATH, "data", "coco.names")).read().splitlines()

#couleur des rectangles de détection de chute
colors = [(100,230,100),(100,100,200)]

#chargement de l'algorithme YOLO
net = cv2.dnn.readNetFromDarknet(os.path.join(DARKNET_PATH, "cfg", "yolov3.cfg"), "yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

#boucle infini qui permet de gérer le flux vidéo
while True:
    #chrono qui permet de calculer la durée de calcul de notre algorithme sur une image
    start_frame = time.time()
    #video live
    ret, image = cap.read()
    if ret == False:
        print("Erreur")
        break

    #récupération de l'image dans une autre forme pour pouvoir effectuer des manipulations dessus
    image_prediction =  Image.fromarray(image)
    image = np.array(image_prediction)
    image = np.array(image)

    #preparation de l'image à envoyer à YOLO
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    #envoie de l'image à YOLO
    net.setInput(blob)
    layer_outputs = net.forward(ln)


    #on récupére les informations que YOLO nous renvoie
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores).item()
            confidence = scores[class_id].item()

            if confidence > 0.3:
                #rectangle de détection
                box = detection[0:4] * np.array([w, h, w, h])
                center_x, center_y, width, height = box.astype(int).tolist()

                #on récupére la position avec la position du centre ainsi que la taille du rectangle
                #de détection
                x = center_x - width//2
                y = center_y - height//2

                #ajout des données formatées dans des listes
                boxes.append([x, y, width, height])
                confidences.append(confidence)
                class_ids.append(class_id)

    #on récupére les rectangles de détection au dessus de 60% de confiance
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.6)
    prev_falls = start_falls
    start_falls = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            box = [x,y,x+w,y+h]
            #on récupére uniquement la détection de personne
            if labels[class_ids[i]]=="person":
                #étiquetage de la personne avec
                #notre réseau de neurones
                fallen = get_prediction(image_prediction, box)
                if fallen:
                    #suivi plus complexe avant affichage de la chute
                    start_falls.append(box)
                else:
                    #affichage de la personne si elle n'est pas tombé
                    cv2.rectangle(image, (x, y), (x + w, y + h), colors[fallen], 2)
                    cv2.putText(image, "Normal person", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, colors[fallen], 2)

        #Suivi de la chute entre 2 images
        #permutation
        permutation_falls = []
        if len(prev_falls)!=0:
            for box1 in start_falls:
                distances = []
                for i in range(len(prev_falls)):
                    box2 = prev_falls[i]
                    distances.append(distanceBox(box1,box2))
                #on récupére l'indice de la distance la plus petite
                min_i,min_value = getMinIndiceAndValueOfMin(distances)
                if min_value > MAX_DISTANCE:
                    min_i = None
                permutation_falls.append(min_i)
        else:
            permutation_falls = [None]*len(start_falls)

        #on effectue les permutations comme il se doit
        prev_start_timer = start_timer
        start_timer = []
        new_falls = []
        #permutation de i0 (indice dans start_falls) vers i1(indice dans prev_falls) si i1 != None
        for i0,i1 in enumerate(permutation_falls):
            #si on a aucune permutation avec l'ancienne liste c'est que c'est une nouvelle detection 
            if i1==None:
                start_timer.append(time.time())
                new_falls.append(start_falls[i0])
            #on a une permutation
            else:
                #on récupére l'ancienne 
                start_timer.append(prev_start_timer[i1])
                new_falls.append(start_falls[i0])
        
        start_falls = new_falls

        
        #on dessine les rectangles avec les informations récupérées
        nbr_fallen = len(start_falls)
        print("{} fallen person".format(nbr_fallen))
        for i in range(nbr_fallen):
            x1, y1, x2, y2 = start_falls[i]
            timer = start_timer[i]
            duration = time.time()-timer
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[1], 2)
            if duration > ALARM_TIMEOUT:
                alarme()
            cv2.putText(image,"Fallen person: {}s".format(round(duration,2)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, colors[1], 2)
            #affichage du nombre de personne tombée et la durée
            print("Person fallen n°{} duration:{}s".format(i,round(duration,2)))
    

    duration_frame = time.time()-start_frame
    cv2.imshow("frame", image)
    print("duration_frame:{}, FPS:{}".format(round(duration_frame,3),round(1.0/duration_frame,1)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
