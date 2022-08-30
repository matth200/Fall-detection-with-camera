#!/usr/bin/python3
import numpy as np
import time
import cv2
import os

from PIL import Image

import tensorflow as tf


#jouer des sons
import simpleaudio as sa

#on initialise des variables pour pouvoir jouer du son
wave_object = sa.WaveObject.from_wave_file('audio/alarme.wav')
print('playing sound using simpleaudio')
play_file = None


#on fixe la condition de chute sur 5secondes pour déclencher l'alarme
ALARM_TIMEOUT = 5
start_fall = 0

#la fonction qui est lancé quand il y a une chute
def alarme():
    global play_file
    print("ALARMEEEEEE")
    if play_file == None or not play_file.is_playing():
        play_file = wave_object.play()

#la taille de la matrice d'entrée pour le réseau de neurones de détection de chute
WIDTH=150
HEIGHT=150

#creation du detecteur
print("creation du modele")
model = tf.keras.models.load_model('./trained_network/weights')


#fonction qui vient formater notre image comme il faut pour pouvoir entrer dans notre réseau de neurones de détection
#chute
def add_padding(im, box):
    #manipulation de couleur, de taille, ...
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


#fonction qui renvoit la prediction d'une chute où non d'une personne
def get_prediction(im, box):
    im = add_padding(im,box)
    
    #prediction
    im = np.array([np.expand_dims(im, axis=2)])
    predictions = model.predict(im)
    fallen = np.argmax(predictions[0])
    print("prediction:",predictions)
    return int(fallen)

#on recupere la caméra de mon ordinateur avec opencv2
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://192.168.1.68:8080/video/mjpeg")


#chemin d'accès à l'algorithme YOLO
DARKNET_PATH = 'yolo/darknet'

#objet que YOLO peut reconnaître
labels = open(os.path.join(DARKNET_PATH, "data", "coco.names")).read().splitlines()

#couleur des rectangles de détection possible
colors = [(100,230,100),(100,100,200)]


#chargement de l'algorithme YOLO
net = cv2.dnn.readNetFromDarknet(os.path.join(DARKNET_PATH, "cfg", "yolov3.cfg"), "./yolo/yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


#on crée une boucle infini qui permet de gérer image par image de la caméra
while True:
    #timer pour pouvoir calculer le temps d'exécution de notre algorithme
    start_frame = time.time()
    #video live
    ret, image = cap.read()
    if ret == False:
        print("Erreur")
        break

    #on récupére l'image dans un autre format pour pouvoir la manipuler
    image_prediction =  Image.fromarray(image)
    image = np.array(image_prediction)
    image = np.array(image)

    #on prépare l'image pour l'envoyer dans YOLO
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    #on la passe dans l'algorithme YOLO
    net.setInput(blob)
    layer_outputs = net.forward(ln)


    #on récupére chaque détection de YOLO
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores).item()
            confidence = scores[class_id].item()

            if confidence > 0.3:
                #information de détection donnée par YOLO
                box = detection[0:4] * np.array([w, h, w, h])
                center_x, center_y, width, height = box.astype(int).tolist()

                #on récupére la position du rectangle avec la taille et la position du centre
                x = center_x - width//2
                y = center_y - height//2

                #on ajoute les données dans des listes
                boxes.append([x, y, width, height])
                confidences.append(confidence)
                class_ids.append(class_id)
    
    #on filtre les détection qui sont en dessous de 60% de confiance
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.6)
    is_a_fall = False
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            box = [x,y,x+w,y+h]
            #on récupére uniquement la détection de personne
            if labels[class_ids[i]]=="person":
                #on prédit avec notre algorithme la chute ou non de la personne
                fallen = get_prediction(image_prediction, box)
                if fallen:
                    text="Fallen person"
                    is_a_fall = True
                else:
                    text="Normal person"

                #on dessine sur l'image pour annoter les chutes détecter
                cv2.rectangle(image, (x, y), (x + w, y + h), colors[fallen], 2)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, colors[fallen], 2)
    
    #detection de chute qui dure dans le temps
    if is_a_fall:
        if start_fall==None:
            start_fall = time.time()
    else:
        start_fall = None

    if start_fall!=None:
        fall_duration = time.time()-start_fall
        if fall_duration>ALARM_TIMEOUT:
            alarme()
        cv2.putText(image, str(round(fall_duration,2)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX,0.5, (colors[1]), 2)
    
    #affichage de l'image finale
    cv2.imshow("frame", image)
    #calcul du temps de calcul et du nombre de FPS
    duration_frame = time.time()-start_frame
    print("duration_frame:{}, FPS:{}".format(round(duration_frame,3),round(1.0/duration_frame,1)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
