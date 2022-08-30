#!/usr/bin/python3
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

from PIL import Image

#taille de la matrice d'entrée de notre réseau de neurone
WIDTH=150
HEIGHT=150

#chemin où se trouve les images non-annotées
path_dir = "./data/data_perso/"

print("Ajout des metadonnées aux images du dossier {}".format(path_dir))

#chemin où se trouve l'algorithme YOLO
DARKNET_PATH = '../yolo/darknet'


#éléments reconnaissable par YOLO
labels = open(os.path.join(DARKNET_PATH, "data", "coco.names")).read().splitlines()

#couleur des rectangles de detection (r,g,b)
colors = [(100,230,100),(100,100,200)]


#récupération de l'algorithme YOLO
net = cv2.dnn.readNetFromDarknet(os.path.join(DARKNET_PATH, "cfg", "yolov3.cfg"), "../yolo/yolov3.weights")

#on récupére les dernières couches
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

liste = []

def extract_person(path):
    image =  Image.open(path)
    image = np.array(image)

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)


    layer_outputs = net.forward(ln)


    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores).item()
            confidence = scores[class_id].item()

            if confidence > 0.3:
                box = detection[0:4] * np.array([w, h, w, h])
                center_x, center_y, width, height = box.astype(int).tolist()

                x = center_x - width//2
                y = center_y - height//2

                boxes.append([x, y, width, height])
                confidences.append(confidence)
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.6)

    list_person = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            if labels[class_ids[i]]=="person":
                list_person.append(boxes[i])

                #test
                #cv2.rectangle(image, (x, y), (x + w, y + h), colors[0], 2)

    # plt.imshow(image)
    # plt.show()

    #ajout des infos aux fichiers du même nom
    path = str(path)
    path_text = "{}/{}.txt".format("/".join(path.split("/")[:-1]),(path.split("/")[-1]).split(".")[0])
    print(path_text)
    with open(path_text, "w") as file:
        for box in list_person:
            x, y, w, h = box
            file.write("{} {} {} {} {}\n".format(-1,x,y,x+w,y+h))
    global liste
    liste.append(path_text)
    return len(list_person)



#récupére tous les chemins vers les images .jpg
images = Path(path_dir).rglob("*.jpg")
images = list(images)
#on extrait toutes les personnes de ces images pour les ajouter à notre base de données
for i,path in enumerate(images):
    print(i,":",path)
    extract_person(path)



def load_image(path):
    path = "./mpii_human_pose_v1/images/"+path
    image =  Image.open(path)
    image = np.array(image)
    plt.imshow(image)
    plt.show()



added_person = 0
added_file = 0

for name in liste_images:
    path = name
    path = "./data/mpii_human_pose_v1/images/"+path
    if path not in liste:
            added_file+=1
            added_person+=extract_person(path)


print("Ajout de",added_person,"personnes")
print("Ajout de",added_file,"fichiers")



liste = list(set(liste))
print(len(liste))