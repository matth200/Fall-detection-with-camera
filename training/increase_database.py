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
path_dir = "./data/mpii_human_pose_v1/"

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



# #récupére tous les chemins vers les images .jpg
# images = Path(path_dir).rglob("*.jpg")
# images = list(images)
# #on extrait toutes les personnes de ces images pour les ajouter à notre base de données
# for i,path in enumerate(images):
#     print(i,":",path)
#     extract_person(path)


#on selectionne des images spécifiques dans la base qu'on ne pourrait pas confondre avec des personnes qui sont tombées
path_file = path_dir+"data.csv"
data = []
with open(path_file,"r") as file:
    data = file.read().split("\n")[:-1]
    tmp = []
    for elt in data:
        elt = elt.split("\"")
        new_elt = elt[0].split(",")
        new_elt.extend(elt[1:])
        tmp.append(new_elt)
    data = tmp

keys = data[0]
data = data[1:]

categories = {}
liste_images = []
for elt in data:
    name = elt[keys.index("NAME")]
    scale = elt[keys.index('Scale')]
    activity = elt[keys.index('Activity')]
    category = elt[keys.index('Category')]
    liste_images.append(name)
    if category not in categories.keys():
        categories[category] = []
    categories[category].append([name,activity,category])

def load_image(path):
    path = "./mpii_human_pose_v1/images/"+path
    image =  Image.open(path)
    image = np.array(image)
    plt.imshow(image)
    plt.show()

#name_categories = ["home activities","sitting, talking in person, on the phone, computer, or text messaging, light effort",'standing, talking in church',"postal carrier, walking to deliver mail","carpentry, outside house, building a fence",'carrying, loading or stacking wood','loading unloading a car, implied walking','food shopping with or without a grocery cart, standing or walking','bakery, general','garbage collector, walking, dumping bins into truck','digging, spading, filling garden, composting','hockey, field','dancing','bicycling, mountain']
name_categories = list(categories.keys())
n = len(categories[name_categories[-1]])
print("il y a",n,"élements de la catégorie",name_categories[-1])

sum = 0
for elt in name_categories:
    sum+=len(categories[elt])
print("Nous avons",sum,"éléments en tout.")


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