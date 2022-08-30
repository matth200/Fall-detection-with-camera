#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import np_utils


#taille de la matrice d'entrée du réseau de neurones de détection
WIDTH=150
HEIGHT=150


#le nombre maximale de donnée de fichier qu'il faut ajouter pour compléter la base de donnée de base
max_file = 1500


#fonction qui vient préparer l'image pour l'envoyer au réseau de neurones de détection des chutes
def load_image(path, box):

    #on manipule un peu l'image
    im=Image.open(path)
    im = im.convert("L")
    im=im.crop(box)
    h,w = np.shape(np.array(im))
    if w>h:
        a,b = 1, h/w
    else:
        a,b = w/h, 1
    #on change sa taille
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

def load_path(i_label, i_content):
    return contents[i_label][i_content],labels[i_label]

#lis les fichiers dans le dossier qui contient notre première base de données
print("on recupere les donnees")
print("recuperation 1")
images = Path("train/").rglob("*.png")
images = list(images)

#on les arrange aux hasards
images = np.array(images)
np.random.shuffle(images)

#on récupére les annotations qui vont avec les images
#ce sont des fichiers txt qui portent le même noms que les images
labels=[]
contents=[]
for path in images:
    path = str(path)
    sep = path.split("/")
    assert len(sep)==3

    text_path = "{}/{}/{}.txt".format(sep[0],sep[1],sep[2].split(".")[0])
    #read annotations
    lines = []
    with open(text_path,"r") as file:
        lines = file.read().split("\n")[:-1]
    for line in lines:
        raw_data = line.split(" ")
        assert len(raw_data) == 5
        fall, left, right, top, bot = raw_data
        box = (int(left),  int(top), int(right), int(bot))
        im = load_image(path, box)
        #on ajoute les infos
        contents.append(im)
        fall = int(fall)
        if fall<0:
            fall = 0
        labels.append(fall)

#2eme base de données
#recuperation de la base mpii_human_pose_v1 avec annotations
print("recuperation 2")
textes= Path("./data/mpii_human_pose_v1/images/").rglob("*.txt")
textes = list(textes)
print("len recup 2:", len(textes))
#on veille à ne pas dépasser un nombre trop grand de données en plus
if len(textes)>=max_file:
    textes = textes[:max_file+1]

#on récupére les annotations en plus 
#dans les fichiers txt portant le même nom sur le même principe que la première base de données
added_person = 0
for path in textes:
    path = str(path)
    sep = path.split("/")
    image_path = "{}/{}.jpg".format("/".join(sep[:-1]),sep[-1].split(".")[0])
    #read annotations
    lines = []
    with open(path,"r") as file:
        lines = file.read().split("\n")[:-1]
    for line in lines:
        added_person+=1
        raw_data = line.split(" ")
        assert len(raw_data) == 5
        fall, left, right, top, bot = raw_data
        box = (int(left),  int(right), int(top), int(bot))

        #on prépare les données à envoyer au réseau de neurones
        im = load_image(image_path, box)

        #on ajoute les infos
        contents.append(im)
        fall = int(fall)
        if fall<0:
            fall = 0
        labels.append(fall)
#on affiche le nombre de personne ajouté avec les fichiers en plus
#car il peut y avoir plus d'une personne par image
print("nombre de personne ajouté: ",added_person)


#on normalise les images pour que chaque valeur de chaque pixel soit compris entre 0 et 1
contents = np.array(contents, dtype="float32")/255.0
#on converti une valeur d'étiquetage en une liste d'étiquetage
labels = np_utils.to_categorical(labels, 2)
print("----\n",labels)
#on forme la bonne dimension pour notre liste
contents = np.expand_dims(contents,axis=3)
#on sépare notre base de données en 2, valeurs d'entrainement et valeurs de test qu'il n'aura jamais vu
x_train, x_test, y_train, y_test = train_test_split(contents, labels, test_size=0.2, shuffle=True)

#on affiche une valeur de test pour voir les données
print("affichage test")
plt.title(y_train[0])
plt.imshow(x_train[0])
plt.show()


#on regarde le nombre d'image que notre réseau de neurones aura pour s'entrainer avec
print("Taille :", len(x_train))

#on regarde la mixiter des valeurs
a,b = 0,0
for fall in labels:
    if fall[0]:
        b+=1
    else:
        a+=1
#on affiche le pourcentage de présence de chaque catégorie
print("personne tombé: {} et non-tombé:{}".format(a,b))
print("pourcentage de personne tombé:{} et non-tombé:{}".format(a/len(labels),b/len(labels)))


#on crée enfin le modèle
print("creation du modele")

# model = Sequential()
# model.add(Conv2D(10, kernel_size=(10, 10), activation='relu', input_shape=(WIDTH,HEIGHT,1)))
# model.add(MaxPool2D(pool_size=(3, 3)))
# model.add(Dropout(0.25))
# model.add(Conv2D(10, kernel_size=(10, 10), activation='relu'))
# model.add(Dropout(0.25))
# model.add(Conv2D(5, kernel_size=(6, 6), activation='relu'))
# model.add(Dropout(0.25))
# model.add(MaxPool2D(pool_size=(3, 3)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dropout(0.4))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(2, activation='softmax'))


#réseau de neurone de convolution assez petit ~ 65,000 paramètres à faire varier
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(30, kernel_size=(10,10), activation="relu", padding="same", input_shape=(WIDTH,HEIGHT,1)),
    tf.keras.layers.MaxPool2D(pool_size=(3,3)),
    tf.keras.layers.Conv2D(10, kernel_size=(10, 10), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(3,3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(20,activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2,activation="softmax")
])

print("entrainement")


#on configure l'entrainement avec des paramètres usuels d'entrainement
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(x_train, y_train, epochs = 3, verbose = 1)

#on affiche la courbe de précision de notre algorithme
print("affichage apprentissage")
plt.plot(history.history["accuracy"])
plt.show()


#on test sur des valeurs inconnus
print("test sur valeur inconnu")
evaluation = model.evaluate(x_test,y_test, verbose=0)
#erreur de classification
print("Erreur:", evaluation[0])
#erreur de précision de classification
print("Precision:", evaluation[1])

#on enregistre notre réseau de neurone entrainé dans un fichier pour le réutiliser dans d'autre programme
print("enregistrement du réseau de neuronne")
savename = "./trained_network/weights"
i=0
while True:
    savefile = savename+str(i)
    if os.path.isfile(savefile):
        model.save(savefile)
        print("Sous le nom:",savefile)
        break


#on essaye sur des images aux hasards et on affiche les résultats
print("test sur une image dans la banque de test au hasard")
def get_prediction(im):
    im = np.array([np.expand_dims(im, axis=2)])
    predictions = model.predict(im)
    fallen = np.argmax(predictions[0])
    return predictions[0]

for x in x_test:
    plt.title(get_prediction(x))
    plt.imshow(x)
    plt.show()