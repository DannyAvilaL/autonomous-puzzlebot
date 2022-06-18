# Autor: Roberto David Manzo González
# se importan las librerias y modulos necesarios
import numpy as np
import pandas as pd
import os
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import random
np.random.seed(42)

from matplotlib import style
style.use('fivethirtyeight')

#path del dataset
train_path = 'training_com/'
test_path = 'test_or/'

#image resize dimension
img_dim = (30,30,3)

#listando las carpetas
num_signals = len(os.listdir(train_path))

#diccionario de clases
class_names = {0: "20km-1", 1: "30km-1", 2: "50km-1", 3: "60km-1", 4: "70km-1", 5: "80km-1", 6: "termina restriccion 80km", 7: "100km-1",
               8: "120km-1", 9: "no revasar", 10: "vehiculos pesados no pueden revasar", 11: "interseccion con prioridad",
               12: "prioridad", 13: "ceder el paso", 14: "stop", 15: "circulacion prohibida", 16: "prohibido paso a camiones",
               17: "entrada prohibida", 18: "otros peligros", 19: "curva peligrosa iz", 20: "curva peligrosa der",
               21: "curvas peligrosas iz", 22: "perfil irregular", 23: "pavimento derrapante", 24: "reduccion por la derecha",
               25: "obras", 26: "semaforo", 27: "paso peatonal", 28: "ninhos", 29: "ciclistas", 30: "pavimento derrapante por nieve",
               31: "animales salvajes", 32: "fin de prohibiciones", 33: "vuelta a la derecha obligatoria", 34: "vuelta a la izquierda obligatoria",
               35: "seguir recto obligatorio", 36: "sentido recto y derecha permitidos", 37: "sentido recto e izquierda permitidos",
               38: "paso obligatorio der", 39: "paso obligatorio iz", 40: "rotonda obligatoria", 41: "termina prohibicion para revasar",
               42: "termina prohibicion para vehiculos pesados para revasar"}


img_list = []
img_lbl = []

#for para hacer lectura de las imagenes
for i in range(num_signals):
    imgs = os.listdir(train_path+str(i))
    for img in imgs:
        try:
            image = cv.imread(train_path + str(i)+ '/' + img)
            #print(train_path + str(i)+ '/' + img)
            #img_clr = Image.fromarray(image, 'RGB')
            img_clr = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            img_resize = cv.resize(img_clr, (img_dim[0], img_dim[1]))
            img_list.append(np.array(img_resize))
            img_lbl.append(i)
        except:
            print('Error, probablemente: #### ' + img + ' #### tiene danos o no es una imagen')

#convirtiendo a np array
img_list = np.array(img_list)
img_lbl = np.array(img_lbl)

#print(img_list.shape, img_lbl.shape)
#shuffle de las imagenes de entrenamiento
shuffle_index = list(np.arange(img_list.shape[0]))
random.shuffle(shuffle_index)
img_list = img_list[shuffle_index]
img_lbl = img_lbl[shuffle_index]

#grupo de entrenamiento del 70% de las imagenes
x_train, x_val, y_train, y_val = train_test_split(img_list, img_lbl, test_size=0.3, random_state=42, shuffle=True)
x_train = x_train/255
x_val = x_val/255
print(x_train.shape)
print(x_val.shape)

y_train = keras.utils.to_categorical(y_train, num_signals)
y_val = keras.utils.to_categorical(y_val, num_signals)

print(y_train.shape)
print(y_val.shape)

#creación del modelo, se declaran las capas del mismo
model = Sequential([

        Conv2D(128, (5, 5), activation='relu', input_shape=img_dim),
        BatchNormalization(axis=-1),
        Conv2D(64, (5, 5), activation='relu'),
        BatchNormalization(axis=-1),
        MaxPooling2D(pool_size=(2, 2)),
        

        #convolution + RELU + pooling + norm
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(axis=-1),
        Conv2D(16, (3, 3), activation='relu'),
        BatchNormalization(axis=-1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(rate=0.4),


        #flatten
        Flatten(),

        #fully connected layer
        Dense(512, activation='relu'),
        Dropout(rate=0.4),
        
        Dense(43, activation='softmax')
])

#learning rate y cantidad de epochs
lr = 0.001
epochs = 30
opt = Adam(lr=lr, decay=lr / (epochs*0.5))
#se compila el modelo
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives()] )

#se rellena de manera artificial 
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest')

#entrenamiento del modelo
history = model.fit(aug.flow(x_train, y_train, batch_size=32), epochs=epochs, validation_data=(x_val, y_val))

#se guarda el modelo
model.save('model.h5')

#se hace una grafica para ver el comportamiento y se guarda
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.savefig('model_output.png')
