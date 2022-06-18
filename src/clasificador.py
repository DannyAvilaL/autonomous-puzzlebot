#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Autores
# Daniela Avila Luna
# Roberto David Manzo Gonzalez

import cv2 as cv
import tensorflow as tf
from tensorflow.keras import backend as bk
from tensorflow.config import set_visible_devices
import numpy as np

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32


# Label Overview
bk.clear_session()
set_visible_devices([], 'GPU')
print('init')

rospy.init_node("neural_network")
rate = rospy.Rate(10)
frame = frame2 = None
r = 0
res, resullt = '', ''
we, he, re = 0, 0, 0

ultimo_result = ""

class_pub = rospy.Publisher("/signal_class", Int32, queue_size=10)

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons',
            43:'Sin definir'}

model_path = "/home/puzzlebot/Desktop/modelos/model4"#/signs_rec/model"
loaded_model = tf.keras.models.load_model(model_path)

if loaded_model:
    print('Modelo cargado')
else:
    print("Error al cargar el modelo, volver a correr el nodo")

def img_callback(msg):
    global frame
    #Obteniendo la imagen de la camara y seleccionando una parte del frame
    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    frame = frame[0:95, 30:len(frame[0])-20]
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
def end_callback():
    rospy.loginfo("Shutting down")
    cv.destroyAllWindows()

rospy.Subscriber("/video_source/raw", Image , img_callback)

def circles(frame):
    global r
    #_,frame = video.read()
    output = frame.copy()

    img = cv.medianBlur(frame,5)
    img = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    # detect circles in the image
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1.2, 100,minRadius=900,maxRadius=1000)
    senales = []
    # ensure at least some circles were found
    if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            #Agregando los círculos en una lisa de senales
            senales.append(frame[y-r-20:y+r+20,x-r-20:x+r+20])
    #print(f"Circulos cortados: {len(senales)}")

    ## LA LISTA DE senales ES LA LISTA QUE TIENE LOS RECORTES DE LOS FRAMES
    try:
        #cv.imshow('muestra', senales[0])
        return True, senales[0]
    except IndexError: #En caso de que no se detecte un senalamiento
        return False, False
    except cv.error: # luego sale un error de opencv donde se pide que la imagen tenga dimensiones
        return False, False
    

def stop():
    global we, he, re
    senales = []
    frame_HSV = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    frame_threshold = cv.inRange(frame_HSV, (0, 80, 59), (180, 255, 255))
    bin_r = cv.dilate(frame_threshold, (2,2), iterations=1)
    cnt,_ = cv.findContours(bin_r, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i in cnt:
        ep = 0.01*cv.arcLength(i, True)
        fig_aprx = cv.approxPolyDP(i, ep, True)
        if cv.contourArea(fig_aprx) > 350:
            x,y,w,h = cv.boundingRect(fig_aprx)
            we, he = w, h
            senales.append(frame[y-10:y+h+10,x-10:x+w+10])

    try:
        return True, senales[0]
    except IndexError: #En caso de que no se detecte un senalamiento
        return False, False 
    except cv.error: # luego sale un error de opencv donde se pide que la imagen tenga dimensiones
        return False, False

def input_model_img(img):
    global res, result, loaded_model
    try:
        frame2 = img
        img = cv.resize(img, (30,30))
        expand_input = np.expand_dims(img,axis=0)
        input_data = np.array(expand_input)
        input_data = input_data/255

        pred = loaded_model.predict(input_data)
        result = pred.argmax()
        res = classes[result]

    except cv.error:
        pass

def main(): 
    global res, frame, re, r, result, ultimo_result
    video = frame
    c, c_img = circles(video)
    s, s_img = stop()
    # Si se encuentra un stop, senales azules o sin_limt
    if c:
        input_model_img(c_img)
    # Si se encuentra un stop
    if s:
        input_model_img(s_img)

    if res == "Stop" and not (ultimo_result == "Stop"):
        class_pub.publish(result)
        ultimo_result = "Stop"
    elif res == "End speed + passing limits" and not(ultimo_result == "End speed + passing limits"):
        class_pub.publish(result)
        ultimo_result = "End speed + passing limits"
    elif res == "Turn right ahead" and not(ultimo_result == "Turn right ahead"):
        class_pub.publish(result)
        ultimo_result = "Turn right ahead"
    elif res == 'Ahead only' and not(ultimo_result == 'Ahead only'):
        class_pub.publish(result)
        ultimo_result = 'Ahead only'

    
if __name__ == "__main__":
    rospy.sleep(1)
    class_pub.publish(1)
    while not rospy.is_shutdown():
        try:
            rate.sleep()
            main()
        except rospy.exceptions.ROSInterruptException:
            pass