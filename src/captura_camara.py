#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Autor: Daniela Avila Luna

# Programa que permite tomar capturas 
# de la imagen al que esta suscrito el nodo

import cv2 as cv
import numpy as np
import cv_bridge
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


# Inicializando el nodo
rospy.init_node("captura_camara", anonymous=False)

# Variables globales
rate = rospy.Rate(10)
frame = frame2 = None
brigde = cv_bridge.CvBridge()
cont = 1


# Funciones de callback
def image_callback(msg):
    global frame, th, img, contornos, M, frame2
    frame = brigde.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    frame = frame[15:95, 30:len(frame[0])-20]
    frame2 = frame.copy()

rospy.Subscriber("/video_source/raw", Image, image_callback)

while not rospy.is_shutdown():
    try:
        rate.sleep()
        cv.imshow("camara", frame)
        
        tecla = cv.waitKey(33)

        if tecla == 113: #q
            break

        elif tecla == 115 or tecla == 101: #s o e
            cv.imwrite("captura_{0}.png".format(cont), frame2)
            cont += 1

    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("DETENIENDO PROGRAMA")
        exit()
    except cv.error: # omite errores originados por OpenCV
        pass
