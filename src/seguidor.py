#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Autores:
# Daniela Avila Luna
# Roberto David Manzo Gonzales

# Importando las librerias
import cv2 as cv
import cv_bridge
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Pose2D
from std_msgs.msg import Float32, Int32
import numpy as np

# Inicializando el nodo
rospy.init_node('seguidorLinea', anonymous=False)

# Publishers
robot_cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

# Variables globales
rate = rospy.Rate(10)
vel_robot = Twist()
robot_pos = Pose2D()
bridge = cv_bridge.CvBridge()
frame = th = img = None
contornos = []
M = None
# Valores geometricos 
wl = wr = signal_cap = 0.0
radio_llanta = 0.05
dist_llantas = 0.19

# Posiciones iniciales 
t0 = rospy.get_time()
x = y = th = 0.0
termina = False
avanzar = True
fin_linea = False

# --- FUNCIONES DE CALLBACK --- #
def end_callback():
    rospy.loginfo("Shutting down")
    vel_robot.linear.x = 0.0
    vel_robot.angular.z = 0.0
    robot_cmd_pub.publish(vel_robot)

def left_wheel_callback(msg):
    global wl
    wl = msg.data

def right_wheel_callback(msg):
    global wr
    wr = msg.data

def signal_callback(msg):
    global signal_cap, avanzar
    signal_cap = msg.data

def image_callback(msg):
    """
    Funcion que procesa la imagen para generar un rectangulo de la linea
    """
    global frame, th, img, contornos, M, img_gr
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    # Tomando solo una seccion de la imagen
    img = frame[7*len(frame)//8:, 2*len(frame[0])//8:6*len(frame[0])//8]

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    negro_low = (86,0,0)
    negro_high = (180, 255, 125)
    th = cv.inRange(img_hsv, negro_low, negro_high)
    img_gr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    th = cv.blur(th, (5,5))
    th = cv.erode(th, (3,3), iterations=1)
    th = cv.dilate(th,(3,3), iterations=2)
    contornos, _ = cv.findContours(th,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
    try:
        cnt = contornos[0]
        M = cv.moments(cnt)
        cv.drawContours(img_gr, contornos, -1, (255, 0, 0), 3)
        
    except ValueError:
        pass
    
    except IndexError:
        pass

rospy.Subscriber("/wl", Float32, left_wheel_callback)
rospy.Subscriber("/wr", Float32, right_wheel_callback)
rospy.Subscriber("/video_source/raw", Image, image_callback)
rospy.Subscriber("/signal_class", Int32, signal_callback)

# -- FIN DE FUNCIONES CALLBACK --- #

def avanzar_lazo_abierto(velocidad= 0.0, distancia=0.0):
    """
    Funcion que movera al robot hacia adelante 
    cierta distancia cierta velocidad
    """
    global termina
    print('inicio avance recto')
    t0 = rospy.get_rostime().to_sec()
    vel_robot.linear.x = velocidad
    vel_robot.angular.z = 0.0
    
    while True:
        robot_cmd_pub.publish(vel_robot)
        t1 = rospy.get_rostime().to_sec()
        # Calculo de la distancia estimada con la diferencia de tiempo
        distancia_estimada = abs(velocidad * (t1 - t0))
        if distancia_estimada > distancia:
            vel_robot.angular.z = 0.0
            vel_robot.linear.x = 0.0
            robot_cmd_pub.publish(vel_robot)
            break
        rate.sleep()
    termina = True

def girar_lazo_cerrado(velocidad=0.0, angulo=0.0):
    
    t0 = rospy.get_rostime().to_sec()
    vel_robot.angular.z = velocidad
    vel_robot.linear.x = 0.0
    theta = 0.0
    
    while theta < angulo:
        t1 = rospy.get_rostime().to_sec()
        dt = t1 - t0
        robot_cmd_pub.publish(vel_robot)
        theta -= ((wr - wl)/dist_llantas) * radio_llanta * dt
        t0 = rospy.get_rostime().to_sec()
        rospy.loginfo("{0} {1} {2}".format(str(np.rad2deg(angulo)), str(np.rad2deg(theta)), 'Vuelta lazo cerrado'))
        rate.sleep()
    
    theta = 0.0
    vel_robot.angular.z = 0.0
    vel_robot.linear.x = 0.0
    robot_cmd_pub.publish(vel_robot)

def semaforo_signal():
    global avanzar
    semaf_rojo_low = (0, 35, 187)
    semaf_rojo_high = (81, 255, 255)
    semaf_verde_low = (43, 21, 90)
    semaf_verde_high = (93, 255, 255)
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_sr = cv.inRange(frame_hsv,semaf_rojo_low, semaf_rojo_high)
    frame_sr = cv.dilate(frame_sr, (3,3), iterations=6)
    frame_sr = cv.erode(frame_sr, (3,3), iterations=6)
    frame_sv = cv.inRange(frame_hsv,semaf_verde_low, semaf_verde_high)
    frame_sv = cv.dilate(frame_sr, (3,3), iterations=6)
    frame_sv = cv.erode(frame_sr, (3,3), iterations=6)     

    cntr, _ = cv.findContours(frame_sr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cntv, _ = cv.findContours(frame_sv, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(cntr) > len(cntv):
        avanzar = False
    else:
        avanzar = True
 
    rospy.loginfo("Semaforo: {0}".format(avanzar))
    

def main():
    """
    Funcion que realiza el seguimiento de una linea
    """
    global t0, th, img_gr, frame, img, cont, termina, fin_linea
    e = 0
    # diferencial de tiempo
    t1 = rospy.get_time()
    dt = t1 - t0
    t0 = t1
    semaforo_signal()

    if len(contornos) > 0 and avanzar:
        if len(contornos) == 1:
            box = cv.minAreaRect(contornos[0])
            for i in range(len(contornos)):
                box = cv.minAreaRect(contornos[i])
                (x_min, y_min), (w_min, h_min), ang = box
                caja = cv.boxPoints(box)
                (x_box, y_box) = caja[0]
            
        else:
            boxes = []
            off_centro = 0
            
        x, y, w, h = cv.boundingRect(contornos[0])
        
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        tht_e = np.arctan2(y-cy, x-cx//2)
        kv = 0.025
        kth = 0.0031

        h, w = th.shape

        e_x = len(img[0])//2 - cx
        e_y = h-cy 


        d_e = ( ((cx-w//2)**2) + ((h-cy)**2))**(0.5)
        p_v = 0.06
        p_w = kth * e_x
        #rospy.loginfo("\nlinear X: {0}\nAngular Z: {1}".format(p_v, p_w))

        
        if signal_cap == 14:
            p_v = 0.0
            p_w = 0.0
            print('-----------------STOP-----------------')
            termina = False
        elif signal_cap == 32:
            print('-----------------FIN DE LIMITES-----------------')
            p_v = 0.7
            termina = False

        # Saturacion de velocidades lineales y agulares
        if p_v > 0.27:
            p_v = 0.27
        elif p_v < -0.27:
            p_v = -0.27

        if p_w > 0.15:
            p_w = 0.15
        elif p_w < -0.15:
            p_w = -0.15
        vel_robot.linear.x = p_v
        vel_robot.angular.z = p_w

    else:
        if signal_cap == 35 and not(termina) and avanzar:
            print('-----------------AHEAD ONLY-----------------')
            avanzar_lazo_abierto(0.06, 0.50)
            termina = False
        elif signal_cap == 33 and not(termina) and avanzar:
            print('-----------------TURN RIGHT AHEAD-----------------')
            avanzar_lazo_abierto(0.06, 0.34)
            girar_lazo_cerrado(-0.06, 8*(np.pi/2)/9)
            avanzar_lazo_abierto(0.05, 0.15)
            print('finaliza vuelta a la derecha')
            termina = False

    robot_cmd_pub.publish(vel_robot)
    image_pub.publish(bridge.cv2_to_imgmsg(th))
    
if __name__ == "__main__":
    while not rospy.is_shutdown():
        try:
            rate.sleep()
            main()
        except rospy.exceptions.ROSInterruptException:
            vel_robot.linear.x = 0
            vel_robot.linear.y = 0
            vel_robot.angular.z = 0
            robot_cmd_pub.publish(vel_robot)
            exit()

    vel_robot.linear.x = 0
    vel_robot.linear.y = 0
    vel_robot.angular.z = 0
    robot_cmd_pub.publish(vel_robot)
