# Puzzlebot autónomo

Este repositorio contiene los códigos utilizados para la implementación de un robot capaz de identificar diferentes señales de tránsito y semáforos mientras se sigue una trayectoria.

Los autores de este reporitorio son:

- Daniela Avila Luna [@DannyAvilaL](https://github.com/DannyAvilaL)
- Roberto David Manzo González [@robertomanzo2203](https://github.com/robertomanzo2203)

Este repositorio fue implementado en ROS utilizando la versión Melodic para Ubuntu 18.04 en una Jetson Nano de Nvidia. Está listo para ser descargado y ser compilado con las instrucción ```$ catkin_make ``` dentro de un workspace de ROS previamente definido.

Dentro de la carpeta de ```src\``` se encuentran los 2 nodos principales para el funcionamiento
- **seguidor.py:** Realiza el seguimiento de una línea con control proporcional e identifica las señales del semáforo.
- **clasificador.py:** Clasifica las señales de tránsito identificadas por área de aproximación de polígonos (STOP).

Adicionalmente, vienen 3 archivos de python:

- **captura_camara.py:** Para poder guardar las imágenes que se transmiten a través del tópico que se utiliza para enviar la imagen de la cámara.
- **jetson_camera_slider.py:** Codigo **originalmente** realiado por OpenCV para la webcam integrada, pero que fue modificado para realizar la misma calibración de parámetros HSV utilizando los tópicos de ROS.
- **train.py:** Para realizar el entrenamiento del modelo de CNN utilizado en la clasificación de señales de tránsito.
