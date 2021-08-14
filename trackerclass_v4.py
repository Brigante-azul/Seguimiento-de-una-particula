# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 19:38:04 2021

@author: Azul
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max


#defino una funcion para encontrar las corrdenadas que dan distancia mínima al objetivo
def max_cercano(maximos, maximo_viejo):
    a=[np.linalg.norm(maximos[i]-maximo_viejo) for i in range(len(maximos))]
    i=np.argmin(a)
    return maximos[i]  

class tracker:
    def __init__(self, path):
        # abre el video, muestra el primer frame, se elije una caja como template, se crea el template
        self.video = cv.VideoCapture(path)
        if (int(self.video.get(7)))>0:
            print('Se abrio el video')
            print("fps {}".format(self.video.get(5)), "frames {}".format(self.video.get(7)))
        else:
            print("no se encontró el video")    
            
    #SELECCIONAR CAJA A MANO
    def setTemplate(self, n0):
        # setea el templeate que se va a usar para trackear
        self.video.set(1,n0)
        ret, frame = self.video.read()
        if ret:
          frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
          #frame = cv.resize(frame,self.size)
          bbox = cv.selectROI(frame)#puntos seleccionados de la caja
          cv.waitKey(1)             #wait time in milliseconds
          cv.destroyAllWindows()
          (c,f,w,h) = [int(a) for a in bbox]  
          self.h, self.w, self.c, self.f = h, w, c, f
          #template = frame[f:f+h,c:c+w]                       #acá se genera el template
          centro=[int((c+c+w)/2), int((f+f+h)/2)]             #coordenada x e y donde empieza el template
          ancho_t=[ int(w/2), int(h/2)]                       #ancho del template x e y
        else:
          print('No se encontro el video')
          
        return centro, ancho_t
    
    
    #TEMPLATE Y REGIÓN QUE SE VA A OBSERVAR INICIALMENTE
    def inicio(path, centro, ancho_t, ancho_v, boolean, no): #ancho_t y  ancho_v es un vector que contiene el ancho en x e y del tiempo y la region qeu se va a observas, respectivamente.
        cap=cv.VideoCapture(path)
        try:
            cap.set(1, no)                                    #fijo el frame 0
            ret, frame=cap.read()       
            imag=frame.copy()
            #dibujo el rectangulo del template
            cv.rectangle(imag, (centro[0]-ancho_t[0], centro[1]-ancho_t[1]), (centro[0]+ancho_t[0], centro[1]+ancho_t[1]), [0,0,0], 2) 
            template=cv.cvtColor(frame[centro[1]-ancho_t[1]:centro[1]+ancho_t[1], centro[0]-ancho_t[0]:centro[0]+ancho_t[0]], cv.COLOR_BGR2GRAY)
            #dijujo el área donde se va a ver 
            cv.rectangle(imag, (centro[0]-ancho_v[0], centro[1]-ancho_v[1]), (centro[0]+ancho_v[0], centro[1]+ancho_v[1]), [0,0,255], 1)
            obs=cv.cvtColor(frame[centro[1]-ancho_v[1]:centro[1]+ancho_v[1], centro[0]-ancho_v[0]:centro[0]+ancho_v[0]], cv.COLOR_BGR2GRAY)
            if boolean==True:
                cv.imshow("trackeo", imag)            
            print("El template y area de observacion inicial fueron definidas")
            
        except:
            print("No se encontro el video")
            template, obs=0, 0
                    
        return template, obs
        
        
    #esta función hace el trackeo correlacionando las imagenes
    #las variables son el video, las condiciones iniciales (tamplete, region de observación y centro xy), tiempo entre cada frame y duracion
    def corr(path, template, obs, centro, tiempo, duracion):
        
        cap=cv.VideoCapture(path) #importo el video
        cap.set(1, duracion[0])   #lo seteo en el frame duracion[0]
        
        #CONDICIONES INICIALES 
        h_t, w_t=[int(i/2) for i in template.shape]  #ancho y altura del template       
        h_v, w_v=[int(i /2) for i in obs.shape]      # ancho y altura del área de observacion
        d_h, d_w=h_v-h_t, w_v-w_t                    #diferencias entre anchos del área de observacion y el template
        x,y=centro[0], centro[1]                     #ubicacion del maximo de correlacion en el frame inicial
        max_loc= [int(w_v/2), int(h_v/2)]            #ubicacion del maximo de correlacion en el área de observacion inicial
                
        n=0                   #empiezo un contador de frames salteados por no haber match
        c=0                   #empiezo un contador de frames     
        
        x_vec = []          
        y_vec = []
        
        A=obs.copy()        #copia del área de observacion
        
        while True:
            ret, frame = cap.read()    #cap.read(1)  returns a bool (True/False)
            c=c+1                      #conteo de frames
            if not ret:
                print("ret = False")
                break
        
            else:
                if c<=duracion[1]-duracion[0]:              
                    
                    img = frame.copy()                          #copia el frame 
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  #da la imagen gris
                    
                    #CORRELACIÓN
                    method = cv.TM_CCOEFF
                    result = cv.matchTemplate(A, template, method) #el método para matchear las imágenes es haciendo la correlación cruzada
                    
                    #TOMO MÁXIMOS CERCANOS DE CORRELACION
                    try:
                        maximos=peak_local_max(result, min_distance=5, threshold_rel=0.8)
                        max_loc=max_cercano(maximos, max_loc)                    
                        
                    except:
                        n=n+1
                        print("se salteo el frame %s"%c)
                        if n>5:
                            print("se detuvo el traqueo en el frame %s"%c) #VEO EL FRAME DONDE FALLA
                            break   
                                            
                    #REDEFINO EL AREA DE OBSERVACIÓN Y CREO TEMPLATE NUEVO 
                    #template 
                    upper_left = (max_loc[1]+x-w_v, max_loc[0]+y-h_v) 
                    bottom_right = (upper_left[0]+2*w_t, upper_left[1]+2*h_t)                                                             
                    #template = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]    #defino un nuevo template                     
                    #observacion
                    try:
                        upper_left_b=(upper_left[0]-d_w, upper_left[1]-d_h) 
                        bottom_right_b = (upper_left_b[0]+2*w_v, upper_left_b[1]+2*h_v)
                        obs= gray[upper_left_b[1]:bottom_right_b[1], upper_left_b[0]:bottom_right_b[0]]                                
                        A=obs.copy()
                    except:
                        print("Area de observacion fuera de limite")
                        print("se detuvo el traqueo en el frame %s"%c) #VEO EL FRAME DONDE FALLA                           
                        break 
                    
                    #COORDENADAS X E Y                
                    x, y =int(upper_left[0])+w_t, int(upper_left[1])+h_t
                    x_vec.append(x)   #posición x 
                    y_vec.append(y)   #posición en y
                    
                    #GRAFICO EL TEMPLATE Y AREA DE OBSERVACIÓN
                    cv.rectangle(img, upper_left, bottom_right, [0,0,0], 2) #Dubuja un rectángulo (imagen, comienzo (x,y), fin(x, y), color, ancho)
                    cv.rectangle(img, upper_left_b, bottom_right_b, [0,0,255], 1)
                    
                    cv.imshow("trackeo", img)

                    if cv.waitKey(tiempo) == ord("q"): #DECIMAL VALUE of q is 113.
                        break                    
                else:
                    break
                                    
        return x_vec, y_vec
                
    #esta función hace el trackeo usando las diferencias cuadraticas de las imagenes (esto usa el trakcer)           
    def diff(path, template, obs, centro, tiempo, duracion):
        
        cap=cv.VideoCapture(path)
        
        #CONDICIONES INICIALES 
        h_t, w_t=[int(i/2) for i in template.shape]  #ancho y altura del template       
        h_v, w_v=[int(i /2) for i in obs.shape]      #ancho y altura del área de observacion
        d_h, d_w=h_v-h_t, w_v-w_t                    #diferencia entre ancho de template y area de observacion
        x,y=centro[0], centro[1]                     #objeto de observacion
        max_loc= [int(w_v/2), int(h_v/2)]            
                
        n=0                   #EMPIEZO UN CONTADOR DE FRAMES SALTEADOS POR NO HABER MATCH
        c=0                   #EMPIEZO UN CONTADOR DE FRAMES                                  
        
        x_vec = []
        y_vec = []
        
        A=obs.copy()        
        
        while True:
            ret, frame = cap.read()    #cap.read(1)  returns a bool (True/False)
            c=c+1
            if not ret:
                print("ret = False")
                break
        
            else:
                if c<=duracion[1]-duracion[0]:
                    
                    img = frame.copy()                          #clon the frame
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  #da la imagen gris
                    
                    #DIFERENCIAS CUADRATICAS
                    method = cv.TM_SQDIFF_NORMED 
                    result = 1-cv.matchTemplate(A, template, method)#el método para matchear las imágenes es haciendo las diferencias cuadraticas
                    
                    #TOMO MÁXIMOS CERCANOS DEL RESULTADO
                    try:
                        maximos=peak_local_max(result, min_distance=5, threshold_rel=0.8)
                        max_loc=max_cercano(maximos, max_loc)                    
                        
                    except:
                        n=n+1
                        print("se salteo %s frame"%n)
                        if n>5:
                            print("se detuvo el traqueo en el frame %s"%c) #VEO EL FRAME DONDE FALLA
                            break   
                                            
                    #REDEFINO EL AREA DE OBSERVACIÓN Y CREO TEMPLATE NUEVO 
                    #template
                    upper_left = (max_loc[1]+x-w_v, max_loc[0]+y-h_v) 
                    bottom_right = (upper_left[0]+2*w_t, upper_left[1]+2*h_t)                      
                    #template = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]    #defino un nuevo template 
                    #observacion
                    try:
                        upper_left_b=(upper_left[0]-d_w, upper_left[1]-d_h) 
                        bottom_right_b = (upper_left_b[0]+2*w_v, upper_left_b[1]+2*h_v)
                        obs= gray[upper_left_b[1]:bottom_right_b[1], upper_left_b[0]:bottom_right_b[0]]                                
                        A=obs.copy()
                    except:
                        print("Area de observacion fuera de limite")
                        print("se detuvo el traqueo en el frame %s"%c) #VEO EL FRAME DONDE FALLA                           
                        break 
                    
                    #COORDENADAS X E Y                
                    x, y =int(upper_left[0])+w_t, int(upper_left[1])+h_t
                    x_vec.append(x)   #posición x 
                    y_vec.append(y)   #posición en y
                    
                    #GRAFICO EL TEMPLATE Y AREA DE OBSERVACIÓN
                    cv.rectangle(img, upper_left, bottom_right, [0,0,0], 2) #draw a rectangle. (imagen, comienzo (x,y), fin(x, y), color, ancho)
                    cv.rectangle(img, upper_left_b, bottom_right_b, [0,0,255], 1)
                    
                    cv.imshow("trackeo", img)

                    if cv.waitKey(tiempo) == ord("q"): #DECIMAL VALUE of q is 113.
                        break                    
                else:
                    break
                       
        return x_vec, y_vec               

    def fps(path):
        cap= cv.VideoCapture(path)
        return cap.get(5)
       
        
        
        
        