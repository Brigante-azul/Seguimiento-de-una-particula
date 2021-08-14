# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:01:08 2021

@author: Azul
"""

from trackerclass_v5 import tracker
import matplotlib.pyplot as plt
import numpy as np
import time
#%%
file="Pollen Grains in Water"
video=file+".mp4"
tracker(video)
fps=tracker.fps(video)
#%%
#VEO LOS TIEMPOS EN FRAMES
s=10         #segundos
print(fps*s) #cantidad de frames por s
#%%
#DURACION DEL VIDEO
n0, l = 50, 250       #l es la cantidad de frames a analizar
duracion=[n0, n0+l]   #frame de inicio y fnal
#%%
#SELECCIONO EL TEMPLATE
centro, ancho_t=tracker.setTemplate(tracker(video), n0)
#MAYUS+ENTER para cerrar el cuadro de seleccion
#se puede definir el centro y ancho del template a mano

#GRAFICO IMAGEN INICIAL
#ancho x e y del área de observación
ancho_v=[40, 40]

#DEFINO TEMPLATE INICIAL (negro) Y DE OBSERVACION (rojo)
template_i, obs_i= tracker.inicio(video, centro, ancho_t, ancho_v, True, n0)
#%%

t0=time.time()
#METODO DE CORRELACION
#TRACKEO
tiempo_trackeo=1          #tiempo entre frame y frame del trackeo en milisegundos
x_corr, y_corr, imagenes=tracker.corr(video, template_i, obs_i, centro, tiempo_trackeo, duracion)     
t=time.time()-t0
print(t)

#GRAFICO TRAYECTORIA
plt.figure(1), plt.clf(), plt.grid(True)
plt.plot(x_corr, y_corr, ".-",color="limegreen",label="metodo de correlacion")
plt.legend()

#%%
#GUARDO LOS DATOS
n=1
nombre="f20_f120_%s"%n
np.savetxt("%s.txt"%nombre, np.transpose([x_corr, y_corr]), header="frames %s, centro=%s, duracion=%s, video=%s"%(len(x_corr), centro, duracion, file))


