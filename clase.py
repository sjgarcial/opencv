import cv2
import numpy as np

#cargamos imagen
image = cv2.imread('static/imagen.jpg')
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#obtener dimensiones de la imagen
hight, width = image.shape[:2]
center = (width/2, hight/2)
#rotar la imagen
angulo = 75
matrix = cv2.getRotationMatrix2D(center, angulo, 1.0)
rotated = cv2.warpAffine(image, matrix, (width, hight))
cv2.imshow('Image', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Definir la matriz de traslación
tx, ty = 100, 50 
M = np.float32([[1, 0, tx], [0, 1, ty]])
#Aplicar la matriz de traslación a la imagen
translated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
# mostrar la imagen trasladada
cv2.imshow('Image', translated)
cv2.waitKey(0)
cv2.destroyAllWindows()

#definir la nuevas dimensiones de la imagen
new_width = 400
new_height = 300

#aplicar la escala de la imagen
scaled = cv2.resize(image, (new_width, new_height))

#mostrar la imagen escalada
cv2.imshow('Image', scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()

#recorte
#definir las coordenadas del area de interes ROI
x, y, w, h = 100, 50, 300, 200
#recortar la imagen
cropped = image[y:y+h, x:x+w]
#mostrar la imagen recortada
cv2.imshow('Image', cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

#suavizar la imagen
#aplicar el filtro gaussiano para suavizar la imagen
smoothed = cv2.GaussianBlur(image, (5, 5), 0)
#mostrar la imagen suavizada
cv2.imshow('Image', smoothed)
cv2.waitKey(0)
cv2.destroyAllWindows()

#realce
#definir el kelner para el filtro de afilado
kernel = np.array([[-1, -1, -1], 
                   [-1, 9, -1], 
                   [-1, -1, -1]])
#aplicar el filtro de afilado
sharpened = cv2.filter2D(image, -1, kernel)
#mostrar la imagen realcada
cv2.imshow('Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()