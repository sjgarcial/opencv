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


# segundo ejercicio

import cv2
import numpy as np

# Ruta al archivo de Haar Cascade para detección de rostros
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Cargar el modelo Haar Cascade
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Inicializa la captura de vídeo (0 es para la cámara principal)
cap = cv2.VideoCapture(0)

# Comienza la detección
print("Presiona 'q' para salir.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break

    # Convierte el frame a escala de grises para mejorar el rendimiento
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostros
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar rectángulos alrededor de los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar el vídeo con las detecciones
    cv2.imshow('Detección de Rostros', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
