#Alunos: Arthur B. Pinotti, Kaue Reblin, Luiz G. Klitzke.

import cv2 as cv2
from matplotlib import pyplot as plt

car_cascade = cv2.CascadeClassifier("cascades/cars.xml")

#https://www.geeksforgeeks.org/python-play-a-video-using-opencv/
card_video_cap = cv2.VideoCapture("/content/samples/cars.avi")
if (card_video_cap.isOpened() == False): 
    print("Erro ao abrir o arquivo de vídeo.") 

frame_idx = 1

while(card_video_cap.isOpened()): 
    ret, frame = card_video_cap.read() 

    if ret == False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Parâmetros: (image , scaleFactor , minNeighbors )
    #https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
    #minNeighbors como 3 foi o melhor equilibrio de qualidade e falsos positivos que eu encontrei.
    cars = car_cascade.detectMultiScale(gray, 1.03, 3)

    for (x, y ,w ,h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255 ,0), 2)

    plt.title("Frame: " + str(frame_idx) + " | Carros: " + str(len(cars)))
    plt.imshow(frame, 'gray')
    plt.show()

    frame_idx += 1

    if cv2.waitKey(1000) == ord('q'):
        break

card_video_cap.release() 
cv2.destroyAllWindows() 