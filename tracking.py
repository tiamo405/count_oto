import cv2
import math
import numpy as np
import torch
import os

max_distance = 50
input_h = 360
input_w = 460
laser_line = input_h - 50
video_path='data\\test\\test.mp4'
imgs = ['camera\\1.jpg']
model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')  # local repo

#print(len(model(imgs).xyxy[0]))
#model(imgs).save()
def dem_car(res) :
    dem=0
    for i in range(len(res.xyxy[0])) :
        if res.xyxy[0][i][5] == 2 or res.xyxy[0][i][5] == 7 :
            dem+=1
    return dem
vid = cv2.VideoCapture(video_path)

# Khoi tao tham so
car_number = 0
i=0

# folder save img 
if not os.path.exists('camera/') : 
    os.mkdir('camera/')

while vid.isOpened():

    laser_line_color = (0, 0, 255)
    boxes = []

    # Doc anh tu video
    _, frame = vid.read()
    if frame is None:
        break

    # Resize nho lai
    frame = cv2.resize(frame, (input_w, input_h))
    
    # luu farme
    i+=1
    cv2.imwrite('camera/'+str(i)+'.jpg',frame)
    path_img=['camera\\'+str(i)+'.jpg']
    
    res=model(path_img)
    res.save()
    # dem so xe
    car_number = dem_car(res)
    print(path_img)
    print(car_number)
    # Hien thi so xe

    
    cv2.putText(frame, "Car number: " + str(car_number), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255 , 0), 2)
    cv2.putText(frame, "Press Esc to quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Draw laser line
    # cv2.line(frame, (0, laser_line), (input_w, laser_line), laser_line_color, 2)
    # cv2.putText(frame, "Laser line", (10, laser_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, laser_line_color, 2)

    # Frame
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

vid.release()
cv2.destroyAllWindows
