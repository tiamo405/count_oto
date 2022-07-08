import cv2
import math
import numpy as np
import torch
import os

video_path='data\\test\\test.mp4'
imgs = ['camera\\1.jpg']
# load model yolov5
model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')  # local repo

#print(len(model(imgs).xyxy[0]))
#model(imgs).save()
def dem_car(res) :
    dem=0
    for i in range(len(res.xyxy[0])) :
        if res.xyxy[0][i][5] == 2 or res.xyxy[0][i][5] == 7 :
            dem+=1
    return dem
def point_center(res,i) :
  xmin,ymin,xmax,ymax=int(res.xyxy[0][i][0]),int(res.xyxy[0][i][1]),int(res.xyxy[0][i][2]),int(res.xyxy[0][i][3])
  xtt=(xmin+xmax)//2
  ytt=(ymin+ymax)//2
  return xtt,ytt

def draw_point_center(frame,res):
    for i in range(len(res.xyxy[0])):
        if res.xyxy[0][i][5] == 2 or res.xyxy[0][i][5] == 7 :
            center_x,center_y=point_center(res,i)
            cv2.circle(frame, (center_x,center_y), radius=5, color=(0, 0, 255), thickness=-1)

# xem video
video = cv2.VideoCapture(video_path)

# stream
# video = VideoStream(src=1).start()

# Khoi tao tham so
car_number = 0
i=0
max_distance = 50
input_h = 360
input_w = 460
laser_line = input_h - 50
laser_line_color = (0, 0, 255)
boxes = []

# folder save img 
if not os.path.exists('camera/') : 
    os.mkdir('camera/')

while video.isOpened():

    # Doc anh tu video
    _, frame = video.read()
    if frame is None:
        break

    # Resize nho lai
    frame = cv2.resize(frame, (input_w, input_h))
    
    # luu farme
    #i+=1
    #cv2.imwrite('camera/'+str(i)+'.jpg',frame)
    #path_img=['camera\\'+str(i)+'.jpg']
    #print(path_img)

    # predict
    #res=model(path_img)
    res=model(frame)
    #res.save()

    # dem so xe
    car_number = dem_car(res)
    print(car_number)

    # draw center cho moi car
    draw_point_center(frame,res)

    # Hien thi so xe
    cv2.putText(frame, "Car number: " + str(car_number), 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255 , 0), 2
                )
    cv2.putText(frame, "Press Esc to quit", 
                (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 0), 2)

    # Draw laser line
    # cv2.line(frame, (0, laser_line), (input_w, laser_line), laser_line_color, 2)
    # cv2.putText(frame, "Laser line", (10, laser_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, laser_line_color, 2)

    # Frame
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

video.release()
cv2.destroyAllWindows
