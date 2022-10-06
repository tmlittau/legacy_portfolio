import numpy as np
import cv2
import os

person = raw_input("Please enter your Name: ")
os.system("mkdir dataset/" + person) 
dirname = "dataset/" + person + "/"

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while(True):
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if( not ret ):
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # Space pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        img_name = dirname + img_name
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
