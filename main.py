import cv2
import mediapipe as mp
import os 
import time 
import numpy as np
import handtrackingmodule as htm


folderpath = "header"
mylist = os.listdir(folderpath)

overlaylist = []
for impath in mylist:
    img = cv2.imread(str(f'{folderpath}/{impath}'))
    overlaylist.append(img)
header = overlaylist[0]

detector = htm.handDetector(detectionCon=0.85)
drawcolor=(255,0,0)
cap = cv2.VideoCapture(0)
while True:
    success , img = cap.read()
    img = cv2.flip(img , 1)
    
    img = detector.findhands(img )
    lmlist = detector.findPosition(img)
    
    if len(detector.lmlist)!=0:
        #print(lmlist[4])
        x1 , y1 = detector.lmlist[8][1:] # x1 , y1 are the coordinates of the tip of the index finger
        x2 , y2 = detector.lmlist[12][1:] # x2 , y2 are the coordinates of the tip of the middle finger
        
        fingers = detector.fingersup()
        print(fingers)
        
        if fingers[1] & fingers[2]:
            cv2.rectangle(img , (x1,y1) , (x2,y2) , drawcolor , cv2.FILLED)
            if y1<107:
                if 0<x1<250:
                    header=overlaylist[0]
                elif 250<x1<380:
                    header = overlaylist[1]
                elif 380<x1<510:
                    header = overlaylist[2]
                elif 510<x1<640:
                    header = overlaylist[3]
            #print("Selection Mode")
            
        if fingers[1] & fingers[2]==False:
            header = overlaylist[0]
            #print("Drawing Mode")
    
    
    img[0:107 , 0:640] = header
    cv2.imshow("Image" , img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break