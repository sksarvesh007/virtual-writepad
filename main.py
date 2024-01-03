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

brushthickness = 15
xp , yp = 0 , 0

detector = htm.handDetector(detectionCon=0.85)
drawcolor=(0,0,0)

imgcanvas = np.zeros((480,640,3),np.uint8)

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
        if fingers[1] & fingers[2]:
            xp , yp = 0 , 0
            if y1<107:
                if 0<x1<250:
                    header=overlaylist[0]
                    #put drawcolors value as red
                    drawcolor=(0,0,255)
                elif 250<x1<380:
                    header = overlaylist[1]
                    drawcolor=(237,235,0)
                elif 380<x1<510:
                    header = overlaylist[2]
                    drawcolor=(0,255,0)
                elif 510<x1<640:
                    header = overlaylist[3]
                    drawcolor=(0,0,0)
            #print("Selection Mode")
            cv2.rectangle(img , (x1,y1) , (x2,y2) , drawcolor , cv2.FILLED)
        if fingers[1] & fingers[2]==False:
            cv2.circle(img , (x1,y1) , brushthickness , drawcolor , cv2.FILLED)
            if xp==0 and yp==0:
                xp , yp = x1 , y1
            if drawcolor==(0,0,0):
                cv2.line(img , (xp,yp) , (x1,y1) , drawcolor , 35)
                cv2.line(imgcanvas , (xp,yp) , (x1,y1) , drawcolor , 35)
            else :
                cv2.line(img , (xp,yp) , (x1,y1) , drawcolor , 7)
                cv2.line(imgcanvas , (xp,yp) , (x1,y1) , drawcolor , 7)
            xp , yp = x1 , y1
            #print("Drawing Mode")
    
    imggray = cv2.cvtColor(imgcanvas , cv2.COLOR_BGR2GRAY)
    _ , imginv = cv2.threshold(imggray , 50 , 255 , cv2.THRESH_BINARY_INV)
    imginv = cv2.cvtColor(imginv , cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img , imginv)
    img = cv2.bitwise_or(img , imgcanvas)
    
    
    img[0:107 , 0:640] = header
    cv2.imshow("Image" , img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break