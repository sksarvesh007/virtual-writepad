import cv2 
import mediapipe as mp 
import time
import logging
logging.basicConfig()


class handDetector():
    def __init__(self , mode=False , maxHands=2  , complexity = 1, detectionCon=0.5 , trackCon=0.5):
        self.mode=mode
        self.maxHands=int(maxHands)  # Ensure maxHands is an integer
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        
        self.mpHands=mp.solutions.hands
        self.complexity = complexity
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)        
        self.mpdraw = mp.solutions.drawing_utils 
        self.tipids=[4,8,12,16,20]
    
    def findhands(self , img , draw=True):
        imgRGB=cv2.cvtColor(img , cv2.COLOR_BGR2RGB) #converting the image to rgb because it only detects rgb images
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img , handlms , self.mpHands.HAND_CONNECTIONS)
        return img
    
    
    
    def findPosition(self , img , handNo=0 ,draw=True ):
        self.lmlist=[]
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id , lm in enumerate(myhand.landmark):
                #print(id,lm)
                h , w , c = img.shape
                cx , cy = int(lm.x*w) , int(lm.y*h)
                self.lmlist.append([id , cx , cy])
                if draw:
                    cv2.circle(img , (cx,cy) , 5 , (0,0,255) , cv2.FILLED)
        return self.lmlist
    def fingersup(self):
        fingers = []
        if self.lmlist[self.tipids[0]][1]<self.lmlist[self.tipids[0]-1][1]:
            fingers.append(1)
        else :
            fingers.append(0)
        for id in range(1,5):
            if self.lmlist[self.tipids[id]][2]<self.lmlist[self.tipids[id]-2][2]:
                fingers.append(1)
            else :
                fingers.append(0)
        return fingers

def main():
    pTime=0
    cTime=0
    cap = cv2.VideoCapture(0)
    detector= handDetector()
    while True :
        success , img = cap.read()
        img = detector.findhands(img )
        lmlist = detector.findPosition(img)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
            
        cv2.putText(img , str(int(fps)) , (10,70) , cv2.FONT_HERSHEY_COMPLEX , 3 , (255,0,255) , 3 )
        cv2.imshow("IMAGE" , img)
        #this loop is for the fps 
        #if we press escape the loop exits and the camera closes
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()