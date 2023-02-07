import cv2 as cv 
import mediapipe as mp 
import numpy as np 
import math 
import os
class handDetector:
    def __init__(self):
        self.mpDraw=mp.solutions.drawing_utils
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands()
    def findHands(self,image):
        imageRGB=cv.cvtColor(image, cv.COLOR_BGR2RGB)
        ih,iw,c=image.shape
        result=self.hands.process(imageRGB)
        multiHand=result.multi_hand_landmarks
        self.landmarkList=[]
        if multiHand:
            for handID,singleHandLandmarks in enumerate(multiHand):
                self.mpDraw.draw_landmarks(image, singleHandLandmarks,self.mpHands.HAND_CONNECTIONS,
                connection_drawing_spec=self.mpDraw.DrawingSpec((0,255,0)))
                for id,landmark in enumerate(singleHandLandmarks.landmark):
                    x,y=int(iw*landmark.x),int(ih*landmark.y)
                    self.landmarkList.append([id,x,y])
        return self.landmarkList
    def fingureCount(self):
        if self.landmarkList:
            ref_point_x=self.landmarkList[0][1]
            ref_point_y=self.landmarkList[0][2]
            points=[[4,3],[8,6],[12,10],[16,14],[20,18]]
            count=[]
            if ref_point_y>self.landmarkList[points[1][0]][2] and ref_point_x>self.landmarkList[points[1][0]][1]:
                for no in points:
                    if no==points[0]:
                        ux,uy=self.landmarkList[no[0]][1],self.landmarkList[no[0]][2]
                        dx,dy=self.landmarkList[no[1]][1],self.landmarkList[no[1]][2]
                        if dx>ux :
                            count.append(1)
                        else:
                            count.append(0)
                    else:
                        ux,uy=self.landmarkList[no[0]][1],self.landmarkList[no[0]][2]
                        dx,dy=self.landmarkList[no[1]][1],self.landmarkList[no[1]][2]
                        if dy>uy:
                            count.append(1)
                        else:
                            count.append(0)
            elif ref_point_x<self.landmarkList[points[1][0]][1] and ref_point_y>self.landmarkList[points[4][0]][2]:
                for no in points:
                    if no==points[0]:
                        ux,uy=self.landmarkList[no[0]][1],self.landmarkList[no[0]][2]
                        dx,dy=self.landmarkList[no[1]][1],self.landmarkList[no[1]][2]
                        if dy>uy :
                            count.append(1)
                        else:
                            count.append(0)
                    else:
                        ux,uy=self.landmarkList[no[0]][1],self.landmarkList[no[0]][2]
                        dx,dy=self.landmarkList[no[1]][1],self.landmarkList[no[1]][2]
                        if dy>uy:
                            count.append(1)
                        else:
                            count.append(0)
            return count.count(1)
            
def main():
    cap=cv.VideoCapture(0,cv.CAP_DSHOW)
    hand=handDetector()
    path="Resource"
    imgPathList=os.listdir(path)
    imgList=[]
    for imgPath in imgPathList:
        imgFingure=cv.imread(f"{path}/{imgPath}")
        imgFingure=cv.resize(imgFingure, (0,0),None,0.15,0.15)
        imgList.append(imgFingure)
    backgound=cv.imread("background.jpg")
    while True:
        ret,img=cap.read()
        ih,iw,c=img.shape
        img=cv.flip(img, 1)
        count=[]
        lst=hand.findHands(img)
        countedNumber=hand.fingureCount()
        backgound[100:100+ih,150:150+iw]=img
        if countedNumber!=None:
            for i in range(0,6):
                    if i==countedNumber:
                        h,w,c1=imgList[i].shape
                        backgound[210:210+h,820:820+w]=imgList[i]
                        print(h,w)
            cv.putText(backgound, "Count:", (10,585), cv.FONT_HERSHEY_COMPLEX, 0.9, (255,255,255),2)
            cv.rectangle(backgound, (20,690), (138,615), (152,182,9),cv.FILLED)
            cv.putText(backgound, f"{countedNumber}", (25,680), cv.FONT_HERSHEY_COMPLEX, 3, (255,255,255),3)
        else:
            backgound[210:210+225,820:820+180]=(255,255,255)
            cv.rectangle(backgound, (20,690), (138,615), (152,182,9),cv.FILLED)
        cv.imshow("Fingure Counter", backgound)
        if cv.waitKey(1)==ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
if __name__=="__main__":
    main()