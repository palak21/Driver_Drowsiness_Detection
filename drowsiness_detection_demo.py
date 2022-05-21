import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time


mixer.init()
sound = mixer.Sound('C:\\Users\\16692\\Desktop\\Driver Drowsi\\Drowsiness detection\\Drowsiness detection\\alarm.wav')


face = cv2.CascadeClassifier(r'C:\Users\16692\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(r'C:\Users\16692\anaconda3\Lib\site-packages\cv2\data\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(r'C:\Users\16692\anaconda3\Lib\site-packages\cv2\data\haarcascade_righteye_2splits.xml')

#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

lbl=['Close','Open']

model = load_model(r'C:\Users\16692\Desktop\Driver Drowsi\Drowsiness detection\Drowsiness detection\models\cnnCatnew.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)  #0 means read from local camera
#cap = cv2.imread(r"C:\Users\16692\Downloads\Drowsiness detection\Drowsiness detection\open.jpg")
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rclass=[99]
lclass=[99]
flag = True
while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(32,32))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2RGB)
        r_eye = cv2.resize(r_eye,(32,32))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(32,32,-1)

        r_eye = np.expand_dims(r_eye,axis=0)
#
        rpred = model.predict(r_eye)

        rclass=np.argmax(rpred,axis=1)

        if(rclass==1):
            lbl='Open' 
        if(rclass==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2RGB)  
        l_eye = cv2.resize(l_eye,(32,32))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(32,32,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict(l_eye)
        print('lpred', lpred)
        lclass=np.argmax(lpred,axis=1)
        print('lclass', lclass)
        if(lclass==1):
            lbl='Open'   
        if(lclass==0):
            lbl='Closed'
        break

    if(rclass==0 and lclass==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rclass==1 or lclass==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    flag = False
cap.release()
cv2.destroyAllWindows()
