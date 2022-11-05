import mediapipe as md
import cv2

cap=cv2.VideoCapture(0)

while cap.isOpened():
    _,frame=cap.read()
    cv2.imshow('Detection model',frame)

    #Pose detection
    

    if cv2.waitKey(10) & 0XFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()