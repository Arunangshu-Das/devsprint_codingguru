import mediapipe as mp
import cv2

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

def calculate_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180/np.pi)
    if(angle>180):
        angle=360-angle
    return angle



cap=cv2.VideoCapture(0)
while cap.isOpened():
    _,frame=cap.read()
    cv2.imshow('Detection model',frame)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False




    #Pose detection


    if cv2.waitKey(10) & 0XFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()