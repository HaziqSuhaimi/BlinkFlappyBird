import cv2
import dlib
import numpy as np

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def EAR(eye):
    denom = np.linalg.norm(eye[0]-eye[3])
    enum1 = np.linalg.norm(eye[1] - eye[5]) 
    enum2 = np.linalg.norm(eye[2] - eye[4]) 

    ear = (enum1 + enum2)/ (2.0 * denom)
    return ear

class Pillar:
    def __init__(self):
        pass
    def pillar(self,img,x,h):
        width = 60
        w = x+width
        h2 = h+200
        hLand = img.shape[0] - 30
        cv2.rectangle(img,(x,0),(w,h),(0,200,0),-1)
        cv2.rectangle(img,(x,h2),(w,hLand),(0,200,0),-1)
        self.x = x
        self.w = w
        self.h = h
        self.h2 = h2

def destroy(list,pillar):
    list.remove(pillar)

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
SCREEN_H, SCREEN_W, channels = frame.shape
BIRD_Y = int(SCREEN_H/2)
BIRD_X = SCREEN_W/4
GRAVITY = 5
RED = (0,0,255)
BIRD_SIZE = 15
LAND = SCREEN_H - 15
JUMP = 50
pil_x = SCREEN_W
pil_gap =150
num = 0

pillars = list()
for i in range(10):
    pillars.append(Pillar())


while True:

    key = cv2.waitKey(1)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects = detector(gray,0)

    pillars[num].pillar(frame,pil_x,pil_gap)

    cv2.line(frame,(0,LAND),(SCREEN_W,LAND),(0,255,0),30)
    cv2.circle(frame,(int(BIRD_X),BIRD_Y), BIRD_SIZE, RED, -1)

    for rect in rects:
        (bX, bY, bW, bH) = rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)

        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        eyes = shape[36:48] #eyes landmarks
        l_eye = eyes[:6]
        r_eye = eyes[6:]
        leftEAR = EAR(l_eye)
        rightEAR = EAR(r_eye)

        ear = (leftEAR + rightEAR) / 2.0
        # print(ear)
        if ear < 0.15:
            print("blinked!", ear)
            BIRD_Y -= JUMP
            cv2.circle(frame,(int(BIRD_X),BIRD_Y), BIRD_SIZE, RED, -1)

        for (i, (x, y)) in enumerate(eyes):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            # cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        BIRD_Y += GRAVITY
        pil_x -= GRAVITY

        if BIRD_Y >= LAND:
            print("GAME OVER")
            key = ord("q")


        while pillars[num].x < BIRD_X and pillars[num].w > BIRD_X:
            if not (BIRD_Y > pillars[num].h and BIRD_Y < pillars[num].h2):
                print("GAME OVER")
                key = ord("q")
                break
            else:
                break

        if pillars[num].x is 0:
            destroy(pillars,pillars[num])
            num +=1
            pil_x = SCREEN_W
            pil_gap =np.random.randint(50, high=250)
            pillars[num].pillar(frame,pil_x ,pil_gap)

    cv2.imshow("Frame", frame)
    if key == ord("q"):
        break

cv2.destroyAllWindows()


