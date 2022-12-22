import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks) #to detected hand

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Get the information with the hand id number and landmarks information
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                height, width, chanel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)  # position
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 10, (150, 50, 255), cv2.FILLED)
                if id == 8:
                    cv2.circle(img, (cx, cy), 10, (10, 50, 25), cv2.FILLED)
                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (250, 250, 255), cv2.FILLED)
                if id == 12:
                    cv2.circle(img, (cx, cy), 10, (0, 250, 250), cv2.FILLED)
                if id == 16:
                    cv2.circle(img, (cx, cy), 10, (50, 250, 25), cv2.FILLED)
                if id == 20:
                    cv2.circle(img, (cx, cy), 10, (150, 100, 100), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 200),
                3)  # (10, 7-) is position value, Font, Skills, color, thickness

    cv2.imshow("Image", img)
    cv2.waitKey(1)
