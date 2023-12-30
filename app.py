import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.models import load_model

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')

f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape
    mid_y = height * 0.6  
    color = (255, 0, 0)
    thickness = 2
    cv2.line(frame, (0, int(mid_y)), (width, int(mid_y)), color, thickness)
    class_names = ["Hand 1", "Hand 2"]
    class_results = ['No Hand Detected', 'No Hand Detected']

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        for hand_idx, handslms in enumerate(result.multi_hand_landmarks):
            landmarks = []
            below_line = False 

            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

                if lmy >= mid_y: 
                    below_line = True

            if not below_line:
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                class_results[hand_idx] = classNames[classID]

    for hand_idx, class_result in enumerate(class_results):
        cv2.putText(frame, f'{class_names[hand_idx]}: {class_result}', (10, 50 + hand_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



