import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


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
    mid_y = height // 2
    color = (255, 0, 0) 
    thickness = 1
    cv2.line(frame, (0, mid_y), (width, mid_y), color, thickness)


    # Get hand landmark prediction
    result = hands.process(framergb)

    className = ''

    if result.multi_hand_landmarks:
        # Post-process the result for each hand.3
        for hand_idx, handslms in enumerate(result.multi_hand_landmarks):
            landmarks = []
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames for each hand
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]


            # Show the prediction on the frame for each hand
            cv2.putText(frame, f'Hand {hand_idx + 1}: {className}', (10, 50 + hand_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




