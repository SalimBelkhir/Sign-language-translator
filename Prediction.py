import pickle
import cv2
import mediapipe as mp
import numpy as np

with open('svm_model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']
scaler = model_dict['scaler']

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5 ,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)  


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


labels_dict = {0: 'A', 1: 'B', 2: 'L', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
               10: 'J', 11: 'K', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
               19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '1', 27: '2',
               28: '3', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9', 35: 'stop', 36: 'help', 37: 'please',38:'hi',39:'bye'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(frame)
    # Drawing the Facial Landmarks


    hand_results = hands.process(frame_rgb)
    mp_drawing.draw_landmarks(
        frame,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(
            color=(255, 0, 255),
            thickness=1,
            circle_radius=1
        ),
        mp_drawing.DrawingSpec(
            color=(0, 255, 255),
            thickness=1,
            circle_radius=1
        )
    )
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            mp_drawing.draw_landmarks(
                frame,  
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,  
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )


            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)

            min_x = min(x_)
            min_y = min(y_)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)


            if len(data_aux) == 42:
                data_aux = np.asarray(data_aux).reshape(1, -1)
                data_aux = scaler.transform(data_aux)
                prediction = model.predict(data_aux)
                predicted_character = labels_dict[int(prediction[0])]

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
