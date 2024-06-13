import os
import cv2
import mediapipe as mp
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

def process_image(class_dir, img_path):
    data_aux = []
    x_ = []
    y_ = []
    img_file = os.path.join(class_dir, img_path)
    img = cv2.imread(img_file)
    if img is None:
        print(f"Error: couldn't read image file {img_file}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(img_rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)

        min_x = min(x_)
        min_y = min(y_)

        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

        label = int(os.path.basename(class_dir))
        return data_aux, label

    return None

with ThreadPoolExecutor() as executor:
    futures = []
    for dir_ in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(class_dir):
            continue
        for img_path in os.listdir(class_dir):
            futures.append(executor.submit(process_image, class_dir, img_path))

    for future in futures:
        result = future.result()
        if result is not None:
            data_aux, label = result
            data.append(data_aux)
            labels.append(label)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
