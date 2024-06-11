import os
import pickle
import mediapipe as mp
import cv2
from concurrent.futures import ThreadPoolExecutor

# Initialize Mediapipe Hands and Face Mesh
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2)

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
    face_results = face_mesh.process(img_rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)

    # Ensure the length of data_aux meets the expected format: 42 hand + 468*2 face = 42 + 936 = 978
    if len(data_aux) == 42 +936:
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

# To see the content of pickle file
'''
with open('data.pickle', 'rb') as f:
    dataset = pickle.load(f)
print(f"Number of samples: {len(dataset['data'])}")
print(f"Labels: {set(dataset['labels'])}")
print(f"First sample data: {dataset['data'][0]}")
print(f"First sample label: {dataset['labels'][0]}")
'''
