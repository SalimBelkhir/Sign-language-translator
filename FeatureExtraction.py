import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True ,min_detection_confidence=0.3)
DATA_DIR='./data'
data=[]
labels=[]

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR,dir_)
    if not os.path.isdir(class_dir):
        continue
    for img_path in os.listdir(class_dir):
        data_aux=[]
        x_ = []
        y_ = []
        img_file = os.path.join(class_dir,img_path)
        img = cv2.imread(img_file)
        if img is None :
            print(f"Error : couldn't read image file{img_file}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks :
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x= hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x-min(x_))
                    data_aux.append(y-min(y_))
            data.append(data_aux)
            labels.append(int(dir_))
with open('data.pickle','wb') as f :
    pickle.dump({'data':data,'labels':labels},f)

with open('data.pickle','rb') as f :
    dataset = pickle.load(f)
print(f"number of samples :{len(dataset['data'])}")
print(f"Labels:{set(dataset['labels'])}")
print(f"First sample data: {dataset['data'][0]}")
print(f"First sample label: {dataset['labels'][0]}")