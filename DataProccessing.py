import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)

DATA_DIR = './data'
folders = ['0', '1', '2']

for folder in folders:
    class_dir=os.path.join(DATA_DIR,folder)
    if not os.path.isdir(class_dir):
        continue
    img_path =os.listdir(class_dir)[0]
    img_file=os.path.join(class_dir,img_path)
    img = cv2.imread(img_file)
    if img is None:
        print(f"Error: Couldn't read image {img_file}")
        continue
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #Hand landmarks detection
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            #showing image with landmarks
            plt.figure()
            plt.imshow(img_rgb)
            plt.title(f'Hand landmarks for sign {folder}')
            plt.axis('off')
            plt.show()
            plt.close()
hands.close()

