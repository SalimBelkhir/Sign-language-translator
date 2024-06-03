import os
import cv2


def find_working_camera():
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
    return -1


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

existing_classes = [int(d) for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))]
existing_classes.sort()

dataset_size = 100

# Find a working camera index
camera_index = find_working_camera()
if camera_index == -1:
    print("Error: Could not find a working camera")
    exit()

cap = cv2.VideoCapture(0)
#you can put index instead of 0 in VideoCapture Method since I have only one camera
if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

for j in range(len(existing_classes) ,len(existing_classes)+2):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break

        cv2.putText(frame, 'Ready? Press "Q" !', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            continue

        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.imshow('frame', frame)
            cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
            counter += 1

        if cv2.waitKey(25) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
