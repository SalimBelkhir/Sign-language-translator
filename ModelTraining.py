import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Filter out samples that do not have 42 features
filtered_data = [sample for sample in data_dict['data'] if len(sample) == 42]
filtered_labels = [data_dict['labels'][i] for i, sample in enumerate(data_dict['data']) if len(sample) == 42]


data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

data = data[:, :42]

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = SVC()
model.fit(x_train, y_train)


y_predict = model.predict(x_test)


score = accuracy_score(y_test, y_predict)
print('{}% of samples were classified correctly!'.format(score * 100))

print("Classification Report:\n", classification_report(y_test, y_predict))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))

with open('svm_model.p', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)
