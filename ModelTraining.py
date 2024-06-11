import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


# Load the dataset
data_dict = pickle.load(open('./data.pickle','rb'))

# Ensure data and labels are numpy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Standardize the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize the model
model = SVC()
model.fit(x_train,y_train)

# Predict on the test data
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_test, y_predict)
print('{}% of samples were classified correctly!'.format(score * 100))

# Print classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_predict))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))

# Save the trained model
with open('svm_model.p', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)
