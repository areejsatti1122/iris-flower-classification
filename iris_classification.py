# IRIS FLOWER CLASSIFICATION - MACHINE LEARNING BEGINNER PROJECT
# (Same code as the "Let's Crack It!" YouTube video)

# ---------------- IMPORTING LIBRARIES ----------------
# IMPORTING ALL THE ESSENTIAL LIBRARIES FOR OUR PROJECT
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------------- LOADING THE DATASET ----------------
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']
# Load the data
df = pd.read_csv('iris.data', names=columns)
print(df.head())

# ---------------- VISUALIZATION OF OUR DATASET ----------------
print(df.describe())

# Visualize the whole dataset
sns.pairplot(df, hue='Class_labels')
plt.show()

# ---------------- SEPARATING INPUT COLUMNS AND THE OUTPUT COLUMN ----------------
# Separate features and target
data = df.values

X = data[:, 0:4]
Y = data[:, 4]
print(Y)

# ---------------- SPLITTING THE DATA INTO TRAINING AND TESTING ----------------
# Split the data to train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# ---------------- MODEL 1: Support vector machine algorithm ----------------
# Support vector machine algorithm
from sklearn.svm import SVC
model_svc = SVC()
model_svc.fit(X_train, y_train)

prediction1 = model_svc.predict(X_test)
# Calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction1) * 100)

for i in range(len(prediction1)):
    print(y_test[i], prediction1[i])

# ---------------- MODEL 2: LOGISTIC REGRESSION ----------------
# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression(max_iter=200)
model_LR.fit(X_train, y_train)

prediction2 = model_LR.predict(X_test)
# Calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction2) * 100)

# ---------------- MODEL 3: Decision Tree Classifier ----------------
# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model_DTC = DecisionTreeClassifier()
model_DTC.fit(X_train, y_train)

prediction3 = model_DTC.predict(X_test)
# Calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction3) * 100)

# ---------------- CLASSIFICATION REPORT ----------------
# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction2))

# ---------------- PREDICTING NEW FLOWERS ----------------
X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])
# Prediction of the species from the input vector
prediction = model_svc.predict(X_new)
print("Prediction of Species: {}".format(prediction))
