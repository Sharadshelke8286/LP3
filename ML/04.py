
# Practical 9: K-Nearest Neighbors on Diabetes Dataset

# Step 1: Import required libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

# Step 2: Load the diabetes dataset
df = pd.read_csv("diabetes.csv")

print("First 5 rows of dataset:")
print(df.head())

# Step 3: Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Step 4: Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Standardize the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

# Step 7: Predict test data
y_pred = knn.predict(X_test)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

# Step 8: Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Step 9: Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)

error_rate = 1 - accuracy

precision = precision_score(y_test, y_pred, zero_division=0)

recall = recall_score(y_test, y_pred, zero_division=0)

# Step 10: Display results
print("\nPerformance Evaluation:")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Error Rate:", round(error_rate * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))

# Step 11: Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
