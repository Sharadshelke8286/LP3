
# %% Cell 1: Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


# %% Cell 2: Create Bank Customer Dataset

data = {
    'CreditScore': [619, 608, 502, 699, 850, 645, 822, 376, 501, 684,
                    528, 497, 476, 549, 635, 616, 653, 587, 726, 732],
    'Age': [42, 41, 42, 39, 43, 44, 50, 29, 44, 27,
            31, 24, 34, 25, 35, 45, 58, 24, 35, 41],
    'Tenure': [2, 1, 8, 1, 2, 8, 7, 4, 4, 2,
               6, 3, 10, 5, 3, 4, 1, 9, 5, 8],
    'Balance': [0, 83807, 159661, 0, 125510, 113755, 0, 115046,
                142051, 134603, 102016, 0, 85679, 120000,
                145000, 90000, 130000, 0, 110000, 95000],
    'NumOfProducts': [1, 1, 3, 2, 1, 2, 2, 4, 2, 1,
                      2, 2, 1, 1, 2, 2, 1, 2, 2, 1],
    'HasCrCard': [1, 0, 1, 0, 1, 1, 1, 1, 0, 1,
                  0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    'IsActiveMember': [1, 0, 0, 0, 1, 1, 1, 0, 1, 1,
                       0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
    'Exited': [1, 0, 1, 0, 0, 1, 0, 1, 0, 0,
               1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

print("Bank Customer Dataset:")
print(df.head())


# %% Cell 3: Prepare Input and Output

X = df.drop('Exited', axis=1)
y = df['Exited']

print("\nInput Features:")
print(X.head())

print("\nTarget Values:")
print(y.head())


# %% Cell 4: Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# %% Cell 5: Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %% Cell 6: Build Neural Network Model

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# %% Cell 7: Train Neural Network

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=4,
    validation_split=0.2,
    verbose=1
)


# %% Cell 8: Evaluate Model

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")


# %% Cell 9: Predict Customer Churn

y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred.flatten())


# %% Cell 10: Classification Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# %% Cell 11: Plot Accuracy and Loss

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# %% Cell 12: Predict for a New Customer

new_customer = np.array([[650, 40, 5, 100000, 2, 1, 1]])

new_customer = scaler.transform(new_customer)

prediction = model.predict(new_customer)

print("\nNew Customer Prediction:")

if prediction[0][0] > 0.5:
    print("Customer will leave the bank (Churn).")
else:
    print("Customer will stay with the bank (Not Churn).")
