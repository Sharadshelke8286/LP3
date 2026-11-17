# %% [markdown]
# PRACTICAL 7: EMAIL SPAM DETECTION USING KNN AND SVM
#
# Binary classification:
# 0 = Normal State (Not Spam)
# 1 = Abnormal State (Spam)
#
# Algorithms:
# 1. K-Nearest Neighbors (KNN)
# 2. Support Vector Machine (SVM)
#
# Run each cell separately in Jupyter Notebook using Shift + Enter.
# This file can also be run directly in VS Code.

# %%
# STEP 1: Import required libraries

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# %%
# STEP 2: Create a small sample email dataset
# Label 0 = Not Spam, Label 1 = Spam

emails = [
    # Normal / Not Spam emails
    "Hi, are we still meeting for lunch today?",
    "Please find the meeting notes attached.",
    "Your appointment is confirmed for tomorrow.",
    "Can you send me the project report?",
    "The class will start at nine in the morning.",
    "Thank you for your help with the assignment.",
    "I will call you when I reach home.",
    "The team meeting has been moved to Friday.",
    "Please review the updated document.",
    "Your order has been delivered successfully.",
    "Happy birthday, have a wonderful day.",
    "The lecture notes are available on the portal.",
    "Let me know when you are free.",
    "I have attached the invoice for your records.",
    "Your password was changed successfully.",
    "Please confirm your attendance for the event.",
    "The train is delayed by fifteen minutes.",
    "Can we discuss the project after class?",
    "Your library book is due next week.",
    "Thanks for registering for the workshop.",

    # Spam emails
    "Congratulations you have won a free cash prize claim now",
    "Click here to get a guaranteed loan immediately",
    "You are selected for a lottery winner reward",
    "Earn money fast from home without any experience",
    "Urgent claim your free gift card today",
    "Limited offer buy now and get huge discount",
    "You won a cash reward click this link",
    "Get cheap medicines online order now",
    "Exclusive deal claim your bonus prize now",
    "Make thousands of dollars in one week",
    "Your account won a free vacation claim now",
    "Act now to receive your guaranteed cash reward",
    "Congratulations claim your lottery prize today",
    "Get a free smartphone by clicking this link",
    "Lowest price offer buy now limited time",
    "You have been chosen to receive free money",
    "Earn extra income easily from your mobile",
    "Claim your special discount before it expires",
    "Win big cash today register for free",
    "Free entry in our lucky draw claim your prize"
]

labels = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
]

data = pd.DataFrame({
    "Email": emails,
    "Label": labels
})

data["Category"] = data["Label"].map({
    0: "Not Spam",
    1: "Spam"
})

print("Email Dataset:")
print(data.head())
print("\nTotal Emails:", len(data))
print("\nClass Distribution:")
print(data["Category"].value_counts())

# %%
# STEP 3: Separate input and output

X = data["Email"]
y = data["Label"]

# Convert email text into numerical TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

print("Text converted into numerical features.")
print("Feature matrix shape:", X.shape)

# %%
# STEP 4: Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training emails:", X_train.shape[0])
print("Testing emails:", X_test.shape[0])

# %%
# STEP 5: Train K-Nearest Neighbors (KNN) model

knn = KNeighborsClassifier(
    n_neighbors=3,
    metric="cosine",
    algorithm="brute"
)

knn.fit(X_train, y_train)

knn_predictions = knn.predict(X_test)

print("KNN model training completed.")

# %%
# STEP 6: Train Support Vector Machine (SVM) model

svm = SVC(kernel="linear")

svm.fit(X_train, y_train)

svm_predictions = svm.predict(X_test)

print("SVM model training completed.")

# %%
# STEP 7: Evaluate KNN model

print("KNN PERFORMANCE")
print("-------------------------")

knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions, zero_division=0)
knn_recall = recall_score(y_test, knn_predictions, zero_division=0)
knn_f1 = f1_score(y_test, knn_predictions, zero_division=0)

print("Accuracy :", round(knn_accuracy * 100, 2), "%")
print("Precision:", round(knn_precision * 100, 2), "%")
print("Recall   :", round(knn_recall * 100, 2), "%")
print("F1 Score :", round(knn_f1 * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    knn_predictions,
    labels=[0, 1],
    target_names=["Not Spam", "Spam"],
    zero_division=0
))

# %%
# STEP 8: Evaluate SVM model

print("SVM PERFORMANCE")
print("-------------------------")

svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, zero_division=0)
svm_recall = recall_score(y_test, svm_predictions, zero_division=0)
svm_f1 = f1_score(y_test, svm_predictions, zero_division=0)

print("Accuracy :", round(svm_accuracy * 100, 2), "%")
print("Precision:", round(svm_precision * 100, 2), "%")
print("Recall   :", round(svm_recall * 100, 2), "%")
print("F1 Score :", round(svm_f1 * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    svm_predictions,
    labels=[0, 1],
    target_names=["Not Spam", "Spam"],
    zero_division=0
))

# %%
# STEP 9: Display confusion matrices

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    knn_predictions,
    display_labels=["Not Spam", "Spam"],
    cmap="Blues",
    ax=axes[0],
    colorbar=False
)
axes[0].set_title("KNN Confusion Matrix")

ConfusionMatrixDisplay.from_predictions(
    y_test,
    svm_predictions,
    display_labels=["Not Spam", "Spam"],
    cmap="Greens",
    ax=axes[1],
    colorbar=False
)
axes[1].set_title("SVM Confusion Matrix")

plt.tight_layout()
plt.show()

# %%
# STEP 10: Compare KNN and SVM performance

comparison = pd.DataFrame({
    "Algorithm": ["KNN", "SVM"],
    "Accuracy": [knn_accuracy * 100, svm_accuracy * 100],
    "Precision": [knn_precision * 100, svm_precision * 100],
    "Recall": [knn_recall * 100, svm_recall * 100],
    "F1 Score": [knn_f1 * 100, svm_f1 * 100]
})

print("PERFORMANCE COMPARISON")
print(comparison.round(2).to_string(index=False))

# %%
# STEP 11: Plot accuracy comparison

plt.figure(figsize=(6, 4))
plt.bar(
    comparison["Algorithm"],
    comparison["Accuracy"]
)
plt.title("KNN vs SVM Accuracy")
plt.xlabel("Classification Algorithm")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.show()

# %%
# STEP 12: Test the models with a new email

new_email = ["Congratulations! You have won a free prize. Claim now"]

new_email_vector = vectorizer.transform(new_email)

knn_result = knn.predict(new_email_vector)[0]
svm_result = svm.predict(new_email_vector)[0]

class_names = {
    0: "Normal State (Not Spam)",
    1: "Abnormal State (Spam)"
}

print("New Email:", new_email[0])
print("\nKNN Prediction:", class_names[knn_result])
print("SVM Prediction:", class_names[svm_result])

# %%
# STEP 13: Final result

if svm_accuracy > knn_accuracy:
    print("SVM performed better than KNN based on accuracy.")
elif knn_accuracy > svm_accuracy:
    print("KNN performed better than SVM based on accuracy.")
else:
    print("Both algorithms achieved the same accuracy.")

print("\nEmail Spam Detection Completed Successfully!")
