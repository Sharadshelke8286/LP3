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
