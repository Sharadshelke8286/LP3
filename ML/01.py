# ============================================================
# PRACTICAL 6: UBER RIDE FARE PRICE PREDICTION
# Machine Learning using Linear Regression
# Dataset required: uber.csv
# Compatible with Jupyter Notebook
# ============================================================

# https://www.kaggle.com/datasets/yasserh/uber-fares-dataset
# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# STEP 2: LOAD DATASET

df = pd.read_csv("uber.csv")

print("First five rows of dataset:")
print(df.head())


# STEP 3: DISPLAY DATASET INFORMATION

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())


# STEP 4: REMOVE MISSING VALUES AND DUPLICATES

df = df.dropna()
df = df.drop_duplicates()

print("\nDataset after cleaning:", df.shape)


# STEP 5: REMOVE UNNECESSARY COLUMNS

df = df.drop(['Unnamed: 0', 'key'], axis=1, errors='ignore')


# STEP 6: REMOVE INVALID DATA

required_columns = [
    'fare_amount',
    'pickup_datetime',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'passenger_count'
]

df = df.dropna(subset=required_columns)

df = df[
    (df['fare_amount'] > 0) &
    (df['passenger_count'] > 0) &
    (df['passenger_count'] <= 6) &
    (df['pickup_latitude'].between(-90, 90)) &
    (df['dropoff_latitude'].between(-90, 90)) &
    (df['pickup_longitude'].between(-180, 180)) &
    (df['dropoff_longitude'].between(-180, 180))
].copy()

print("\nDataset after removing invalid data:", df.shape)


# STEP 7: CONVERT PICKUP DATE AND TIME

df['pickup_datetime'] = pd.to_datetime(
    df['pickup_datetime'], errors='coerce', utc=True
)

df = df.dropna(subset=['pickup_datetime']).copy()

df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# STEP 8: CALCULATE DISTANCE BETWEEN PICKUP AND DROP-OFF

def calculate_distance(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2 +
        np.cos(lat1) * np.cos(lat2) *
        np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return 6371 * c


df['distance'] = calculate_distance(
    df['pickup_latitude'],
    df['pickup_longitude'],
    df['dropoff_latitude'],
    df['dropoff_longitude']
)

print("\nDistance calculated successfully.")


# STEP 9: REMOVE OUTLIERS

df = df[
    (df['fare_amount'] < 200) &
    (df['distance'] > 0) &
    (df['distance'] < 50)
].copy()

print("\nFinal cleaned dataset:", df.shape)


# STEP 10: SELECT INPUT AND OUTPUT VARIABLES

X = df[
    [
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
        'passenger_count',
        'distance',
        'hour',
        'day',
        'month'
    ]
]

y = df['fare_amount']

print("\nInput variables:")
print(X.head())

print("\nOutput variable:")
print(y.head())

# STEP 11: SPLIT DATASET INTO TRAINING AND TESTING DATA

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# STEP 12: TRAIN LINEAR REGRESSION MODEL

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")


# STEP 13: PREDICT FARE PRICE

y_pred = model.predict(X_test)

print("\nActual Fare:")
print(y_test.head(10).to_list())

print("\nPredicted Fare:")
print(y_pred[:10])


# STEP 14: EVALUATE MODEL PERFORMANCE

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


# STEP 15: VISUALIZE ACTUAL VS PREDICTED FARE

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")

plt.title("Uber Fare Prediction")

plt.show()


# STEP 16: PREDICT FARE FOR A NEW RIDE

pickup_lat = 40.7614
pickup_lon = -73.9776

drop_lat = 40.6413
drop_lon = -73.7781

passengers = 2

distance = calculate_distance(
    pickup_lat,
    pickup_lon,
    drop_lat,
    drop_lon
)

new_ride = pd.DataFrame(
    [[
        pickup_lon,
        pickup_lat,
        drop_lon,
        drop_lat,
        passengers,
        distance,
        10,     # Pickup hour
        16,     # Pickup day
        11      # Pickup month
    ]],
    columns=X.columns
)

predicted_fare = model.predict(new_ride)

print("\nNew Uber Ride Prediction:")
print("Distance:", round(float(distance), 2), "KM")

print(
    "Predicted Uber Fare: $",
    round(float(predicted_fare[0]), 2)
)


# ============================================================
# END OF PRACTICAL 6
# ============================================================
  
