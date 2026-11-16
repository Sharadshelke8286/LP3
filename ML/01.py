
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
