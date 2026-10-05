import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Dataset
data = {
    "Area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500],
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    "Bathrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 5],
    "Price": [25, 30, 40, 48, 55, 65, 75, 85, 95, 110]
}

df = pd.DataFrame(data)

# Features and Target
X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction on test data
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("House Price Prediction")
print("----------------------")
print("MAE:", mae)
print("R2 Score:", r2)

# Take user input
area = float(input("\nEnter house area in sq.ft: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

# Predict price
new_house = pd.DataFrame(
    [[area, bedrooms, bathrooms]],
    columns=["Area", "Bedrooms", "Bathrooms"]
)

predicted_price = model.predict(new_house)

print("\nPredicted House Price:", round(predicted_price[0], 2), "Lakhs")