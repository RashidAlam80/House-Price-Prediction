import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("🏠 House Price Prediction")

st.write("Enter house details to predict the estimated price.")

# Sample dataset
data = {
    "Area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500],
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    "Bathrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 5],
    "Price": [25, 30, 40, 48, 55, 65, 75, 85, 95, 110]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# User inputs
area = st.number_input(
    "Area (sq.ft)",
    min_value=500,
    max_value=10000,
    value=1500
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

# Prediction
if st.button("Predict House Price"):

    new_house = pd.DataFrame(
        [[area, bedrooms, bathrooms]],
        columns=["Area", "Bedrooms", "Bathrooms"]
    )

    prediction = model.predict(new_house)

    st.success(
        f"Estimated House Price: ₹{prediction[0]:.2f} Lakhs"
    )