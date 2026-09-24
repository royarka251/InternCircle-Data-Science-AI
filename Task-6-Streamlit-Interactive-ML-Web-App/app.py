
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger information to predict survival.")

# Inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

age = st.slider(
    "Age",
    min_value=0.0,
    max_value=80.0,
    value=30.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=8,
    value=0
)

parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=6,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=30.0
)

embarked = st.selectbox(
    "Embarkation Port",
    ["S", "C", "Q"]
)

if st.button("Predict Survival"):

    input_data = pd.DataFrame({
        "pclass": [pclass],
        "sex": [sex],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare],
        "embarked": [embarked]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🚢 Prediction: Passenger is likely to survive.")
    else:
        st.error("🚢 Prediction: Passenger is unlikely to survive.")
