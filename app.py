import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

st.sidebar.header("Enter House Details")

crim = st.sidebar.number_input("Crime rate", min_value=0.0, value=0.1)
zn = st.sidebar.number_input("Residential land zone (%)", min_value=0.0, value=0.0)
indus = st.sidebar.number_input("Industrial area proportion", min_value=0.0, value=5.0)
chas = st.sidebar.selectbox("Near river?", [0, 1])
nox = st.sidebar.number_input("Air pollution (NOX)", min_value=0.0, value=0.5)
rm = st.sidebar.number_input("Average number of rooms", min_value=1.0, value=6.0)
age = st.sidebar.number_input("Age of house (years)", min_value=0.0, value=50.0)
dis = st.sidebar.number_input("Distance to city centers", min_value=0.0, value=4.0)
rad = st.sidebar.number_input("Highway accessibility index", min_value=0, value=4)
tax = st.sidebar.number_input("Property tax rate", min_value=0, value=300)
ptratio = st.sidebar.number_input("Student–teacher ratio", min_value=1.0, value=18.0)
b = st.sidebar.number_input("Population proportion (B)", min_value=0.0, value=390.0)
lstat = st.sidebar.number_input("Lower income population (%)", min_value=0.0, value=12.0)

if st.button("Predict Price"):
    features = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
    prediction = model.predict(features)

    price_lakhs = prediction[0]
    price_rupees = price_lakhs * 100000

    st.success(f"🏷️ Estimated House Price: ₹ {price_rupees:,.0f}")
    st.caption("This prediction is based on a Linear Regression model trained on historical housing data.")

