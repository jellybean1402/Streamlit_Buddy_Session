import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

with open('rfr.pkl', 'rb') as file:
    rfr = pickle.load(file)

rfc = joblib.load('rfc.joblib')

st.title('Streamlit Frontend')

def main():

    st.header("Diabetes Prediction")

    pregnancies = st.slider("Pregnancies",
                            min_value = 0,
                            max_value = 17,
                            value = 0,
                            step = 1)
    glucose = st.slider("Glucose",
                              min_value = 0,
                              max_value = 199,
                              step = 1)
    blood_pressure = st.slider("BloodPressure",
                                     min_value = 0,
                                     max_value = 122,
                                     step = 1)
    skin_thickness = st.slider("Skin Thickness",
                                     min_value = 0,
                                     max_value = 99,
                                     step = 1)
    insulin = st.slider("Insulin",
                              min_value=0,
                              max_value=846,
                              step=1)
    bmi = st.number_input("BMI",
                          placeholder="0.0 to 67.1",
                          min_value=0.0,
                          max_value=67.1,
                          step=0.1)
    diabetes_pedigree_function = st.slider("DiabetesPedigreeFunction",
                                                 min_value=0.078,
                                                 max_value=2.42,
                                                 step=0.001)
    age = st.slider("Age",
                          min_value=21,
                          max_value=81,
                          step=1)

    rfc_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age
    }

    rfc_test_df = pd.DataFrame(rfc_dict, index = [0])
    rfc_pred = rfc.predict(rfc_test_df)

    st.header("Result")
    if rfc_pred == 1:
        st.error("Diabetes")
    elif rfc_pred == 0:
        st.success("No Diabetes")




    st.header("Sales Prediction")

    tv = st.slider("TV Sales", min_value = 0.7, max_value = 296.4, step = 0.1)
    radio = st.slider("Radio Sales", min_value = 0.0, max_value = 49.6, step = 0.1)
    newspaper = st.slider("Newspaper", min_value = 0.3, max_value = 114.4, step = 0.1)

    rfr_dict = {
        "TV": tv,
        "Radio": radio,
        "Newspaper": newspaper
    }

    rfr_test_df = pd.DataFrame(rfr_dict, index = [0])
    rfr_pred = rfr.predict(rfr_test_df)

    st.header(f"Sales: {rfr_pred[0]:.1f}")

if __name__ == '__main__':
    main()
