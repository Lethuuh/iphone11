import streamlit as st
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt

# Load data
patients = pd.read_csv("patients.csv")
vitals = pd.read_csv("vitals.csv")
medications = pd.read_csv("medications.csv")
patient_features = pd.read_csv("patient_features.csv")
daily_temp = pd.read_csv("daily_temp.csv")
forecast = pd.read_csv("forecast.csv")
clf = joblib.load("risk_classifier.pkl")

# App layout
st.set_page_config(page_title="Healthcare Dashboard", layout="wide")
st.title("🩺 Synthetic Healthcare Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Patients",
    "📈 Vitals & Forecast",
    "💊 Medications",
    "⚠️ Risk Prediction"
])

# Tab 1: Patients
with tab1:
    st.header("Patient Demographics")
    st.dataframe(patients)

# Tab 2: Vitals & Forecast
with tab2:
    st.subheader("Vitals Summary")
    selected_id = st.selectbox("Select PatientID", patients['PatientID'])
    vitals_filtered = vitals[vitals['PatientID'] == selected_id]
    st.dataframe(vitals_filtered)

    st.subheader("Temperature Forecast")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3)
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("30-Day Temperature Forecast")
    ax.legend()
    st.pyplot(fig)

# Tab 3: Medications
with tab3:
    st.header("Medication Records")
    meds_filtered = medications[medications['PatientID'] == selected_id]
    st.dataframe(meds_filtered)

# Tab 4: Risk Prediction
with tab4:
    st.header("Predict Risk for New Patient")
    age = st.slider("Age", 20, 90, 50)
    hr = st.slider("Average Heart Rate", 60, 150, 85)
    temp = st.slider("Average Temperature", 35.0, 42.0, 37.0)
    days = st.slider("Active Days", 1, 365, 30)

    input_df = pd.DataFrame([[age, hr, temp, days]], columns=['Age', 'AvgHeartRate', 'AvgTemp', 'ActiveDays'])
    prediction = clf.predict(input_df)[0]
    st.write("🩺 Risk Prediction:", "🔴 High Risk" if prediction == 1 else "🟢 Low Risk")