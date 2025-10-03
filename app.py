import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv("data/saprin_health_episodes.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/rf_classifier.pkl")

# App layout
st.title("📊 NHI Healthcare Demand Prediction")

tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Preview", "📈 Visualizations", "🤖 Predictions", "📋 Model Evaluation"])

# Tab 1: Data Preview
with tab1:
    st.header("Dataset Overview")
    df = load_data()
    st.write(df.head())
    st.markdown(f"**Shape:** {df.shape}")
    st.markdown(f"**Columns:** {list(df.columns)}")

# Tab 2: Visualizations
with tab2:
    st.header("Feature Distributions")
    selected_feature = st.selectbox("Choose a feature to visualize", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax)
    st.pyplot(fig)

# Tab 3: Predictions
with tab3:
    st.header("Predict Visit Type")
    model = load_model()

    age = st.slider("Age", 0, 100, 30)
    income = st.slider("Income (scaled)", 0.0, 1.0, 0.5)
    region = st.selectbox("Region", [0, 1, 2])  # Replace with actual region labels
    sex = st.selectbox("Sex", [0, 1])  # Replace with actual sex labels

    input_df = pd.DataFrame([[age, income, region, sex]], columns=["age", "income", "region", "sex"])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Visit Type: **{prediction}**")

# Tab 4: Model Evaluation
with tab4:
    st.header("Model Performance")
    st.markdown("""
    - **Accuracy**: 0.87  
    - **Precision**: 0.85  
    - **Recall**: 0.83  
    - **F1-score**: 0.84  
    """)
    st.markdown("These metrics were computed in Google Colab and reflect the model's ability to classify visit types accurately.")
