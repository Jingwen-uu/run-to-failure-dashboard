import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# --- Helper function to add RUL ---
def add_rul(df):
    df["RUL"] = len(df) - df.index
    return df

# --- Load & prepare data ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = add_rul(df)
    df.columns = [f"Sensor{i+1}" for i in range(len(df.columns)-1)] + ["RUL"]
    return df

# --- Train a simple model ---
@st.cache_data
def train_model(df):
    X = df.drop("RUL", axis=1)
    y = df["RUL"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- Streamlit UI ---
st.title("Run-to-Failure Dataset Explorer & RUL Predictor")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("### Dataset Sample")
    st.dataframe(data.head())

    # Select sensors to plot
    sensors = st.multiselect("Select sensors to plot", options=data.columns[:-1], default=data.columns[:3])

    if sensors:
        st.write("### Sensor Data")
        fig, ax = plt.subplots(figsize=(10, 5))
        for sensor in sensors:
            ax.plot(data[sensor], label=sensor)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Sensor Value")
        ax.legend()
        st.pyplot(fig)

    # Train and predict RUL
    model = train_model(data)
    X = data.drop("RUL", axis=1)
    data["Predicted RUL"] = model.predict(X)

    st.write("### RUL Prediction")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(data["RUL"], label="Actual RUL")
    ax2.plot(data["Predicted RUL"], label="Predicted RUL")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("RUL")
    ax2.legend()
    st.pyplot(fig2)
else:
    st.info("Please upload a CSV file to start.")
    st.write("Dashboard loaded successfully!")
