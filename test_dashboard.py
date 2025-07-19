import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Title
st.title("Run-to-Failure Dataset Dashboard")

# Dataset selector
csv_files = [f for f in os.listdir() if f.endswith(".csv")]
selected_file = st.selectbox("Choose a dataset file:", csv_files)

# Load selected file
df = pd.read_csv(selected_file)
st.write(f"Loaded file: `{selected_file}`")

# Show raw data
st.subheader("Raw Data")
st.write(df.head())

# Plot feature means
st.subheader("Feature Means")
mean_vals = df.mean()
fig, ax = plt.subplots()
mean_vals.plot(kind='bar', ax=ax)
st.pyplot(fig)
st.subheader("Feature Over Time")
feature = st.selectbox("Select a feature to plot over time", df.columns)
st.line_chart(df[feature])
if st.button("Run Model on Selected File"):
    # Insert code here to process the selected file through your model
    st.success("Model run completed!")
