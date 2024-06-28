import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title of the app
st.title('Credit Card Fraud Detection')

# Upload CSV data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Display the data
    st.write(data.head())

    # Make predictions
    predictions = model.predict(data)

    # Add predictions to the dataframe
    data['Prediction'] = predictions

    # Display the predictions
    st.write(data)
