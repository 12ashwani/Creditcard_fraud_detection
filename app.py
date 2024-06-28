import streamlit as st
import pandas as pd
import pickle

# Function to load the model
def load_model(uploaded_file):
    model = pickle.load(uploaded_file)
    return model

# Title of the app
st.title('Credit Card Fraud Detection')

# Upload model file
uploaded_model_file = st.file_uploader("Upload your trained model", type=["pkl"])

# Load the model if uploaded
if uploaded_model_file is not None:
    model = load_model(uploaded_model_file)
    st.success("Model loaded successfully")

    # Upload CSV data
    uploaded_data_file = st.file_uploader("Upload CSV file for prediction", type="csv")

    if uploaded_data_file is not None:
        # Read the CSV file
        data = pd.read_csv(uploaded_data_file)

        # Display the data
        st.write("Input data preview:")
        st.write(data.head())

        # Check if model is loaded
        if model:
            # Make predictions
            predictions = model.predict(data)

            # Add predictions to the dataframe
            data['Prediction'] = predictions

            # Display the predictions
            st.write("Predictions:")
            st.write(data)
else:
    st.info("Please upload your trained model file to proceed.")
