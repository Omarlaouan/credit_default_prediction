"""
# Credit Default Prediction App

This app predicts whether a customer will default on their loan based on their personal and financial details. To use the app, upload a CSV file containing the customer data and click the "Predict" button to see the predictions and confidence levels for each customer.

The prediction model is built using the XGBoost algorithm, which is known for its high performance and accuracy in various classification tasks.

Please note that the accuracy of the predictions may vary depending on the quality and relevance of the input data. Always use the app with caution and consider additional factors before making any critical decisions.

To get started, simply follow the instructions below : 

1. Open your terminal (MacOS) or command prompt (Windows).
2. Navigate to the directory where the Streamlit app file (e.g., `app.py`) is located.
3. Make sure you have Streamlit and other required libraries installed. If not, run:
4. Launch the app by running the following command in the terminal: streamlit run app.py
5. A new browser window will open automatically with the app running. If it doesn't, copy and paste the URL provided in the terminal into your web browser.

Now you're ready to upload your CSV file and get predictions 🙂 !
"""


import streamlit as st
import os
import sys
import pandas as pd
import joblib
import base64

# Get the absolute path to the credit_default_prediction directory
credit_default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Import the processing function
src_path = os.path.join(credit_default_dir, 'src')
sys.path.append(src_path)
from data_processing import process_data

# Load the model
model_name = 'xgb_tuned.joblib'
model_path = os.path.join(credit_default_dir, 'models', model_name)
model = joblib.load(model_path)

def create_download_link(df, filename='output.csv'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'

# Define the Streamlit app
def app():
    st.set_page_config(page_title='Default Prediction App')
    st.title('Default Prediction App')
    st.write("""
    - This app predicts whether a customer will default on their loan based on their financial details 💶. 
    - To use the app, upload a CSV file 📄 containing the customer data and click the **"Predict"** button to see the predictions and confidence levels for each customer.
    - The prediction model is built using the XGBoost algorithm 🤖, which is known for its high performance and accuracy in various classification tasks.
    - 🚨 Please note that the accuracy of the predictions may vary depending on the quality and relevance of the input data. Always use the app with caution and consider additional factors before making any critical decisions.""")
    
    st.write('Upload a CSV file with the customer details to predict if they will default on a loan:')

    # Define the file uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the uploaded file as a DataFrame
        input_data = pd.read_csv(uploaded_file)

        # Store the customer_ID column before processing the input data
        customer_ids = input_data['customer_id']

        if st.button('Predict'):
            # Preprocess the input data
            input_data_processed = process_data(input_data)

            # Make a prediction using the model
            prediction_probs = model.predict_proba(input_data_processed)
            prediction_classes = model.predict(input_data_processed)

            # Display the predictions and confidence
            output = []
            for i, (prediction_prob, prediction_class) in enumerate(zip(prediction_probs[:, 1], prediction_classes)):
                if prediction_class == 0:
                    output.append(f"{customer_ids.iloc[i]}: Not likely to default with a confidence of {round((1-prediction_prob)*100, 2)}%")
                else:
                    output.append(f"{customer_ids.iloc[i]}: Likely to default with a confidence of {round(prediction_prob*100, 2)}%")
            
            for line in output:
                st.write(line)

            # Add predictions and confidence to the input DataFrame
            input_data['prediction'] = prediction_classes
            input_data['confidence'] = prediction_probs.max(axis=1) * 100

            # Replace numeric values in the 'prediction' column with descriptive labels
            input_data['prediction'] = input_data['prediction'].replace({0: 'no default', 1: 'default'})

            # Download the result as a CSV file
            st.markdown(create_download_link(input_data), unsafe_allow_html=True)

if __name__ == '__main__':
    app()
