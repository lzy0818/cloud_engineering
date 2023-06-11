import os
from pathlib import Path

import numpy as np
import streamlit as st
import boto3
import pickle
import pandas as pd
import argparse
from typing import Tuple

import aws_utils as aws 


# write a app.py that deploy the data and model on AWS S3
# and use streamlit to create a web app that can take the input and output the prediction
# the app.py should be able to take the input and output the prediction

# Create an argument parser to accept command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--access_key_id", help="AWS access key ID")
parser.add_argument("--secret_access_key", help="AWS secret access key")
parser.add_argument("--session_token", help="AWS session token")
args = parser.parse_args()

# Set up AWS credentials
AWS_ACCESS_KEY_ID = args.access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = args.secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = args.session_token or os.getenv("AWS_SESSION_TOKEN")

# Set up S3 bucket information, defaulting to the test bucket on zlk1071's account with name zlk1071-test-0 
BUCKET_NAME = os.getenv("BUCKET_NAME", "group4-project")
ARTIFACTS_PREFIX = Path(os.getenv("ARTIFACTS_PREFIX", ""))

# Create artifacts directory to keep model files
artifacts = Path() #/ "experiments"
artifacts.mkdir(exist_ok=True)

@st.cache_data
def load_model_versions(bucket_name = 'group4-project', directory_prefix='experiments/'):
    directory_names = aws.get_s3_directory_names(bucket_name, directory_prefix)
    return directory_names


@st.cache_data
def load_data(data_file, cloud_s3_key):
    print("Loading artifacts from: ", artifacts.absolute())
    # Download file from S3
    aws.download_s3(BUCKET_NAME, cloud_s3_key, data_file)
    # Load files into memory
    cloud = pd.read_csv(data_file)
    return cloud
    
@st.cache_resource
def load_model(model_file, cloud_model_s3_key):
    # Download the model file from S3
    aws.download_s3(BUCKET_NAME, cloud_model_s3_key, model_file)
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    return model

def slider_values(series) -> Tuple[float, float, float]:
    return (
        float(series.min()),
        float(series.max()),
        float(series.mean()),
    )

# Create the application title and description
st.title("Credit Card Fraud Detection")
st.write("This app detects credit card fraud based on the provided features using the Random Forest Classfier.")

st.subheader("Model Selection")
model_version = os.getenv("DEFAULT_MODEL_VERSION", "2022-version")
# st.write(f"Selected model version: {model_version}")

# Find available model versions in artifacts dir
available_models = load_model_versions()
# Create a dropdown to select the model
model_version = st.selectbox("Select Model", list(available_models))
st.write(f"Selected model version: {model_version}")

# Establish the dataset and model locations based on selection
data_file = artifacts / model_version / "meta.csv"
model_file = artifacts / model_version / "trained_model_object.pkl"

# configure the S3 location for each artifact
card_s3_key = str(ARTIFACTS_PREFIX/model_version/data_file.name)
card_model_s3_key = str(ARTIFACTS_PREFIX/model_version/model_file.name)

# Load the dataset and model into memory
X = load_data(data_file, card_s3_key)
model = load_model(model_file, card_model_s3_key)

# Sidebar inputs for feature values
st.sidebar.header("Input Parameters")
distance_from_home = st.sidebar.slider("distance_from_home", float(X.iloc[0, 0]), float(X.iloc[0, 1]+1))
distance_from_last_transaction = st.sidebar.slider("distance_from_last_transaction", float(X.iloc[1, 0]), float(X.iloc[1, 1]+1))
ratio_to_median_purchase_price = st.sidebar.slider("ratio_to_median_purchase_price",float(X.iloc[2, 0]), float(X.iloc[2, 1]+1))
repeat_retailer = st.radio("repeat_retailer", [0, 1])
used_chip = st.radio("used_chip", [0, 1])
used_pin_number = st.radio("used_pin_number", [0, 1])
online_order = st.radio("online_order", [0, 1])

# Make predictions on user inputs
# Normalize the inputs
num_features_norm = []
for i, feature in enumerate([distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price]):
    num_features_norm.append((feature - X.iloc[i, 2]) / X.iloc[i, 3])
input_data = pd.DataFrame([num_features_norm+[repeat_retailer, used_chip, used_pin_number, online_order]],\
                          columns=['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price',\
                                   'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order'])
prediction = int(model.predict(input_data))
if prediction == 0:
    pred_class = "Safe transaction."
else:
    pred_class = "Fraudulent transaction."

# Display the predicted class and probability
st.subheader("Prediction")
st.write(f"Predicted Class: {pred_class}")
st.write(f"Probability: {model.predict_proba(input_data)[0][prediction]:.2f}")
