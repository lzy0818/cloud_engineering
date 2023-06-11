import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

def acquire_data_kaggle():
    """
    This function downloads the data from the Kaggle dataset and returns a pandas DataFrame.
    Kaggle data link: https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud
    """
    # Initialize Kaggle API
    api = KaggleApi()

    # Set the path to the kaggle.json file
    kaggle_json_path = 'src/kaggle.json'

    # Authenticate using the specified kaggle.json file
    api.authenticate()

    # Define the dataset name and destination path
    dataset_name = 'dhanushnarayananr/credit-card-fraud'
    dest_path = 'data/credit_card_fraud/'

    # Download the dataset
    api.dataset_download_files(dataset_name, path=dest_path, unzip=True)

    # Read the extracted CSV file into a pandas DataFrame
    data_df = pd.read_csv(dest_path + 'card_transdata.csv')

    # data_df = pd.read_csv(data + '')
    return data_df
