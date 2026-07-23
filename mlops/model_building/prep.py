# for data manipulation
import pandas as pd
import numpy as np
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/TanishkV18/telco-customer-churn/Telco-Customer-Churn.csv"
telco_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the customer identifier - it carries no predictive information
telco_dataset = telco_dataset.drop(columns=["customerID"])

# TotalCharges is loaded as an object dtype because a handful of new customers
# (tenure == 0) have blank strings instead of a numeric value. Coerce to numeric
# and drop the resulting missing rows before splitting.
telco_dataset["TotalCharges"] = pd.to_numeric(telco_dataset["TotalCharges"], errors="coerce")
telco_dataset = telco_dataset.dropna(subset=["TotalCharges"])

# Define the target variable for the classification task
target = 'Churn'

# List of numerical features in the dataset
numeric_features = [
    'SeniorCitizen',     # Whether the customer is a senior citizen (binary: 0 or 1)
    'tenure',            # Number of months the customer has stayed with the company
    'MonthlyCharges',    # The amount charged to the customer monthly
    'TotalCharges',      # The total amount charged to the customer
]

# List of categorical features in the dataset
categorical_features = [
    'gender',              # Customer's gender
    'Partner',             # Whether the customer has a partner
    'Dependents',          # Whether the customer has dependents
    'PhoneService',        # Whether the customer has phone service
    'MultipleLines',       # Whether the customer has multiple lines
    'InternetService',     # Customer's internet service provider
    'OnlineSecurity',      # Whether the customer has online security
    'OnlineBackup',        # Whether the customer has online backup
    'DeviceProtection',    # Whether the customer has device protection
    'TechSupport',         # Whether the customer has tech support
    'StreamingTV',         # Whether the customer has streaming TV
    'StreamingMovies',     # Whether the customer has streaming movies
    'Contract',            # The contract term of the customer
    'PaperlessBilling',    # Whether the customer has paperless billing
    'PaymentMethod',       # The customer's payment method
]

# Define predictor matrix (X) using selected numeric and categorical features
X = telco_dataset[numeric_features + categorical_features]

# Define target variable (encode Yes/No as 1/0)
y = telco_dataset[target].map({'Yes': 1, 'No': 0})


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42,   # Ensures reproducibility by setting a fixed random seed
    stratify=y         # Preserve the churn/no-churn ratio in both splits
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="TanishkV18
        /telco-customer-churn",
        repo_type="dataset",
    )
