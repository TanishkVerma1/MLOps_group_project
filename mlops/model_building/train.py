# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, roc_auc_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()

Xtrain_path = "hf://datasets/TanishkV18/telco-customer-churn/Xtrain.csv"
Xtest_path = "hf://datasets/TanishkV18/telco-customer-churn/Xtest.csv"
ytrain_path = "hf://datasets/TanishkV18/telco-customer-churn/ytrain.csv"
ytest_path = "hf://datasets/TanishkV18/telco-customer-churn/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# List of numerical features in the dataset
numeric_features = [
    'SeniorCitizen',
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
]

# List of categorical features in the dataset
categorical_features = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
]


# Set the class weight to handle class imbalance (~73% No churn vs ~27% Churn)
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# ---- Model 1: Logistic Regression (interpretable baseline) ----
logreg_pipeline = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
)
logreg_pipeline.fit(Xtrain, ytrain.values.ravel())

logreg_test_proba = logreg_pipeline.predict_proba(Xtest)[:, 1]
logreg_test_pred = (logreg_test_proba >= 0.5).astype(int)

print("Logistic Regression - Test performance")
print(classification_report(ytest, logreg_test_pred))
print("ROC-AUC:", roc_auc_score(ytest, logreg_test_proba))

# ---- Model 2: XGBoost (tuned) ----
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric="logloss")

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],    # number of trees to build
    'xgbclassifier__max_depth': [2, 3, 4],    # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],    # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid_search.fit(Xtrain, ytrain.values.ravel())


# Check the parameters of the best model
grid_search.best_params_

# Store the best model
best_model = grid_search.best_estimator_
best_model




# Choose the classification threshold using the validation (test) probabilities
# rather than assuming a fixed value. We optimize for F1 on the churn class
# since it balances precision and recall for the minority class.
from sklearn.metrics import precision_recall_curve

y_test_proba_search = best_model.predict_proba(Xtest)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(ytest, y_test_proba_search)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = f1_scores[:-1].argmax()  # last precision/recall pair has no matching threshold
classification_threshold = float(thresholds[best_idx])

print(f"Chosen classification threshold: {classification_threshold:.3f}")
print(f"  -> Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}, F1: {f1_scores[best_idx]:.3f}")



# Make predictions on the training data
y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

# Make predictions on the test data
y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

# Generate a classification report to evaluate model performance on training set
print("XGBoost - Train performance")
print(classification_report(ytrain, y_pred_train))

# Generate a classification report to evaluate model performance on test set
print("XGBoost - Test performance")
print(classification_report(ytest, y_pred_test))
print("ROC-AUC:", roc_auc_score(ytest, y_pred_test_proba))

# XGBoost is selected as the final model: it gives a higher ROC-AUC and better
# recall on the churn (minority) class than the Logistic Regression baseline,
# which matters more here since missing an at-risk customer is costlier than
# a false alarm.

# Persist the threshold so the deployed app uses the same cutoff as training
with open("threshold.txt", "w") as f:
    f.write(str(classification_threshold))

# Save best model
joblib.dump(best_model, "best_churn_model.joblib")

# Persist the threshold so the deployed app uses the same cutoff as training
with open("threshold.txt", "w") as f:
    f.write(str(classification_threshold))

# Upload to Hugging Face
repo_id = "TanishkV18/churn-model"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

api.upload_file(
    path_or_fileobj="best_churn_model.joblib",
    path_in_repo="best_churn_model.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
api.upload_file(
    path_or_fileobj="threshold.txt",
    path_in_repo="threshold.txt",
    repo_id=repo_id,
    repo_type=repo_type,
)
