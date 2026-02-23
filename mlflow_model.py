# import mlflow
# import mlflow.sklearn
# import joblib


# mlflow.set_experiment("Heart_Disease_Project") 

# rf = {
#     "n_estimators": 100,
#     "max_depth": 5,
#     "random_state": 42,
#     "min_samples_split": 2,
# }

# matrf = {
#     "Accuracy": 0.88215,
#     "AUC": 0.94857
# }

# xgb = {
#     "device": "cuda",
#     "tree_method": "hist",
#     "n_estimators": 1000,
#     "learning_rate": 0.05,
#     "max_depth": 6,
#     "subsample": 0.8,
#     "colsample_bytree": 0.8,
#     "scale_pos_weight": 1,
#     "eval_metric": 'auc',
#     "early_stopping_rounds": 50,   # stops if no improvement for 50 rounds
#     "random_state": 42,
#     "n_jobs": -1
# }

# matxgb = {
#     "Accuracy": 0.8867,
#     "AUC": 0.9551
# }

# xgb_tuned = {
#     "subsample": 0.9,
#     "reg_lambda": 1,
#     "reg_alpha": 0,
#     "n_estimators": 1000,
#     "min_child_weight": 5,
#     "max_depth": 4,
#     "learning_rate": 0.05,
#     "gamma": 0.1,
#     "colsample_bytree": 0.9
# }

# matxgb_tuned = {
#     "Accuracy": 0.8870,
#     "AUC": 0.9552
# }

# cat = {
#     "iterations": 1000,
#     "learning_rate": 0.05,
#     "depth": 6,
#     "task_type": "CPU",           # Switched to CPU due to CUDA driver incompatibility
#     "loss_function": "Logloss",
#     "eval_metric": "AUC",
#     "early_stopping_rounds": 50,
#     "random_seed": 42,
#     "verbose": 100,
# }

# matcat = {
#     "Accuracy": 0.8845,
#     "AUC": 0.95529
# }

# cat_tuned = {
#     "iterations": 5000,
#     "learning_rate": 0.05,
#     "depth": 5,
#     "task_type": "GPU",           # Switched to GPU mode (assuming CUDA is available)
#     "devices": "0",              # Specify GPU device if multiple are present
#     "loss_function": "Logloss",
#     "eval_metric": "AUC",
#     "early_stopping_rounds": 300,
#     "verbose": 500,
#     "l2_leaf_reg": 8,           # L2 regularization
#     "scale_pos_weight": 1
# }

# matcat_tuned = {
#     "Accuracy": 0.8897,
#     "AUC": 0.9561
# }

# lbgm = {
#     "n_estimators": 1000,
#     "learning_rate": 0.05,
#     "max_depth": 6,
#     "subsample": 0.8,
#     "colsample_bytree": 0.8,
#     "scale_pos_weight": 1,
#     "device": "gpu",            # Use GPU for training
#     "random_state": 42,
#     "early_stopping_rounds": 50,
#     "verbose": 100
# }

# matlgbm = {
#     "Accuracy": 0.8867,
#     "AUC": 0.9551
# }

# Random_Forest = joblib.load("Model/rf.pkl")
# XGBoost = joblib.load("Model/xgb.pkl")
# CatBoost = joblib.load("Model/cat.pkl")
# LightBGM = joblib.load("Model/lgbm.pkl")
# XGBoost_Tuned = joblib.load("Model/xgb_Tuned.pkl")
# CatBoost_Tuned = joblib.load("Model/catboostclaude.pkl")

# def log_model(model, params, metrics, run_name):
#     with mlflow.start_run(run_name=run_name):
#         for key, value in params.items():
#             mlflow.log_param(key, value)

#         for key, value in metrics.items():
#             mlflow.log_metric(key, value)

#         mlflow.sklearn.log_model(model, "model")

# log_model(Random_Forest, rf, matrf, "Random_Forest")
# log_model(XGBoost, xgb, matxgb, "XGBoost")
# log_model(CatBoost, cat, matcat, "CatBoost")
# log_model(LightBGM, lbgm, matlgbm, "LightBGM")
# log_model(XGBoost_Tuned, xgb_tuned, matxgb_tuned, "XGBoost_Tuned")
# log_model(CatBoost_Tuned, cat_tuned, matcat_tuned, "Cat_Boost_Tuned")

import mlflow
import mlflow.sklearn
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Train.csv")

X = df.drop('Heart Disease', axis=1)

le = LabelEncoder()
df['Heart Disease'] = le.fit_transform(df['Heart Disease'])
y = df['Heart Disease']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Heart_Disease_Project")

def log_model(model_path, run_name, X_test, y_test):
    model = joblib.load(model_path)

    with mlflow.start_run(run_name=run_name):

        # Log parameters automatically
        params = model.get_params()
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Calculate metrics dynamically
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model
        mlflow.sklearn.log_model(model, "model")

log_model("Model/rf.pkl", "Random_Forest", X_test, y_test)
log_model("Model/xgb.pkl", "XGBoost", X_test, y_test)
# log_model("Model/cat.pkl", "CatBoost", X_test, y_test)
log_model("Model/lgbm.pkl", "LightBGM", X_test, y_test)
log_model("Model/xgb_tuned.pkl", "XGBoost_Tuned", X_test, y_test)
# log_model(CatBoost_Tuned, "Cat_Boost_Tuned", X_test, y_test)
