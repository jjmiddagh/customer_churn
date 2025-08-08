# config.py

# Base directory for data and models
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# Derived paths
RAW_DATA_PATH = rf"{BASE_DIR}\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEANED_DATA_PATH = rf"{BASE_DIR}\data\processed\cleaned_data.csv"
DATA_WITH_IDS_PATH = rf"{BASE_DIR}\data\processed\data_with_ids.csv"

MODEL_DIR = rf"{BASE_DIR}\models"
VOTING_MODEL_PATH = rf"{MODEL_DIR}\voting_classifier.pkl"
RF_TUNING_RESULTS_PATH = rf"{MODEL_DIR}\rf_gridsearch_summary.csv"
XGB_TUNING_RESULTS_PATH = rf"{MODEL_DIR}\xgb_gridsearch_summary.csv"
LOG_FILE_PATH = rf"{MODEL_DIR}\tuning_logs.log"

# Script paths for automation
SCRIPT_LOAD_DATA = f"{BASE_DIR}\scripts\churn_01_load_data.py"
SCRIPT_MODELING = f"{BASE_DIR}\scripts\churn_02_modeling.py"

SCRIPTS_TO_RUN = [
    SCRIPT_LOAD_DATA,
    SCRIPT_MODELING
]
