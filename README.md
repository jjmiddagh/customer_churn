
# Customer Churn Prediction Project

## Overview
This project predicts customer churn using machine learning models and provides an interactive dashboard for analysis. The pipeline includes data preprocessing, model training, and visualization of results.

---

## Project Structure
```
CustomerChurn/
├── data/
│   ├── raw/                  # Original dataset
│   ├── processed/            # Cleaned and encoded datasets
├── models/                   # Trained models
├── scripts/                  # Python scripts for the pipeline
│   ├── churn_01_load_data.py # Data preprocessing
│   ├── churn_02_modeling.py  # Model training and evaluation
│   ├── dashboard.py          # Interactive dashboard
│   ├── run_pipeline.py       # Executes the entire pipeline
```

---

## How to Run

### 1. Setup Environment
1. Install Python 3.10 or higher.
2. Install required packages using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Execute the Pipeline
Run the pipeline to preprocess the data and train models:
```bash
python run_pipeline.py
```

### 3. Launch the Dashboard
Start the interactive dashboard:
```bash
streamlit run scripts/dashboard.py
```

---

## Key Features

### 1. Preprocessing
- Cleans missing values.
- Encodes categorical variables.
- Balances the dataset using SMOTE.

### 2. Modeling
- Trains Logistic Regression, Random Forest, and XGBoost models.
- Tunes hyperparameters for improved accuracy.
- Combines models using a Voting Classifier for better predictions.

### 3. Dashboard
- Displays top customers likely to churn.
- Visualizes churn probabilities and feature importance.
- Allows custom predictions for what-if scenarios.

---

## Dependencies
This project requires the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `streamlit`
- `seaborn`
- `matplotlib`
- `joblib`

For a full list, see `requirements.txt`.

---

## Dataset
The original dataset is located in `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

---

## Contact
For questions or feedback, please reach out to the project maintainer.
