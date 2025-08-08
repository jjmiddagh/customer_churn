
# churn_01_load_data.py Documentation

## Overview
This script is responsible for data preprocessing, including handling missing values, encoding categorical features, 
and saving the cleaned dataset for further analysis.

---

## Dependencies
- **Libraries**:
  - `pandas`: For data manipulation.
- **Input File**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Output Files**:
  - `cleaned_data.csv`: Processed dataset for modeling.
  - `data_with_ids.csv`: Dataset with `customerID` retained for dashboard usage.

---

## Code Section Breakdown

### 1. Load the Dataset
```python
data = pd.read_csv(r'Y:\DataScience\CustomerChurn\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv')
```
- **What it does**: Reads the raw dataset into a Pandas DataFrame.
- **Why it's done**: This step loads the dataset for preprocessing.

---

### 2. Convert `TotalCharges` to Numeric
```python
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
```
- **What it does**: Converts `TotalCharges` column to numeric. Non-numeric values are replaced with `NaN`.
- **Why it's done**: Machine learning models require numerical data for continuous variables.

---

### 3. Handle Missing Values in `TotalCharges`
```python
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
```
- **What it does**: Replaces missing values in `TotalCharges` with the median.
- **Why it's done**: Imputation ensures the dataset has no missing values, preventing errors during modeling.

---

### 4. Encode the Target Variable (`Churn`)
```python
data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})
```
- **What it does**: Maps `No` to `0` and `Yes` to `1` in the `Churn` column.
- **Why it's done**: Converts the target variable into a format suitable for machine learning algorithms.

---

### 5. One-Hot Encode Categorical Features
```python
data = pd.get_dummies(data, drop_first=True)
```
- **What it does**: Converts categorical variables into one-hot encoded columns.
- **Why it's done**: Many machine learning models require numerical input, so categorical data must be encoded.

---

### 6. Save the Processed Datasets
```python
data_with_ids = data.copy()
data_with_ids.to_csv(r'Y:\DataScience\CustomerChurn\data\processed\data_with_ids.csv', index=False)
data.to_csv(r'Y:\DataScience\CustomerChurn\data\processed\cleaned_data.csv', index=False)
```
- **What it does**: Saves two versions of the dataset:
  - One with `customerID` for dashboard use.
  - One without `customerID` for modeling.
- **Why it's done**: Ensures the correct dataset is used for each task.

---

## Summary
This script prepares the raw dataset for analysis and modeling by cleaning, encoding, and saving the necessary files. 
It is the foundation for the entire project pipeline.

