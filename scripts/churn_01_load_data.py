import pandas as pd
import config


# Load the dataset
data = pd.read_csv(config.RAW_DATA_PATH)

# Convert TotalCharges to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with median
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# Keep a copy of the original data for analysis
data_with_ids = data.copy()

# Drop customerID for machine learning purposes
data = data.drop('customerID', axis=1)

# Encode the target variable (Churn)
data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})

# One-hot encode categorical features
data = pd.get_dummies(data, drop_first=True)

# Save the cleaned dataset
data.to_csv(config.CLEANED_DATA_PATH, index=False)

# Save the dataset with customerID for visualization
data_with_ids['Churn'] = data['Churn']
data_with_ids.to_csv(config.DATA_WITH_IDS_PATH, index=False)

print("Data cleaning and encoding completed. Processed dataset saved.")
