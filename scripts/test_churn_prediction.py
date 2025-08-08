import pandas as pd
import joblib

# Load the new model and column definitions
model = joblib.load("scripts/voting_classifier.pkl")
columns = joblib.load("scripts/model_columns.pkl")

# 🔍 Define a realistic high-churn risk customer
user_input = {
    "SeniorCitizen": 1,
    "tenure": 1,
    "MonthlyCharges": 115.0,
    "TotalCharges": 115.0,
    "gender_Male": 1,
    "Partner_No": 1,
    "Dependents_No": 1,
    "PhoneService_Yes": 1,
    "MultipleLines_Yes": 1,
    "InternetService_Fiber optic": 1,
    "OnlineSecurity_No": 1,
    "OnlineBackup_No": 1,
    "DeviceProtection_No": 1,
    "TechSupport_No": 1,
    "StreamingTV_Yes": 1,
    "StreamingMovies_Yes": 1,
    "Contract_Month-to-month": 1,
    "PaperlessBilling_Yes": 1,
    "PaymentMethod_Electronic check": 1
}


# Convert to DataFrame
user_df = pd.DataFrame([user_input])

# Match training columns
user_df = user_df.reindex(columns=columns, fill_value=0)

# Predict
prediction = model.predict(user_df)[0]
probability = model.predict_proba(user_df)[0][1]

print("\n✅ Prediction:", "Churn" if prediction == 1 else "No Churn")
print(f"🔢 Churn Probability: {probability:.2%}")
print("\n🧾 Final Input Vector Passed to Model:")
print(user_df.T)


