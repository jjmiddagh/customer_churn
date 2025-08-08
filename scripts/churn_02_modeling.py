import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
file_path = r"\\MYCLOUDEX2ULTRA\Documents\DataScience\customerchurn\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# Drop unnecessary ID column
df.drop(columns=["customerID"], inplace=True)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Binary encode target
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# One-hot encode features
X = pd.get_dummies(df.drop("Churn", axis=1))
y = df["Churn"]

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Build individual classifiers
clf1 = LogisticRegression(max_iter=1000, random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=[("lr", clf1), ("rf", clf2), ("gb", clf3)],
    voting="soft"
)

# Train the model
print("Training voting classifier on SMOTE-balanced data...")
voting_clf.fit(X_resampled, y_resampled)

# Evaluate model
y_pred = voting_clf.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save the model and columns in Streamlit-friendly locations
os.makedirs("models", exist_ok=True)
joblib.dump(voting_clf, "models/voting_classifier.pkl")
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

print("\n✅ Model and column names saved to /models/")
