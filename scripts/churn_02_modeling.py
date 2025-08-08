import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os

# Load the data
file_path = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(file_path)

# Drop customerID
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Handle TotalCharges conversion and missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Binary encode target
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# One-hot encode categorical features
X = pd.get_dummies(df.drop('Churn', axis=1))
y = df['Churn']

# Apply SMOTE to training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Build voting classifier
clf1 = LogisticRegression(max_iter=1000, random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = GradientBoostingClassifier(n_estimators=100, random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('lr', clf1),
    ('rf', clf2),
    ('gb', clf3)
], voting='soft')

# Train model
print("Training voting classifier on SMOTE-balanced data...")
voting_clf.fit(X_resampled, y_resampled)

# Evaluate
y_pred = voting_clf.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model and columns
joblib.dump(voting_clf, 'scripts/voting_classifier.pkl')
joblib.dump(X.columns, 'scripts/model_columns.pkl')
print("\n✅ Model and column definitions saved.")
