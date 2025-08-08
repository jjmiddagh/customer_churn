
# churn_02_modeling.py Documentation

## Overview
This script handles the machine learning tasks, including training, tuning, and evaluating multiple models for predicting customer churn. 
It also saves the best-performing model for later use.

---

## Dependencies
- **Libraries**:
  - `pandas`: For data manipulation.
  - `numpy`: For numerical operations.
  - `sklearn`: For machine learning models and evaluation metrics.
  - `imbalanced-learn`: For handling imbalanced datasets (SMOTE).
  - `joblib`: For saving the trained model.
- **Input File**: `cleaned_data.csv`
- **Output File**: `voting_classifier.pkl`

---

## Code Section Breakdown

### 1. Load the Dataset
```python
data = pd.read_csv(r'Y:\DataScience\CustomerChurn\data\processed\cleaned_data.csv')
```
- **What it does**: Reads the preprocessed dataset.
- **Why it's done**: Provides the cleaned data for model training.

---

### 2. Split Data into Features and Target
```python
X = data.drop('Churn', axis=1)
y = data['Churn']
```
- **What it does**: Splits the dataset into feature variables (`X`) and target variable (`y`).
- **Why it's done**: Separates inputs and outputs for model training.

---

### 3. Handle Class Imbalance with SMOTE
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```
- **What it does**: Balances the dataset by generating synthetic samples for the minority class.
- **Why it's done**: Prevents bias in the model caused by class imbalance.

---

### 4. Split Data into Training and Testing Sets
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
```
- **What it does**: Divides the data into training and testing sets.
- **Why it's done**: Ensures the model is evaluated on unseen data.

---

### 5. Train Logistic Regression and Random Forest Models
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

lr = LogisticRegression(random_state=42)
rf = RandomForestClassifier(random_state=42)
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
```
- **What it does**: Trains two baseline models: Logistic Regression and Random Forest.
- **Why it's done**: Provides initial benchmarks for churn prediction.

---

### 6. Hyperparameter Tuning for Random Forest
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)
```
- **What it does**: Tunes the Random Forest model using grid search.
- **Why it's done**: Optimizes model performance by finding the best hyperparameters.

---

### 7. Evaluate Models
```python
from sklearn.metrics import classification_report, accuracy_score

lr_predictions = lr.predict(X_test)
rf_predictions = grid_search.best_estimator_.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
```
- **What it does**: Evaluates model accuracy and generates classification reports.
- **Why it's done**: Measures model performance on the test set.

---

### 8. Combine Models with Voting Classifier
```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[('lr', lr), ('rf', grid_search.best_estimator_)], voting='soft')
voting_clf.fit(X_train, y_train)
```
- **What it does**: Combines Logistic Regression and Random Forest into an ensemble model.
- **Why it's done**: Leverages the strengths of both models to improve performance.

---

### 9. Save the Best Model
```python
joblib.dump(voting_clf, r'Y:\DataScience\CustomerChurn\models\voting_classifier.pkl')
```
- **What it does**: Saves the trained Voting Classifier model.
- **Why it's done**: Ensures the model can be reused without retraining.

---

## Summary
This script builds, evaluates, tunes, and saves models for predicting customer churn, making it a critical step in the pipeline.
