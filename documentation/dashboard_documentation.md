
# dashboard.py Documentation

## Overview
This script provides an interactive Streamlit dashboard for visualizing customer churn predictions, 
making new predictions, and exploring key features related to churn.

---

## Dependencies
- **Libraries**:
  - `streamlit`: For creating the interactive dashboard.
  - `pandas`: For data manipulation.
  - `joblib`: For loading the trained model.
  - `seaborn` and `matplotlib`: For data visualization.
- **Input Files**:
  - `voting_classifier.pkl`: Trained model.
  - `cleaned_data.csv`: Processed dataset.
  - `data_with_ids.csv`: Dataset with `customerID` for visualization.

---

## Code Section Breakdown

### 1. Load Model and Data
```python
model_path = r'Y:\DataScience\CustomerChurn\models\voting_classifier.pkl'
data_path = r'Y:\DataScience\CustomerChurn\data\processed\cleaned_data.csv'
ids_path = r'Y:\DataScience\CustomerChurn\data\processed\data_with_ids.csv'
model = joblib.load(model_path)
data = pd.read_csv(data_path)
data_with_ids = pd.read_csv(ids_path)
```
- **What it does**: Loads the trained model and datasets.
- **Why it's done**: Provides the necessary data and model for the dashboard functionality.

---

### 2. Predict Churn Probabilities
```python
X = data.drop(['Churn', 'CustomerID'], axis=1)
probabilities = model.predict_proba(X)[:, 1]
data['Churn Probability'] = probabilities
```
- **What it does**: Computes churn probabilities using the trained model.
- **Why it's done**: Enables visualizations and insights into customers most likely to churn.

---

### 3. Dashboard Layout
#### Title and Overview
```python
st.title("Customer Churn Prediction Dashboard")
```
- **What it does**: Sets the main title for the dashboard.
- **Why it's done**: Provides a clear heading for the application.

#### Top Churn Probabilities
```python
st.header("Top 10 Customers Most Likely to Churn")
st.dataframe(data_with_ids[['customerID', 'Churn Probability']]
             .sort_values(by='Churn Probability', ascending=False)
             .head(10))
```
- **What it does**: Displays the top 10 customers with the highest churn probabilities.
- **Why it's done**: Allows users to focus on at-risk customers.

---

### 4. Visualizations
#### Churn Probability Distribution
```python
st.header("Churn Probability Distribution")
fig, ax = plt.subplots()
sns.histplot(data['Churn Probability'], bins=20, kde=True, ax=ax)
plt.title("Distribution of Churn Probabilities")
plt.xlabel("Churn Probability")
plt.ylabel("Frequency")
st.pyplot(fig)
```
- **What it does**: Visualizes the distribution of churn probabilities as a histogram.
- **Why it's done**: Provides insights into the overall risk distribution.

#### Feature Importance
```python
st.header("Feature Importance")
rf_model = [est for name, est in model.estimators if isinstance(est, RandomForestClassifier)][0]
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots()
rf_importances.head(10).plot(kind='bar', ax=ax, title='Top 10 Feature Importances (Random Forest)')
plt.ylabel("Importance")
plt.xlabel("Feature")
st.pyplot(fig)
```
- **What it does**: Displays the top 10 features influencing churn predictions.
- **Why it's done**: Highlights key factors driving churn.

#### Churn by Contract Type
```python
st.header("Churn by Contract Type")
contract_churn = data_with_ids.groupby('Contract', as_index=False)['Churn'].mean()
fig, ax = plt.subplots()
sns.barplot(x='Contract', y='Churn', data=contract_churn, ax=ax, palette='Blues_d')
ax.set_title("Churn Rate by Contract Type")
ax.set_xlabel("Contract Type")
ax.set_ylabel("Churn Rate")
st.pyplot(fig)
```
- **What it does**: Shows churn rates grouped by contract type.
- **Why it's done**: Identifies which contract types are most associated with churn.

---

### 5. Custom Predictions
```python
st.header("Make a Prediction")
user_input = {}
for col in X.columns:
    if data[col].dtype == 'object':
        user_input[col] = st.selectbox(f"{col}:", data[col].unique())
    else:
        user_input[col] = st.slider(f"{col}:", float(data[col].min()), float(data[col].max()), float(data[col].mean()))

if st.button("Predict Churn"):
    user_df = pd.DataFrame([user_input])
    prediction = model.predict(user_df)[0]
    probability = model.predict_proba(user_df)[0, 1]
    st.write(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    st.write(f"Churn Probability: {probability:.2f}")
```
- **What it does**: Allows users to input custom data and predict churn.
- **Why it's done**: Provides interactive functionality for what-if analyses.

---

## Summary
This dashboard allows users to explore churn predictions, visualize key metrics, and make custom predictions, providing valuable insights for customer retention strategies.
