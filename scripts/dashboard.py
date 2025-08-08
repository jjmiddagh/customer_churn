import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

# --- Load and preprocess data ---
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

df_original = df.copy()  # Preserve customerID for dashboard
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# One-hot encode and align to numeric-only
X = pd.get_dummies(df.drop(columns=["customerID", "Churn"]), drop_first=True)
X = X.loc[:, X.dtypes != 'object']  # Keep only numeric columns just in case
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# --- Train model ---
clf1 = LogisticRegression(max_iter=1000)
clf2 = RandomForestClassifier()
clf3 = GradientBoostingClassifier()

model = VotingClassifier(
    estimators=[("lr", clf1), ("rf", clf2), ("gb", clf3)],
    voting="soft"
)
model.fit(X_train, y_train)

# --- Prepare dashboard dataset ---
df_with_ids = df_original.iloc[X_test.index].copy()
df_with_ids["Churn Probability"] = model.predict_proba(X_test)[:, 1]
df_with_ids["Churn"] = y_test.reset_index(drop=True)

# --- Streamlit App ---
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📉 Customer Churn Prediction Dashboard")
st.markdown("Explore predicted churn risk across your customer base.")

# Sidebar filters
st.sidebar.header("Filter by Churn Probability")
prob_threshold = st.sidebar.slider("Minimum churn probability", 0.0, 1.0, 0.5, 0.01)
filtered_data = df_with_ids[df_with_ids["Churn Probability"] >= prob_threshold]

# Show filtered customer list
st.subheader(f"Customers with Churn Probability ≥ {prob_threshold:.2f}")
st.write(filtered_data[["customerID", "Churn Probability", "Churn"]].sort_values("Churn Probability", ascending=False))

# What-if tool
st.sidebar.header("Simulate a New Customer")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
monthly_charges = st.sidebar.slider("Monthly Charges", 0, 150, 70)
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

if st.sidebar.button("Predict Churn"):
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "MonthlyCharges": monthly_charges,
        "tenure": tenure,
    }
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)
    prob = model.predict_proba(input_df)[0][1]
    st.sidebar.markdown(f"### 🔮 Churn Probability: **{prob:.2%}**")

