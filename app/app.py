# customer_churn/app/app.py
import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

st.set_page_config(page_title="Customer Churn – Interactive Demo", layout="wide")
# --- Usage Guide (appears above the Customer Churn section) ---
def render_usage_guide():
    with st.expander("How to use this tool", expanded=False):
        st.markdown(
            """
**What this is.** An interactive churn demo on the Telco dataset. It shows a real
ML pipeline (scaler + one-hot encoding → Logistic Regression) and lets you explore
risk by probability threshold and simulate a new customer.

**How to use**
1. **Set the minimum churn probability** (left panel). The table shows test-set customers with predicted probability ≥ the threshold.
2. **Simulate a new customer** using the five inputs (Gender, SeniorCitizen, Partner, MonthlyCharges, Tenure), then click **Predict Churn** to see their probability.
3. **Scan the flagged customers** in the table (sorted high→low) to decide who to target for retention.

**Interpreting metrics**
- **ROC-AUC**: ranking quality (1.0 is perfect; 0.5 is random).
- **F1 / Precision / Recall**: computed on the test set at a 0.50 default threshold.
  Tune the threshold to trade precision (fewer false positives) vs recall (catch more churners).

**Under the hood**
- Train/test split: 80/20, stratified, `random_state=42`.
- Preprocessing: `StandardScaler` (numeric), `OneHotEncoder(handle_unknown="ignore")` (categorical).
- Model: `LogisticRegression(solver="liblinear", max_iter=2000, class_weight="balanced")`.

**Data & scope**
- Expects **`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`** with target column **`Churn`**.
- This is a demo; no cost curve, calibration, or fairness audit yet.

**Next steps (nice enhancements)**
- Probability **calibration** (Platt/Isotonic) + **cost-based threshold**.
- **Explainability** (SHAP) for top-risk customers.
- **Bias / stability checks** and a lightweight **model card**.
            """
        )

# ----------------------------
# 1) Data loading
# ----------------------------
POSSIBLE_PATHS = [
    "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

def load_df():
    for p in POSSIBLE_PATHS:
        if os.path.exists(p):
            return pd.read_csv(p)
    # Fallback so the app still runs for thumbnails/demo
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=7, n_redundant=2,
        weights=[0.73, 0.27], random_state=42
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    df["Churn"] = y
    return df

df = load_df()

# Pick target column
TARGET_CANDIDATES = ["Churn", "churn", "Exited", "target", "Target"]
target_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
if target_col is None:
    st.error("No target column found. Expected one of: " + ", ".join(TARGET_CANDIDATES))
    st.stop()

# Normalize y if strings
y = df[target_col]
if y.dtype == "object":
    y = y.astype(str).str.lower().map({"yes": 1, "no": 0, "true": 1, "false": 0}).fillna(0).astype(int)

X = df.drop(columns=[target_col])

# ----------------------------
# 2) Train/test + pipeline
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

clf = Pipeline(steps=[
    ("pre", pre),
    ("logit", LogisticRegression(
        solver="liblinear",        # stable with sparse OHE
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    )),
])

clf.fit(X_train, y_train)

# Test-set metrics
proba_test = clf.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= 0.5).astype(int)
roc = roc_auc_score(y_test, proba_test)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred_test, average="binary", zero_division=0)

# Defaults for simulation inputs
NUM_DEFAULTS = {c: float(X_train[c].median()) for c in num_cols}
CAT_DEFAULTS = {c: X_train[c].mode().iloc[0] for c in cat_cols}

# Helper to build a single-row input matching training columns
def build_input_row(user_overrides: dict) -> pd.DataFrame:
    row = {}
    # start from defaults
    row.update(NUM_DEFAULTS)
    row.update(CAT_DEFAULTS)
    # apply user overrides
    row.update({k: v for k, v in user_overrides.items() if k in X_train.columns})
    # ensure all columns exist
    for col in X_train.columns:
        if col not in row:
            row[col] = NUM_DEFAULTS.get(col, CAT_DEFAULTS.get(col, 0))
    return pd.DataFrame([row])[X_train.columns]

# ----------------------------
# 3) UI
# ----------------------------
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Filter by Churn Probability")
    threshold = st.slider("Minimum churn probability", 0.0, 1.0, 0.50, 0.01)

    st.markdown("### Simulate a New Customer")

    # Common Telco columns if present; otherwise fall back to generic choices
    gender_col = next((c for c in ["gender", "Gender"] if c in X_train.columns), None)
    senior_col = next((c for c in ["SeniorCitizen", "senior_citizen", "is_senior"] if c in X_train.columns), None)
    partner_col = next((c for c in ["Partner", "HasPartner", "partner"] if c in X_train.columns), None)
    monthly_col = next((c for c in ["MonthlyCharges", "monthly_charges", "charges_monthly"] if c in X_train.columns), None)
    tenure_col  = next((c for c in ["tenure", "Tenure"] if c in X_train.columns), None)

    user = {}

    if gender_col:
        gender_choices = sorted(X_train[gender_col].dropna().astype(str).unique().tolist())
        user[gender_col] = st.selectbox("Gender", gender_choices, index=0)

    if senior_col:
        senior_choices = sorted(X_train[senior_col].dropna().astype(str).unique().tolist())
        user[senior_col] = st.selectbox("Senior Citizen", senior_choices, index=0)

    if partner_col:
        partner_choices = sorted(X_train[partner_col].dropna().astype(str).unique().tolist())
        user[partner_col] = st.selectbox("Has Partner?", partner_choices, index=0)

    if monthly_col:
        lo, hi = float(X_train[monthly_col].min()), float(X_train[monthly_col].max())
        default = float(NUM_DEFAULTS.get(monthly_col, (lo + hi) / 2))
        user[monthly_col] = st.slider("Monthly Charges", lo, hi, default)

    if tenure_col:
        lo, hi = int(X_train[tenure_col].min()), int(X_train[tenure_col].max())
        default = int(NUM_DEFAULTS.get(tenure_col, max(1, lo)))
        user[tenure_col] = st.slider("Tenure (months)", lo, hi, default)

    predict_click = st.button("Predict Churn", use_container_width=False)

with right:
    render_usage_guide()          # <-- add this line
    st.title("Customer Churn")
    st.caption("Interactive demo • Feature engineering + Logistic Regression (liblinear)")
      # Metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{roc:.3f}")
    c2.metric("F1", f"{f1:.3f}")
    c3.metric("Recall", f"{rec:.3f}")

    # Test-set table filtered by threshold
    df_test_view = df.loc[X_test.index].copy()   # label-safe (avoids iloc bug)
    df_test_view["churn_probability"] = proba_test
    filtered = df_test_view[df_test_view["churn_probability"] >= threshold] \
        .sort_values("churn_probability", ascending=False)

    st.subheader(f"Customers with p(churn) ≥ {threshold:.2f}")
    st.dataframe(filtered.head(20), use_container_width=True)

    if predict_click:
        st.subheader("Simulated Customer Prediction")
        input_row = build_input_row(user)
        p = float(clf.predict_proba(input_row)[0, 1])
        st.write(f"**Predicted churn probability:** {p:.3f}")
        st.progress(min(max(p, 0.0), 1.0))

st.info("Tip: adjust the threshold to see which customers would be flagged for outreach; use the controls to simulate a new customer profile.")
