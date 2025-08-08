import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import config

st.set_page_config(layout="wide", page_title="Customer Churn Dashboard", page_icon="📉")

# Load model and data
model = joblib.load(config.VOTING_MODEL_PATH)
data = pd.read_csv(config.CLEANED_DATA_PATH)
data_with_ids = pd.read_csv(config.DATA_WITH_IDS_PATH)

# Add customerID back for visualization
data['CustomerID'] = data_with_ids['customerID']

# Predict churn probabilities
X = data.drop(['Churn', 'CustomerID'], axis=1)
probabilities = model.predict_proba(X)[:, 1]
data['Churn Probability'] = probabilities

st.title("📉 Customer Churn Prediction Dashboard")

# Layout split
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔝 Top 10 Customers Most Likely to Churn")
    st.dataframe(data[['CustomerID', 'Churn Probability']]
                 .sort_values(by='Churn Probability', ascending=False)
                 .head(10))

with col2:
    st.subheader("📊 Feature Importance (Random Forest)")
    rf_model = [est for name, est in model.estimators if isinstance(est, RandomForestClassifier)][0]
    rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_imp, ax_imp = plt.subplots()
    rf_importances.head(10).plot(kind='bar', ax=ax_imp, title='Top 10 Features')
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    st.pyplot(fig_imp)

with st.expander("📈 Churn Probability Distribution"):
    fig_dist, ax_dist = plt.subplots()
    sns.histplot(data['Churn Probability'], bins=20, kde=True, ax=ax_dist)
    plt.title("Distribution of Churn Probabilities")
    plt.xlabel("Churn Probability")
    plt.ylabel("Frequency")
    st.pyplot(fig_dist)

with st.expander("📃 Churn Rate by Contract Type"):
    contract_churn = data_with_ids.groupby('Contract')['Churn'].mean().reset_index()
    fig_contract, ax_contract = plt.subplots()
    sns.barplot(x='Contract', y='Churn', data=contract_churn, ax=ax_contract, palette='Blues_d')
    ax_contract.set_title("Churn Rate by Contract Type")
    ax_contract.set_xlabel("Contract Type")
    ax_contract.set_ylabel("Churn Rate")
    st.pyplot(fig_contract)

# Sidebar for custom prediction
with st.sidebar:
    st.header("🧪 Make a Prediction")

    # Select representative row for types
    sample_row = data.iloc[0]
    user_input = {}

    # Separate original feature names for UI (before one-hot)
    original_features = data_with_ids.drop(columns=['customerID', 'Churn']).columns

    for col in original_features:
        if data_with_ids[col].dtype == 'object':
            options = data_with_ids[col].dropna().unique().tolist()
            user_input[col] = st.selectbox(f"{col}:", options)
        else:
            min_val = float(data_with_ids[col].min())
            max_val = float(data_with_ids[col].max())
            mean_val = float(data_with_ids[col].mean())
            user_input[col] = st.slider(f"{col}:", min_val, max_val, mean_val)

    st.write("🧾 Raw User Input", user_input)

    if st.button("Predict Churn"):
        st.write("🔄 Button clicked. Running prediction...")

        # ✅ Load model
        try:
            model = joblib.load("scripts/voting_classifier.pkl")
            st.success("✅ Model loaded successfully.")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.stop()

        # ✅ Load model columns
        try:
            model_columns = joblib.load("scripts/model_columns.pkl")
            st.success("✅ Model columns loaded successfully.")
        except Exception as e:
            st.error(f"❌ Failed to load model columns: {e}")
            st.stop()

        # ✅ Transform raw user input to encoded format
        try:
            user_df = pd.DataFrame([user_input])
            user_encoded = pd.get_dummies(user_df)

            # Ensure all expected columns are present
            missing_cols = set(model_columns) - set(user_encoded.columns)
            for col in missing_cols:
                user_encoded[col] = 0

            # Reorder columns to match model training
            user_encoded = user_encoded[model_columns]

            st.write("🧾 Encoded User Input", user_encoded)
        except Exception as e:
            st.error(f"⚠️ Failed to encode user input: {e}")
            st.stop()

        # ✅ Make prediction
        try:
            prediction = model.predict(user_encoded)[0]
            probability = model.predict_proba(user_encoded)[0][1]
            st.markdown(f"### 🔮 Prediction: {'**Churn**' if prediction == 1 else '**No Churn**'}")
            st.markdown(f"### 📊 Churn Probability: **{probability:.2%}**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

