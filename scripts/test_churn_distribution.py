import pandas as pd

# ✔️ Updated path to reflect your folder structure
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ✔️ Show churn distribution
print(df['Churn'].value_counts(normalize=True))
