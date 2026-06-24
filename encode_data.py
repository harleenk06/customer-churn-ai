import pandas as pd

# Load dataset
df = pd.read_csv("dataset/customer_churn_dataset.csv")

# Cleaning
df["InternetService"] = df["InternetService"].fillna("Unknown")
df.drop("CustomerID", axis=1, inplace=True)

# Convert target column
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# One-hot encode categorical columns
df = pd.get_dummies(
    df,
    columns=[
        "Gender",
        "ContractType",
        "InternetService",
        "PaymentMethod"
    ],
    drop_first=True
)

print(df.head())

print("\nShape:")
print(df.shape)