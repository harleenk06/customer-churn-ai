import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset/customer_churn_dataset.csv")

# Cleaning
df["InternetService"] = df["InternetService"].fillna("Unknown")
df.drop("CustomerID", axis=1, inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# One-hot encoding
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

# Features
X = df.drop("Churn", axis=1)

# Target
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)