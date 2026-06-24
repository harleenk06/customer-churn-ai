import pandas as pd

df = pd.read_csv("dataset/customer_churn_dataset.csv")

print("Before Cleaning")
print(df.shape)

# Fill missing values
df["InternetService"] = df["InternetService"].fillna("Unknown")

# Remove CustomerID
df.drop("CustomerID", axis=1, inplace=True)

print("\nAfter Cleaning")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nColumns")
print(df.columns)