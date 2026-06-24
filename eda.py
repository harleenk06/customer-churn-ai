import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset/customer_churn_dataset.csv")

# Graph 1
plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.show()

# Graph 2
plt.figure(figsize=(8,5))
sns.countplot(x="ContractType", hue="Churn", data=df)
plt.title("Contract Type vs Churn")
plt.show()

# Graph 3
plt.figure(figsize=(8,5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# Graph 4
plt.figure(figsize=(8,5))
sns.boxplot(x="Churn", y="Tenure", data=df)
plt.title("Tenure vs Churn")
plt.show()