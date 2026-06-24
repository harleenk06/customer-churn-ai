import pandas as pd
import joblib
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("dataset/customer_churn_dataset.csv")

# -----------------------------
# 2. Clean data
# -----------------------------
df["InternetService"] = df["InternetService"].fillna("Unknown")
df.drop("CustomerID", axis=1, inplace=True)

# -----------------------------
# 3. Encode target
# -----------------------------
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# -----------------------------
# 4. One-hot encoding
# -----------------------------
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

# -----------------------------
# 5. Features / Target
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeature Count:", X.shape[1])
print("Features:", list(X.columns))

# Save feature names (IMPORTANT for Flask)
joblib.dump(list(X.columns), "feature_names.pkl")

# -----------------------------
# 6. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 7. Model
# -----------------------------
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

print("\nTraining complete ✔")

# -----------------------------
# 8. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 9. SAVE MODEL (CLEAN + SAFE)
# -----------------------------
if os.path.exists("churn_model.pkl"):
    os.remove("churn_model.pkl")

with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully ✔")
print("Feature names saved successfully ✔")