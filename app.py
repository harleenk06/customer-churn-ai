from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib

app = Flask(__name__)

# Load model + features
model = pickle.load(open("churn_model.pkl", "rb"))
features = joblib.load("feature_names.pkl")


# -----------------------------
# Build correct input vector
# -----------------------------
def create_input(data):
    input_dict = {f: 0 for f in features}

    # numeric features
    input_dict["Age"] = data["Age"]
    input_dict["Tenure"] = data["Tenure"]
    input_dict["MonthlyCharges"] = data["MonthlyCharges"]
    input_dict["SupportCalls"] = data["SupportCalls"]

    # Gender
    if data["Gender"] == "Male":
        input_dict["Gender_Male"] = 1

    # ContractType
    if data["ContractType"] == "One Year":
        input_dict["ContractType_One Year"] = 1
    elif data["ContractType"] == "Two Year":
        input_dict["ContractType_Two Year"] = 1

    # InternetService
    if data["InternetService"] == "Fiber":
        input_dict["InternetService_Fiber"] = 1
    elif data["InternetService"] == "Unknown":
        input_dict["InternetService_Unknown"] = 1

    # PaymentMethod
    if data["PaymentMethod"] == "Credit Card":
        input_dict["PaymentMethod_Credit Card"] = 1
    elif data["PaymentMethod"] == "E-Wallet":
        input_dict["PaymentMethod_E-Wallet"] = 1
    elif data["PaymentMethod"] == "UPI":
        input_dict["PaymentMethod_UPI"] = 1

    return np.array([list(input_dict.values())])


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data (MUST match HTML names)
        data = {
            "Age": int(request.form["Age"]),
            "Tenure": int(request.form["Tenure"]),
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "SupportCalls": int(request.form["SupportCalls"]),
            "Gender": request.form["Gender"],
            "ContractType": request.form["ContractType"],
            "InternetService": request.form["InternetService"],
            "PaymentMethod": request.form["PaymentMethod"]
        }

        # Convert to model input
        input_data = create_input(data)

        # Prediction
        prediction = model.predict(input_data)[0]

        # Probability (confidence)
        confidence = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0]
            confidence = round(float(max(prob)) * 100, 2)

        # Result
        if prediction == 1:
            result = "🔴 Customer Will CHURN"
        else:
            result = "🟢 Customer Will STAY"

        return render_template(
            "index.html",
            prediction_text=result,
            confidence=confidence
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            confidence=None
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)