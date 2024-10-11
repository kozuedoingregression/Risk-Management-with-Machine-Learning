import joblib
import numpy as np
from flask import Flask, request, render_template

# Load the trained model (from the newly trained version with scikit-learn 1.5.2)
model = joblib.load("model/creditRisk.pkl")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Assuming the user sends form data with the feature values
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)  # Reshape to match model input shape

    # Make prediction using the model
    prediction = model.predict(features)

    # Convert prediction to a human-readable message (binary classification)
    result = "High Credit Risk" if prediction[0] else "Low Credit Risk"
    
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
