from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("model/creditRisks.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data from the form
    data = request.form
    age = int(data['age'])
    sex = int(data['sex'])
    housing = int(data['housing'])
    saving_accounts = int(data['saving_accounts'])
    checking_account = int(data['checking_account'])
    duration = int(data['duration'])
    purpose = int(data['purpose'])
    additional_feature = np.random.randint(1, 10)  # Random additional feature

    # Prepare the input array for prediction
    input_features = np.array([[age, sex, housing, saving_accounts, checking_account, duration, purpose, additional_feature]])
    
    # Make prediction
    prediction = model.predict(input_features)

    # Determine the prediction result
    prediction_result = "High Credit Risk" if prediction[0] else "Low Credit Risk"

    # Generate and save heatmap
    create_heatmap(prediction)

    return jsonify({'prediction': int(prediction[0]), 'result': prediction_result})

def create_heatmap(prediction):
    # Dummy data for heatmap (you can modify this based on your dataset)
    data = np.random.rand(10, 10)
    
    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, cmap='coolwarm', cbar=True)
    plt.title(f'Heatmap for Prediction: {int(prediction[0])}')
    
    # Save the heatmap
    plt.savefig('static/heatmap.png')
    plt.close()  # Close the figure to avoid display issues

if __name__ == '__main__':
    app.run(debug=True)
