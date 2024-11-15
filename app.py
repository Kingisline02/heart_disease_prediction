from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the model
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from the request
    
    # Extract the features from the JSON data
    age = data['age']
    blood_pressure = data['blood_pressure']
    height = data['height']
    weight = data['weight']
    smoking_status = data['smoking_status']
    
    # Convert the features to a numpy array
    input_data = np.array([age, blood_pressure, height, weight, smoking_status]).reshape(1, -1)
    
    # Get the prediction from the model
    prediction = model.predict(input_data)
    
    # Map prediction to risk level
    risk_level = {0: 'Normal', 1: 'Moderate', 2: 'High'}[prediction[0]]
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': risk_level})

if __name__ == '__main__':
    app.run(debug=True)
