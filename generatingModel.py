import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Define the number of entries
num_entries = 300

# Generate random data
np.random.seed(42)  # For reproducibility
age = np.random.randint(20, 61, num_entries)
blood_pressure = np.random.randint(80, 181, num_entries)  # Assuming a reasonable range for blood pressure
height = np.random.uniform(150, 200, num_entries)  # Height in cm
weight = np.random.uniform(50, 120, num_entries)  # Weight in kg
smoking_status = np.random.randint(0, 2, num_entries)  # 0 for non-smoker, 1 for smoker

# Calculate BMI
bmi = weight / ((height / 100) ** 2)

# Calculate the risk score
risk_score = (0.1 * age) + (0.3 * blood_pressure) + (0.2 * bmi) + (0.3 * smoking_status)

# Determine the risk level based on the risk score
def get_risk_level(score):
    if score < 45:
        return 0  # Normal
    elif score <= 65:
        return 1  # Moderate
    else:
        return 2  # High

risk_level = np.array([get_risk_level(score) for score in risk_score])

# Create a DataFrame
data = {
    'Age': age,
    'Blood_Pressure': blood_pressure,
    'Height': height,
    'Weight': weight,
    'Smoking_Status': smoking_status,
    'Risk_Score': risk_score,
    'Risk_Level': risk_level
}

df = pd.DataFrame(data)

# Split the data
X = df[['Age', 'Blood_Pressure', 'Height', 'Weight', 'Smoking_Status']]
y = df['Risk_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'heart_disease_model.pkl')

print("Model trained and saved as 'heart_disease_model.pkl'.")
