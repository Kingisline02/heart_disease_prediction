import numpy as np
import pandas as pd

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
risk_level = np.where(risk_score < 45, 'Normal', np.where(risk_score <= 65, 'Moderate', 'High'))

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

# Convert DataFrame to NumPy array
data_array = df.to_numpy()

# Save the dataset to a CSV file (optional)
df.to_csv('health_risk_data.csv', index=False)

print("Dataset has been created and saved to 'health_risk_data.csv'.")
