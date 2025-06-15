import pandas as pd
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load('disease_model.pkl')
le = joblib.load('label_encoder.pkl')

# Load the training dataset just to get correct feature columns
df = pd.read_csv('Training.csv')

# Drop unwanted columns (same as you did while training!)
if 'Unnamed: 133' in df.columns:
    df.drop(['Unnamed: 133'], axis=1, inplace=True)
if 'fluid_overload.1' in df.columns:
    df.drop(['fluid_overload.1'], axis=1, inplace=True)

# Extract correct feature names
feature_names = df.drop('prognosis', axis=1).columns.tolist()

# Simulate some selected symptoms (you can change these later!)
selected_symptoms = ['itching', 'skin_rash', 'vomiting']

# Create input vector: 1 for present, 0 for not present
input_vector = [1 if symptom in selected_symptoms else 0 for symptom in feature_names]

# Convert to array
input_array = np.array([input_vector])

# Predict
prediction = model.predict(input_array)[0]
predicted_disease = le.inverse_transform([prediction])[0]

print(f"Predicted Disease: {predicted_disease}")