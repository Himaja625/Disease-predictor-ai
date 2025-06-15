import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the dataset
df = pd.read_csv('Training.csv')

# Remove unwanted columns if present
for col in ['Unnamed: 133', 'fluid_overload.1']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Separate features and target
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Fill missing values in features
X = X.fillna(0)

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train the model
model = MultinomialNB()
model.fit(X, y_encoded)

# Save the model and label encoder
joblib.dump(model, 'disease_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("✅ Model and Label Encoder saved successfully!")