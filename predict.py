import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('../model/diabetes_model.pkl', 'rb'))
_, _, _, _, scaler = pickle.load(open('../model/preprocessed.pkl', 'rb'))

def predict_diabetes(input_data):
    input_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    prediction = model.predict(input_scaled)
    return "Diabetes" if prediction[0] == 1 else "No Diabetes"

# Example prediction
sample = [6, 148, 72, 35, 0, 33.6, 0.627, 50]  # Example row
print(predict_diabetes(sample))
