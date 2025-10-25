import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('../data/diabetes.csv')

# Fill missing values if any (replace 0 with median for glucose, bp, etc.)
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols:
    data[col] = data[col].replace(0, data[col].median())

# Split features and labels
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessed data for training
import pickle
pickle.dump((X_train_scaled, X_test_scaled, y_train, y_test, scaler), open('../model/preprocessed.pkl', 'wb'))
