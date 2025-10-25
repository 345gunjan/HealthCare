import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
X_train, X_test, y_train, y_test, scaler = pickle.load(open('../model/preprocessed.pkl', 'rb'))

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save trained model
pickle.dump(model, open('../model/diabetes_model.pkl', 'wb'))
