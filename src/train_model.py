# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Try block to handle any errors gracefully
try:
    # Load the dataset
    df = pd.read_csv('data/diabetes.csv')  # Ensure the path is correct

    # Prepare the data
    X = df.drop('Outcome', axis=1)  # Features: All columns except 'Outcome'
    y = df['Outcome']  # Target variable: 'Outcome' (Diabetic or not)

    # Scaling the data for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Normalize the features

    # Split the data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train the model using Logistic Regression
    model = LogisticRegression()  # Initialize the logistic regression model
    model.fit(X_train, y_train)  # Fit the model to the training data

    # Check and create the 'models' directory to save the model and scaler
    models_path = "models"
    os.makedirs(models_path, exist_ok=True)  # Create folder if it doesn't exist

    # Save the trained model and scaler as .pkl files
    joblib.dump(model, os.path.join(models_path, "diabetes_model.pkl"))  # Save the model
    joblib.dump(scaler, os.path.join(models_path, "scaler.pkl"))  # Save the scaler

except Exception as e:
    print("❌ ERROR:", e)




