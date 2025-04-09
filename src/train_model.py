# app/train_model_debug.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

try:
    print("🚀 Loading dataset...")
    df = pd.read_csv('data/diabetes.csv')
    print("✅ Dataset loaded successfully!")

    print("🛠️ Preparing data...")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print("🤖 Training model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("✅ Model trained!")

    print("📁 Checking/creating 'models' folder...")
    models_path = "models"
    os.makedirs(models_path, exist_ok=True)
    print(f"📂 Folder ready at: {os.path.abspath(models_path)}")

    print("💾 Saving model and scaler...")
    joblib.dump(model, os.path.join(models_path, "diabetes_model.pkl"))
    joblib.dump(scaler, os.path.join(models_path, "scaler.pkl"))
    print("✅ Model and scaler saved successfully!")

except Exception as e:
    print("❌ ERROR:", e)

input("Press Enter to exit...")

