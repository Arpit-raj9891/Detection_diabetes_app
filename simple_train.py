#!/usr/bin/env python3
"""
Simple training script for 3 ML models only
Trains Logistic Regression, Random Forest, and SVM models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

def create_synthetic_data(n_samples=500):
    """Create synthetic diabetes dataset"""
    np.random.seed(42)
    
    # Generate synthetic features
    pregnancies = np.random.randint(0, 10, n_samples)
    glucose = np.random.normal(120, 30, n_samples)
    blood_pressure = np.random.normal(70, 15, n_samples)
    skin_thickness = np.random.normal(30, 10, n_samples)
    insulin = np.random.normal(120, 80, n_samples)
    bmi = np.random.normal(32, 8, n_samples)
    diabetes_pedigree = np.random.exponential(0.5, n_samples)
    age = np.random.randint(20, 80, n_samples)
    
    # Create target variable based on some rules
    diabetes_risk = (
        (glucose > 140) * 0.3 +
        (bmi > 30) * 0.2 +
        (age > 50) * 0.15 +
        (blood_pressure > 80) * 0.1 +
        (insulin > 140) * 0.1 +
        np.random.random(n_samples) * 0.15
    )
    
    diabetes = (diabetes_risk > 0.5).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age,
        'Outcome': diabetes
    })
    
    return data

def train_models():
    """Train 3 ML models only"""
    print("🩺 Training 3 ML Models for Diabetes Detection")
    print("=" * 50)
    
    # Create data
    print("📊 Creating dataset...")
    data = create_synthetic_data(1000)
    print(f"✅ Dataset created with {len(data)} samples")
    print(f"   Diabetes rate: {data['Outcome'].mean():.1%}")
    
    # Prepare features and target
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = data[features]
    y = data['Outcome']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n🤖 Training 3 models...")
    
    # Logistic Regression
    print("   Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_score = lr_model.score(X_test_scaled, y_test)
    print(f"   ✅ Logistic Regression - Accuracy: {lr_score:.3f}")
    
    # Random Forest
    print("   Training Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train_scaled, y_train)
    rf_score = rf_model.score(X_test_scaled, y_test)
    print(f"   ✅ Random Forest - Accuracy: {rf_score:.3f}")
    
    # Support Vector Machine (SVM)
    print("   Training Support Vector Machine...")
    svm_model = SVC(random_state=42, probability=True)
    svm_model.fit(X_train_scaled, y_train)
    svm_score = svm_model.score(X_test_scaled, y_test)
    print(f"   ✅ SVM - Accuracy: {svm_score:.3f}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save models and scaler
    print("\n💾 Saving models...")
    joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    joblib.dump(svm_model, 'models/svm_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(imputer, 'models/imputer.pkl')
    joblib.dump(features, 'models/feature_names.pkl')
    
    # Save processed data
    joblib.dump({
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test
    }, 'models/processed_data.pkl')
    
    print("✅ All 3 models saved successfully!")
    print(f"   Models saved in: {os.path.abspath('models')}")
    
    # Test predictions
    print("\n🧪 Testing predictions...")
    sample_patient = np.array([[1, 140, 80, 35, 120, 32, 0.5, 45]])
    sample_scaled = scaler.transform(sample_patient)
    
    lr_pred = lr_model.predict(sample_scaled)[0]
    lr_prob = lr_model.predict_proba(sample_scaled)[0][1]
    rf_pred = rf_model.predict(sample_scaled)[0]
    rf_prob = rf_model.predict_proba(sample_scaled)[0][1]
    svm_pred = svm_model.predict(sample_scaled)[0]
    svm_prob = svm_model.predict_proba(sample_scaled)[0][1]
    
    print(f"   Sample patient prediction:")
    print(f"   Logistic Regression: {'Diabetic' if lr_pred else 'Non-diabetic'} (Risk: {lr_prob:.1%})")
    print(f"   Random Forest: {'Diabetic' if rf_pred else 'Non-diabetic'} (Risk: {rf_prob:.1%})")
    print(f"   SVM: {'Diabetic' if svm_pred else 'Non-diabetic'} (Risk: {svm_prob:.1%})")
    
    print("\n🎉 Training complete! You can now run the web app.")
    print("   Run: streamlit run app.py")

if __name__ == "__main__":
    train_models()
