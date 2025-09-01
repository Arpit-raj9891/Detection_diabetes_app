#!/usr/bin/env python3
"""
Diabetes Detection ML - Demo Script
Simplified demonstration with 3 models only
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

def train_models(X_train, X_test, y_train, y_test):
    """Train 3 ML models"""
    models = {}
    
    # Logistic Regression
    print("   Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_score = lr_model.score(X_test, y_test)
    models['Logistic Regression'] = lr_model
    print(f"   ✅ Logistic Regression - Accuracy: {lr_score:.3f}")
    
    # Random Forest
    print("   Training Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_score = rf_model.score(X_test, y_test)
    models['Random Forest'] = rf_model
    print(f"   ✅ Random Forest - Accuracy: {rf_score:.3f}")
    
    # Support Vector Machine
    print("   Training Support Vector Machine...")
    svm_model = SVC(random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    svm_score = svm_model.score(X_test, y_test)
    models['SVM'] = svm_model
    print(f"   ✅ SVM - Accuracy: {svm_score:.3f}")
    
    return models

def plot_feature_importance(models, feature_names):
    """Plot feature importance for tree-based models"""
    plt.figure(figsize=(12, 6))
    
    # Random Forest feature importance
    rf_importance = models['Random Forest'].feature_importances_
    
    # Sort features by importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_importance
    }).sort_values('Importance', ascending=True)
    
    plt.barh(range(len(feature_importance_df)), feature_importance_df['Importance'])
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.show()

def predict_diabetes_risk(model, scaler, patient_data, features):
    """Make prediction for a patient"""
    # Prepare patient data
    patient_df = pd.DataFrame([patient_data])
    patient_df = patient_df[features]
    
    # Scale features
    patient_scaled = scaler.transform(patient_df)
    
    # Make prediction
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]
    
    return prediction, probability

def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

def get_recommendations(probability):
    """Get health recommendations based on risk"""
    if probability < 0.3:
        return [
            "Continue with healthy lifestyle",
            "Regular check-ups recommended",
            "Monitor blood sugar levels occasionally"
        ]
    elif probability < 0.6:
        return [
            "Consult with healthcare provider",
            "Monitor blood sugar levels regularly",
            "Consider lifestyle modifications",
            "Regular exercise recommended"
        ]
    else:
        return [
            "Immediate medical consultation required",
            "Frequent blood sugar monitoring",
            "Strict dietary control",
            "Regular exercise program",
            "Consider medication if prescribed"
        ]

def main():
    """Main demo function"""
    print("🩺 Diabetes Detection ML - Demo")
    print("=" * 50)
    
    # Create demo dataset
    print("📊 Creating demo dataset...")
    data = create_synthetic_data(500)
    print(f"✅ Created dataset with {len(data)} samples")
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
    models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Evaluate models
    print("\n📈 Model Performance:")
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   {name}: {accuracy:.3f}")
    
    # Feature importance
    print("\n🔍 Feature Importance Analysis:")
    plot_feature_importance(models, features)
    
    # Interactive predictions
    print("\n" + "=" * 50)
    print("🎯 INTERACTIVE PREDICTIONS")
    print("=" * 50)
    
    # Sample patient predictions
    sample_patients = [
        {
            'Pregnancies': 1, 'Glucose': 85, 'BloodPressure': 66,
            'SkinThickness': 29, 'Insulin': 0, 'BMI': 26.6,
            'DiabetesPedigreeFunction': 0.351, 'Age': 31
        },
        {
            'Pregnancies': 3, 'Glucose': 150, 'BloodPressure': 85,
            'SkinThickness': 35, 'Insulin': 150, 'BMI': 35.2,
            'DiabetesPedigreeFunction': 0.8, 'Age': 45
        }
    ]
    
    print("\n📋 Sample Patient Predictions:")
    for i, patient in enumerate(sample_patients, 1):
        print(f"\n   Patient {i}:")
        for name, model in models.items():
            prediction, probability = predict_diabetes_risk(model, scaler, patient, features)
            risk_level = get_risk_level(probability)
            print(f"     {name}: {'Diabetic' if prediction else 'Non-diabetic'} "
                  f"(Risk: {probability:.1%}, Level: {risk_level})")
    
    # User input prediction
    print("\n" + "=" * 50)
    print("👤 ENTER PATIENT INFORMATION")
    print("=" * 50)
    
    try:
        print("\nEnter patient information for prediction:")
        pregnancies = int(input("Number of Pregnancies (0-10): "))
        glucose = float(input("Glucose Level (mg/dL): "))
        blood_pressure = float(input("Blood Pressure (mm Hg): "))
        skin_thickness = float(input("Skin Thickness (mm): "))
        insulin = float(input("Insulin Level (mu U/ml): "))
        bmi = float(input("BMI (kg/m²): "))
        diabetes_pedigree = float(input("Diabetes Pedigree Function: "))
        age = int(input("Age (years): "))
        
        patient_data = {
            'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
            'DiabetesPedigreeFunction': diabetes_pedigree, 'Age': age
        }
        
        print("\n🔮 PREDICTION RESULTS:")
        print("=" * 30)
        
        for name, model in models.items():
            prediction, probability = predict_diabetes_risk(model, scaler, patient_data, features)
            risk_level = get_risk_level(probability)
            print(f"\n{name}:")
            print(f"  Prediction: {'Diabetic' if prediction else 'Non-diabetic'}")
            print(f"  Risk Probability: {probability:.1%}")
            print(f"  Risk Level: {risk_level}")
            
            recommendations = get_recommendations(probability)
            print(f"  Recommendations:")
            for rec in recommendations:
                print(f"    • {rec}")
        
    except (ValueError, KeyboardInterrupt):
        print("\n⚠️ Input cancelled or invalid. Demo completed.")
    
    print("\n🎉 Demo completed! Run 'streamlit run app.py' for the full web application.")

if __name__ == "__main__":
    main()
