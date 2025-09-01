#!/usr/bin/env python3
"""
Diabetes Detection ML - Demo Script

This script demonstrates the basic functionality of the diabetes detection system
without requiring the full pipeline to be run first.

Usage:
    python demo.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def create_demo_data(n_samples=500):
    """Create demo diabetes dataset."""
    np.random.seed(42)
    
    # Generate synthetic features
    pregnancies = np.random.poisson(3, n_samples)
    glucose = np.random.normal(120, 30, n_samples)
    blood_pressure = np.random.normal(70, 12, n_samples)
    skin_thickness = np.random.normal(20, 10, n_samples)
    insulin = np.random.normal(80, 40, n_samples)
    bmi = np.random.normal(32, 7, n_samples)
    diabetes_pedigree = np.random.exponential(0.5, n_samples)
    age = np.random.normal(33, 12, n_samples)
    
    # Create target variable based on some rules
    diabetes_risk = (
        (glucose - 120) / 30 * 0.3 +
        (bmi - 32) / 7 * 0.2 +
        (age - 33) / 12 * 0.15 +
        (insulin - 80) / 40 * 0.1 +
        np.random.normal(0, 0.2, n_samples)
    )
    
    outcome = (diabetes_risk > 0.5).astype(int)
    
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
        'Outcome': outcome
    })
    
    return data

def train_demo_model(data):
    """Train a simple Random Forest model for demo."""
    # Prepare features and target
    features = [col for col in data.columns if col != 'Outcome']
    X = data[features]
    y = data['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, X_test_scaled, y_test, y_pred, y_prob, accuracy, features

def predict_diabetes_risk(model, scaler, patient_data, feature_names):
    """Make diabetes prediction for a patient."""
    # Convert patient data to array
    patient_array = np.array([patient_data[feature] for feature in feature_names]).reshape(1, -1)
    
    # Scale the data
    patient_scaled = scaler.transform(patient_array)
    
    # Make prediction
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0, 1]
    
    return prediction, probability

def get_risk_level(probability):
    """Get risk level based on probability."""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

def get_recommendations(probability):
    """Get health recommendations based on risk level."""
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

def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(feature_importance_df))
    plt.barh(y_pos, feature_importance_df['Importance'], color='lightblue')
    plt.yticks(y_pos, feature_importance_df['Feature'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Diabetes Prediction')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """Main demo function."""
    print("🩺 Diabetes Detection ML - Demo")
    print("=" * 50)
    
    # Step 1: Create demo data
    print("\n📊 Creating demo dataset...")
    data = create_demo_data(500)
    print(f"✅ Created dataset with {len(data)} samples")
    print(f"   Diabetes rate: {data['Outcome'].mean():.1%}")
    
    # Step 2: Train model
    print("\n🤖 Training Random Forest model...")
    model, scaler, X_test, y_test, y_pred, y_prob, accuracy, features = train_demo_model(data)
    print(f"✅ Model trained successfully!")
    print(f"   Accuracy: {accuracy:.3f}")
    
    # Step 3: Model performance
    print("\n📈 Model Performance:")
    print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))
    
    # Step 4: Feature importance
    print("\n🔍 Feature Importance Analysis:")
    plot_feature_importance(model, features)
    
    # Step 5: Sample predictions
    print("\n🔮 Sample Patient Predictions:")
    print("-" * 40)
    
    sample_patients = [
        {
            'name': 'Patient A (Low Risk)',
            'Pregnancies': 1,
            'Glucose': 85,
            'BloodPressure': 66,
            'SkinThickness': 29,
            'Insulin': 0,
            'BMI': 26.6,
            'DiabetesPedigreeFunction': 0.351,
            'Age': 31
        },
        {
            'name': 'Patient B (Medium Risk)',
            'Pregnancies': 3,
            'Glucose': 140,
            'BloodPressure': 80,
            'SkinThickness': 35,
            'Insulin': 120,
            'BMI': 28.5,
            'DiabetesPedigreeFunction': 0.8,
            'Age': 45
        },
        {
            'name': 'Patient C (High Risk)',
            'Pregnancies': 2,
            'Glucose': 200,
            'BloodPressure': 95,
            'SkinThickness': 45,
            'Insulin': 200,
            'BMI': 35.2,
            'DiabetesPedigreeFunction': 1.2,
            'Age': 55
        }
    ]
    
    for patient in sample_patients:
        name = patient.pop('name')
        prediction, probability = predict_diabetes_risk(model, scaler, patient, features)
        risk_level = get_risk_level(probability)
        recommendations = get_recommendations(probability)
        
        print(f"\n👤 {name}")
        print(f"   Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
        print(f"   Risk Probability: {probability:.1%}")
        print(f"   Risk Level: {risk_level}")
        print(f"   Top Recommendations:")
        for i, rec in enumerate(recommendations[:2], 1):
            print(f"     {i}. {rec}")
        
        # Add name back for next iteration
        patient['name'] = name
    
    # Step 6: Interactive prediction
    print("\n" + "=" * 50)
    print("🎯 Interactive Prediction Demo")
    print("=" * 50)
    
    print("\nEnter patient information for prediction:")
    
    try:
        pregnancies = int(input("Number of Pregnancies (0-20): ") or "1")
        glucose = float(input("Glucose Level (mg/dL): ") or "120")
        blood_pressure = float(input("Blood Pressure (mm Hg): ") or "70")
        skin_thickness = float(input("Skin Thickness (mm): ") or "20")
        insulin = float(input("Insulin Level (mu U/ml): ") or "80")
        bmi = float(input("BMI (kg/m²): ") or "32")
        diabetes_pedigree = float(input("Diabetes Pedigree Function: ") or "0.5")
        age = int(input("Age (years): ") or "33")
        
        patient_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': diabetes_pedigree,
            'Age': age
        }
        
        prediction, probability = predict_diabetes_risk(model, scaler, patient_data, features)
        risk_level = get_risk_level(probability)
        recommendations = get_recommendations(probability)
        
        print(f"\n🔮 Prediction Results:")
        print(f"   Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
        print(f"   Risk Probability: {probability:.1%}")
        print(f"   Risk Level: {risk_level}")
        print(f"   Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"     {i}. {rec}")
            
    except ValueError:
        print("❌ Invalid input. Please enter numeric values.")
    except KeyboardInterrupt:
        print("\n\n👋 Demo completed!")
    
    print("\n🎉 Demo completed successfully!")
    print("\n💡 To run the full pipeline:")
    print("   python run_pipeline.py")
    print("\n🌐 To launch the web application:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()
