#!/usr/bin/env python3
"""
Diabetes Detection ML Pipeline Runner

This script runs the complete diabetes detection pipeline including:
1. Data preprocessing
2. Model training
3. Model evaluation
4. Web application setup

Usage:
    python run_pipeline.py
"""

import os
import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_preprocessing import DataPreprocessor
from model_training import DiabetesModelTrainer
from model_evaluation import ModelEvaluator
from utils import create_sample_data

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step_name):
    """Print a step indicator."""
    print(f"\n🔹 {step_name}...")

def main():
    """Run the complete diabetes detection pipeline."""
    print_header("DIABETES DETECTION ML PIPELINE")
    print("Starting complete pipeline execution...")
    
    start_time = time.time()
    
    try:
        # Step 1: Data Preprocessing
        print_step("Data Preprocessing")
        preprocessor = DataPreprocessor()
        
        # Load and process data
        data = preprocessor.load_data()
        preprocessor.explore_data()
        preprocessor.handle_missing_values()
        preprocessor.feature_engineering()
        preprocessor.feature_selection(k=8)
        preprocessor.scale_features()
        preprocessor.split_data()
        preprocessor.save_processed_data()
        
        print("✅ Data preprocessing completed successfully!")
        
        # Step 2: Model Training
        print_step("Model Training")
        trainer = DiabetesModelTrainer()
        trainer.load_data()
        trainer.initialize_models()
        trainer.train_models()
        trainer.tune_all_models()
        trainer.train_models()  # Retrain with tuned parameters
        trainer.save_models()
        
        print("✅ Model training completed successfully!")
        
        # Step 3: Model Evaluation
        print_step("Model Evaluation")
        evaluator = ModelEvaluator('models/best_model.pkl', 'data/processed_data.pkl')
        evaluator.load_test_data('data/processed_data.pkl')
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics()
        if metrics:
            print("\n📊 Model Performance Metrics:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Generate evaluation plots
        evaluator.plot_confusion_matrix(save_path='confusion_matrix.png')
        evaluator.plot_roc_curve(save_path='roc_curve.png')
        evaluator.plot_precision_recall_curve(save_path='pr_curve.png')
        evaluator.plot_feature_importance(save_path='feature_importance.png')
        
        # Generate detailed report
        evaluator.generate_detailed_report()
        
        print("✅ Model evaluation completed successfully!")
        
        # Step 4: Create sample predictions
        print_step("Sample Predictions")
        from utils import DiabetesPredictor
        
        predictor = DiabetesPredictor('models/best_model.pkl')
        
        # Sample patient data
        sample_patient = {
            'Pregnancies': 1,
            'Glucose': 85,
            'BloodPressure': 66,
            'SkinThickness': 29,
            'Insulin': 0,
            'BMI': 26.6,
            'DiabetesPedigreeFunction': 0.351,
            'Age': 31
        }
        
        prediction, probability = predictor.predict_single(**sample_patient)
        risk_level = predictor.get_risk_level(probability)
        
        print(f"\n🔮 Sample Prediction:")
        print(f"   Patient Data: {sample_patient}")
        print(f"   Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
        print(f"   Risk Probability: {probability:.1%}")
        print(f"   Risk Level: {risk_level}")
        
        print("✅ Sample predictions completed successfully!")
        
        # Step 5: Summary
        end_time = time.time()
        execution_time = end_time - start_time
        
        print_header("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
        print(f"📁 Generated files:")
        print(f"   - models/best_model.pkl (Best trained model)")
        print(f"   - data/processed_data.pkl (Processed dataset)")
        print(f"   - confusion_matrix.png (Confusion matrix plot)")
        print(f"   - roc_curve.png (ROC curve plot)")
        print(f"   - pr_curve.png (Precision-Recall curve plot)")
        print(f"   - feature_importance.png (Feature importance plot)")
        print(f"   - model_evaluation_report.txt (Detailed evaluation report)")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Run the web application: streamlit run app.py")
        print(f"   2. Open the Jupyter notebook: jupyter notebook notebooks/diabetes_analysis.ipynb")
        print(f"   3. Explore the generated plots and reports")
        
        print("\n🎉 Pipeline execution completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        print("Please check the error message and try again.")
        return False
    
    return True

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
        'plotly', 'streamlit', 'joblib', 'xgboost', 'lightgbm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data', 'models', 'notebooks']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directories created/verified!")

if __name__ == "__main__":
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("📁 Creating directories...")
    create_directories()
    
    print("🚀 Starting pipeline...")
    success = main()
    
    if success:
        print("\n🎯 Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Pipeline failed!")
        sys.exit(1)
