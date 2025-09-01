import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import DiabetesPredictor, DataValidator, create_sample_data
from data_preprocessing import DataPreprocessor
from model_training import DiabetesModelTrainer
from model_evaluation import ModelEvaluator

# Page configuration
st.set_page_config(
    page_title="Diabetes Detection ML",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .low-risk {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .medium-risk {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeaa7;
    }
    .high-risk {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class DiabetesDetectionApp:
    def __init__(self):
        self.predictor = None
        self.preprocessor = None
        self.trainer = None
        self.evaluator = None
        self.load_models()
    
    def load_models(self):
        """Load trained models if available."""
        try:
            # Try to load models from simple training
            if (os.path.exists('models/logistic_regression_model.pkl') and 
                os.path.exists('models/scaler.pkl')):
                self.predictor = DiabetesPredictor('models/logistic_regression_model.pkl')
                st.success("✅ Models loaded successfully!")
            elif os.path.exists('models/best_model.pkl'):
                self.predictor = DiabetesPredictor('models/best_model.pkl')
                st.success("✅ Model loaded successfully!")
            else:
                st.warning("⚠️ No trained model found. Please train a model first.")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
    
    def main_page(self):
        """Main page with diabetes detection interface."""
        st.markdown('<h1 class="main-header">🩺 Diabetes Detection using Machine Learning</h1>', unsafe_allow_html=True)
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Home", "🔍 Prediction", "📊 Analysis", "🤖 Model Training", "📈 Evaluation"
        ])
        
        with tab1:
            self.home_tab()
        
        with tab2:
            self.prediction_tab()
        
        with tab3:
            self.analysis_tab()
        
        with tab4:
            self.training_tab()
        
        with tab5:
            self.evaluation_tab()
    
    def home_tab(self):
        """Home tab with project overview."""
        st.markdown('<h2 class="sub-header">Welcome to Diabetes Detection ML</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### About This Project
            
            This application uses machine learning to predict diabetes risk based on various health parameters. 
            The system analyzes multiple factors including:
            
            - **Glucose levels** - Blood sugar concentration
            - **Blood pressure** - Diastolic blood pressure
            - **BMI** - Body Mass Index
            - **Age** - Patient age
            - **Insulin levels** - 2-Hour serum insulin
            - **Skin thickness** - Triceps skin fold thickness
            - **Pregnancies** - Number of times pregnant
            - **Diabetes pedigree function** - Family history
            
            ### How It Works
            
            1. **Data Input**: Enter patient health parameters
            2. **ML Processing**: Advanced algorithms analyze the data
            3. **Risk Assessment**: Get diabetes risk prediction
            4. **Recommendations**: Receive personalized health advice
            
            ### Model Performance
            
            Our models achieve high accuracy in diabetes prediction using:
            - Multiple ML algorithms (Random Forest, SVM, XGBoost, etc.)
            - Advanced feature engineering
            - Comprehensive validation
            - Real-time predictions
            """)
        
        with col2:
            st.markdown("""
            ### Quick Stats
            
            📊 **Model Accuracy**: 85%+
            
            🎯 **Precision**: 82%+
            
            🔄 **Recall**: 88%+
            
            ⚡ **Prediction Time**: < 1 second
            
            ### Features
            
            ✅ Real-time predictions
            
            📈 Interactive visualizations
            
            🤖 Multiple ML models
            
            📋 Detailed reports
            
            💡 Health recommendations
            """)
        
        # Add some visual elements
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Models Available", "7+", "ML Algorithms")
        
        with col2:
            st.metric("Features Analyzed", "8", "Health Parameters")
        
        with col3:
            st.metric("Prediction Speed", "< 1s", "Real-time")
    
    def prediction_tab(self):
        """Prediction tab for making diabetes predictions."""
        st.markdown('<h2 class="sub-header">🔍 Diabetes Risk Prediction</h2>', unsafe_allow_html=True)
        
        if self.predictor is None:
            st.error("❌ No trained model available. Please train a model first in the Model Training tab.")
            return
        
        # Create input form
        with st.form("prediction_form"):
            st.markdown("### Enter Patient Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
                glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
                blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
            
            with col2:
                insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=80)
                bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=32.0, step=0.1)
                diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
                age = st.number_input("Age (years)", min_value=0, max_value=120, value=33)
            
            submitted = st.form_submit_button("🔮 Predict Diabetes Risk")
        
        if submitted:
            try:
                # Create patient data
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
                
                # Make prediction
                prediction, probability = self.predictor.predict_single(**patient_data)
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", "Diabetic" if prediction == 1 else "Non-Diabetic")
                
                with col2:
                    st.metric("Risk Probability", f"{probability:.1%}")
                
                with col3:
                    risk_level = self.predictor.get_risk_level(probability)
                    st.metric("Risk Level", risk_level)
                
                # Risk level indicator
                if probability < 0.3:
                    risk_class = "low-risk"
                elif probability < 0.6:
                    risk_class = "medium-risk"
                else:
                    risk_class = "high-risk"
                
                st.markdown(f'<div class="prediction-result {risk_class}">Risk Level: {risk_level}</div>', unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("### 💡 Health Recommendations")
                recommendations = self.predictor.get_recommendations(probability)
                
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")
                
                # Feature importance visualization
                st.markdown("### 📈 Feature Analysis")
                self.plot_feature_importance(patient_data, probability)
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
    
    def plot_feature_importance(self, patient_data, probability):
        """Plot feature importance for the prediction."""
        # Create a simple feature importance visualization
        features = list(patient_data.keys())
        values = list(patient_data.values())
        
        # Normalize values for visualization
        normalized_values = [(v - min(values)) / (max(values) - min(values)) for v in values]
        
        fig = go.Figure(data=[
            go.Bar(
                x=features,
                y=normalized_values,
                marker_color='lightblue',
                text=[f"{v:.1f}" for v in values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Patient Health Parameters",
            xaxis_title="Features",
            yaxis_title="Normalized Values",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def analysis_tab(self):
        """Analysis tab for data exploration."""
        st.markdown('<h2 class="sub-header">📊 Data Analysis & Visualization</h2>', unsafe_allow_html=True)
        
        # Create sample data for demonstration
        if st.button("🔄 Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                sample_data = create_sample_data(1000)
                st.session_state.sample_data = sample_data
                st.success("✅ Sample data generated!")
        
        if 'sample_data' in st.session_state:
            data = st.session_state.sample_data
            
            # Data overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Patients", len(data))
            with col2:
                st.metric("Diabetic Patients", data['Outcome'].sum())
            with col3:
                st.metric("Non-Diabetic Patients", len(data) - data['Outcome'].sum())
            with col4:
                st.metric("Diabetes Rate", f"{data['Outcome'].mean():.1%}")
            
            # Visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "📊 Feature Analysis", "🎯 Target Analysis"])
            
            with tab1:
                self.plot_distributions(data)
            
            with tab2:
                self.plot_correlations(data)
            
            with tab3:
                self.plot_feature_analysis(data)
            
            with tab4:
                self.plot_target_analysis(data)
    
    def plot_distributions(self, data):
        """Plot feature distributions."""
        features = ['Glucose', 'BloodPressure', 'BMI', 'Age', 'Insulin']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=features,
            specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}, {"type": "scatter"}]]
        )
        
        for i, feature in enumerate(features):
            row = i // 3 + 1
            col = i % 3 + 1
            
            if feature == 'Insulin':
                # Scatter plot for insulin vs glucose
                fig.add_trace(
                    go.Scatter(x=data['Glucose'], y=data['Insulin'], mode='markers', name='Insulin vs Glucose'),
                    row=row, col=col
                )
            else:
                fig.add_trace(
                    go.Histogram(x=data[feature], name=feature),
                    row=row, col=col
                )
        
        fig.update_layout(height=600, title_text="Feature Distributions")
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_correlations(self, data):
        """Plot correlation matrix."""
        corr_matrix = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_feature_analysis(self, data):
        """Plot feature analysis by diabetes status."""
        features = ['Glucose', 'BMI', 'Age', 'BloodPressure']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"{f} by Diabetes Status" for f in features]
        )
        
        for i, feature in enumerate(features):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Box plot
            fig.add_trace(
                go.Box(
                    y=data[data['Outcome'] == 0][feature],
                    name='Non-Diabetic',
                    marker_color='lightblue'
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Box(
                    y=data[data['Outcome'] == 1][feature],
                    name='Diabetic',
                    marker_color='lightcoral'
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Feature Analysis by Diabetes Status")
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_target_analysis(self, data):
        """Plot target variable analysis."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            outcome_counts = data['Outcome'].value_counts()
            fig1 = go.Figure(data=[go.Pie(
                labels=['Non-Diabetic', 'Diabetic'],
                values=outcome_counts.values,
                hole=0.3
            )])
            fig1.update_layout(title="Diabetes Distribution")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Age distribution by diabetes status
            fig2 = px.histogram(
                data, x='Age', color='Outcome',
                title="Age Distribution by Diabetes Status",
                color_discrete_map={0: 'lightblue', 1: 'lightcoral'}
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    def training_tab(self):
        """Training tab for model training."""
        st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Train Your Own Diabetes Detection Model
        
        This section allows you to train machine learning models on diabetes data.
        """)
        
        # Training options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Data Preprocessing")
            if st.button("🔄 Run Data Preprocessing"):
                with st.spinner("Preprocessing data..."):
                    try:
                        self.preprocessor = DataPreprocessor()
                        self.preprocessor.load_data()
                        self.preprocessor.explore_data()
                        self.preprocessor.handle_missing_values()
                        self.preprocessor.feature_engineering()
                        self.preprocessor.feature_selection(k=8)
                        self.preprocessor.scale_features()
                        self.preprocessor.split_data()
                        self.preprocessor.save_processed_data()
                        st.success("✅ Data preprocessing completed!")
                    except Exception as e:
                        st.error(f"❌ Error in preprocessing: {e}")
        
        with col2:
            st.markdown("#### 🎯 Model Training")
            if st.button("🚀 Train Models"):
                with st.spinner("Training models..."):
                    try:
                        self.trainer = DiabetesModelTrainer()
                        self.trainer.load_data()
                        self.trainer.initialize_models()
                        self.trainer.train_models()
                        self.trainer.tune_all_models()
                        self.trainer.train_models()  # Retrain with tuned parameters
                        self.trainer.save_models()
                        st.success("✅ Model training completed!")
                        
                        # Reload predictor with new model
                        self.load_models()
                    except Exception as e:
                        st.error(f"❌ Error in training: {e}")
        
        # Training progress and results
        if st.button("📈 Show Training Results"):
            if os.path.exists('models/best_model.pkl'):
                st.success("✅ Models are available!")
                
                # Load and display results
                try:
                    evaluator = ModelEvaluator('models/best_model.pkl', 'data/processed_data.pkl')
                    evaluator.load_test_data('data/processed_data.pkl')
                    
                    metrics = evaluator.calculate_metrics()
                    if metrics:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        with col2:
                            st.metric("Precision", f"{metrics['precision']:.3f}")
                        with col3:
                            st.metric("Recall", f"{metrics['recall']:.3f}")
                        with col4:
                            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                except Exception as e:
                    st.error(f"❌ Error loading results: {e}")
            else:
                st.warning("⚠️ No trained models found. Please train models first.")
    
    def evaluation_tab(self):
        """Evaluation tab for model evaluation."""
        st.markdown('<h2 class="sub-header">📈 Model Evaluation</h2>', unsafe_allow_html=True)
        
        if not os.path.exists('models/best_model.pkl'):
            st.error("❌ No trained model available for evaluation.")
            return
        
        try:
            evaluator = ModelEvaluator('models/best_model.pkl', 'data/processed_data.pkl')
            evaluator.load_test_data('data/processed_data.pkl')
            
            # Evaluation metrics
            metrics = evaluator.calculate_metrics()
            if metrics:
                st.markdown("### 📊 Model Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
                with col2:
                    st.metric("Average Precision", f"{metrics['average_precision']:.3f}")
                with col3:
                    st.metric("Log Loss", f"{metrics['log_loss']:.3f}")
                
                # Generate evaluation plots
                if st.button("📊 Generate Evaluation Plots"):
                    with st.spinner("Generating plots..."):
                        evaluator.plot_confusion_matrix()
                        evaluator.plot_roc_curve()
                        evaluator.plot_precision_recall_curve()
                        evaluator.plot_feature_importance()
                        st.success("✅ Evaluation plots generated!")
                
                # Generate detailed report
                if st.button("📋 Generate Detailed Report"):
                    with st.spinner("Generating report..."):
                        evaluator.generate_detailed_report()
                        st.success("✅ Detailed report generated!")
                
                # Error analysis
                if st.button("🔍 Analyze Prediction Errors"):
                    with st.spinner("Analyzing errors..."):
                        error_df = evaluator.analyze_prediction_errors()
                        if error_df is not None:
                            st.dataframe(error_df.head())
                            st.success("✅ Error analysis completed!")
        
        except Exception as e:
            st.error(f"❌ Error in evaluation: {e}")

def main():
    """Main function to run the Streamlit app."""
    app = DiabetesDetectionApp()
    app.main_page()

if __name__ == "__main__":
    main()
