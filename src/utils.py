import pandas as pd
import numpy as np
import joblib
import pickle
import os
import warnings
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

class DiabetesPredictor:
    """
    A utility class for making diabetes predictions using trained models.
    """
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the fitted scaler
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
        if scaler_path:
            self.load_scaler(scaler_path)
    
    def load_model(self, model_path: str):
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to the model file
        """
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            
            # Try to automatically load the scaler from the same directory
            model_dir = os.path.dirname(model_path)
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.load_scaler(scaler_path)
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def load_scaler(self, scaler_path: str):
        """
        Load a fitted scaler from file.
        
        Args:
            scaler_path: Path to the scaler file
        """
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded successfully from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler: {e}")
    
    def validate_input(self, data: Union[pd.DataFrame, Dict, List]) -> pd.DataFrame:
        """
        Validate and format input data for prediction.
        
        Args:
            data: Input data in various formats
            
        Returns:
            Formatted DataFrame
        """
        required_features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        if isinstance(data, dict):
            # Convert dict to DataFrame
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input data must be a dictionary, list of dictionaries, or DataFrame")
        
        # Check for required features
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only required features in correct order
        df = df[required_features]
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the input data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        # Use median values for missing data
        median_values = {
            'Pregnancies': 3,
            'Glucose': 120,
            'BloodPressure': 70,
            'SkinThickness': 20,
            'Insulin': 80,
            'BMI': 32,
            'DiabetesPedigreeFunction': 0.5,
            'Age': 33
        }
        
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(median_values[col])
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input data for prediction.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # For simple models, just return the basic features
        feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        return df[feature_columns]
    
    def predict(self, data: Union[pd.DataFrame, Dict, List]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make diabetes predictions.
        
        Args:
            data: Input data
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Validate and format input
        df = self.validate_input(data)
        
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            processed_df = pd.DataFrame(
                self.scaler.transform(processed_df),
                columns=processed_df.columns
            )
        
        # Make predictions
        predictions = self.model.predict(processed_df)
        probabilities = self.model.predict_proba(processed_df)[:, 1]
        
        return predictions, probabilities
    
    def predict_single(self, **kwargs) -> Tuple[int, float]:
        """
        Make prediction for a single patient.
        
        Args:
            **kwargs: Patient features
            
        Returns:
            Tuple of (prediction, probability)
        """
        predictions, probabilities = self.predict(kwargs)
        return int(predictions[0]), float(probabilities[0])
    
    def get_risk_level(self, probability: float) -> str:
        """
        Get risk level based on prediction probability.
        
        Args:
            probability: Prediction probability
            
        Returns:
            Risk level string
        """
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.6:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def get_recommendations(self, probability: float) -> List[str]:
        """
        Get recommendations based on prediction probability.
        
        Args:
            probability: Prediction probability
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if probability < 0.3:
            recommendations.extend([
                "Continue with healthy lifestyle",
                "Regular check-ups recommended",
                "Monitor blood sugar levels occasionally"
            ])
        elif probability < 0.6:
            recommendations.extend([
                "Consult with healthcare provider",
                "Monitor blood sugar levels regularly",
                "Consider lifestyle modifications",
                "Regular exercise recommended"
            ])
        else:
            recommendations.extend([
                "Immediate medical consultation required",
                "Frequent blood sugar monitoring",
                "Strict dietary control",
                "Regular exercise program",
                "Consider medication if prescribed"
            ])
        
        return recommendations

class DataValidator:
    """
    A utility class for validating diabetes dataset.
    """
    
    @staticmethod
    def validate_diabetes_data(df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate diabetes dataset for common issues.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'summary': {}
        }
        
        # Check required columns
        required_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing columns: {missing_columns}")
        
        # Check data types
        numeric_columns = [col for col in required_columns if col != 'Outcome']
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                validation_results['warnings'].append(f"Column {col} is not numeric")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            validation_results['warnings'].append(f"Missing values found: {missing_counts.to_dict()}")
        
        # Check for outliers
        for col in numeric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                if len(outliers) > 0:
                    validation_results['warnings'].append(f"Outliers found in {col}: {len(outliers)} values")
        
        # Check value ranges
        range_checks = {
            'Pregnancies': (0, 20),
            'Glucose': (0, 300),
            'BloodPressure': (0, 200),
            'SkinThickness': (0, 100),
            'Insulin': (0, 1000),
            'BMI': (10, 70),
            'DiabetesPedigreeFunction': (0, 3),
            'Age': (0, 120)
        }
        
        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                invalid_values = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(invalid_values) > 0:
                    validation_results['warnings'].append(f"Values out of range in {col}: {len(invalid_values)} values")
        
        # Check target distribution
        if 'Outcome' in df.columns:
            outcome_counts = df['Outcome'].value_counts()
            validation_results['summary']['outcome_distribution'] = outcome_counts.to_dict()
            
            if len(outcome_counts) < 2:
                validation_results['issues'].append("Only one class present in target variable")
        
        # Summary statistics
        validation_results['summary']['total_rows'] = len(df)
        validation_results['summary']['total_columns'] = len(df.columns)
        validation_results['summary']['missing_values'] = missing_counts.sum()
        
        return validation_results
    
    @staticmethod
    def clean_diabetes_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean diabetes dataset by handling common issues.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Handle missing values with median
        numeric_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        for col in numeric_columns:
            if col in df_clean.columns:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        
        # Remove rows with extreme outliers
        for col in numeric_columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        return df_clean

class ModelManager:
    """
    A utility class for managing trained models.
    """
    
    def __init__(self, models_directory: str = 'models/'):
        """
        Initialize the model manager.
        
        Args:
            models_directory: Directory containing model files
        """
        self.models_directory = models_directory
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """
        Load all models from the models directory.
        """
        if not os.path.exists(self.models_directory):
            print(f"Models directory {self.models_directory} does not exist.")
            return
        
        for filename in os.listdir(self.models_directory):
            if filename.endswith('.pkl'):
                model_name = filename.replace('.pkl', '')
                model_path = os.path.join(self.models_directory, filename)
                try:
                    self.models[model_name] = joblib.load(model_path)
                    print(f"Loaded model: {model_name}")
                except Exception as e:
                    print(f"Error loading model {model_name}: {e}")
    
    def get_model(self, model_name: str):
        """
        Get a specific model by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Trained model
        """
        return self.models.get(model_name)
    
    def list_models(self) -> List[str]:
        """
        List all available models.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def compare_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Compare performance of all loaded models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with comparison results
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        results = []
        
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                results.append({
                    'Model': name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1_Score': f1_score(y_test, y_pred),
                    'ROC_AUC': roc_auc_score(y_test, y_prob)
                })
            except Exception as e:
                print(f"Error evaluating model {name}: {e}")
        
        return pd.DataFrame(results)

def save_model_with_metadata(model, model_name: str, metadata: Dict, directory: str = 'models/'):
    """
    Save a model with metadata.
    
    Args:
        model: Trained model
        model_name: Name for the model
        metadata: Dictionary with model metadata
        directory: Directory to save the model
    """
    os.makedirs(directory, exist_ok=True)
    
    # Save model
    model_path = os.path.join(directory, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata_path = os.path.join(directory, f"{model_name}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Model and metadata saved: {model_path}, {metadata_path}")

def load_model_with_metadata(model_name: str, directory: str = 'models/') -> Tuple[any, Dict]:
    """
    Load a model with its metadata.
    
    Args:
        model_name: Name of the model
        directory: Directory containing the model
        
    Returns:
        Tuple of (model, metadata)
    """
    # Load model
    model_path = os.path.join(directory, f"{model_name}.pkl")
    model = joblib.load(model_path)
    
    # Load metadata
    metadata_path = os.path.join(directory, f"{model_name}_metadata.pkl")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return model, metadata

def create_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Create sample diabetes data for testing.
    
    Args:
        n_samples: Number of samples to create
        
    Returns:
        DataFrame with sample data
    """
    np.random.seed(42)
    
    data = {
        'Pregnancies': np.random.poisson(3, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples),
        'BloodPressure': np.random.normal(70, 12, n_samples),
        'SkinThickness': np.random.normal(20, 10, n_samples),
        'Insulin': np.random.normal(80, 40, n_samples),
        'BMI': np.random.normal(32, 7, n_samples),
        'DiabetesPedigreeFunction': np.random.exponential(0.5, n_samples),
        'Age': np.random.normal(33, 12, n_samples)
    }
    
    # Create target variable
    diabetes_risk = (
        (data['Glucose'] - 120) / 30 * 0.3 +
        (data['BMI'] - 32) / 7 * 0.2 +
        (data['Age'] - 33) / 12 * 0.15 +
        (data['Insulin'] - 80) / 40 * 0.1 +
        np.random.normal(0, 0.2, n_samples)
    )
    
    data['Outcome'] = (diabetes_risk > 0.5).astype(int)
    
    return pd.DataFrame(data)

def main():
    """
    Main function to demonstrate utility functions.
    """
    print("=== DIABETES DETECTION UTILITIES ===")
    
    # Create sample data
    sample_data = create_sample_data(50)
    print(f"Sample data created: {sample_data.shape}")
    
    # Validate data
    validator = DataValidator()
    validation_results = validator.validate_diabetes_data(sample_data)
    print(f"Validation results: {validation_results['is_valid']}")
    
    # Create predictor (example)
    predictor = DiabetesPredictor()
    print("Predictor initialized (no model loaded)")
    
    # Example prediction (would need a trained model)
    # sample_patient = {
    #     'Pregnancies': 1,
    #     'Glucose': 85,
    #     'BloodPressure': 66,
    #     'SkinThickness': 29,
    #     'Insulin': 0,
    #     'BMI': 26.6,
    #     'DiabetesPedigreeFunction': 0.351,
    #     'Age': 31
    # }
    # prediction, probability = predictor.predict_single(**sample_patient)
    # print(f"Sample prediction: {prediction}, Probability: {probability:.3f}")

if __name__ == "__main__":
    main()
