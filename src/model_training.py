import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import XGBoost and LightGBM if available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

class DiabetesModelTrainer:
    """
    A comprehensive model training class for diabetes detection.
    """
    
    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Initialize the model trainer.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.results = {}
        
    def load_data(self, data_path='data/processed_data.pkl'):
        """
        Load processed data from file.
        
        Args:
            data_path (str): Path to the processed data file
        """
        try:
            import pickle
            with open(data_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            self.X_train = data_dict['X_train']
            self.X_test = data_dict['X_test']
            self.y_train = data_dict['y_train']
            self.y_test = data_dict['y_test']
            
            print(f"Data loaded successfully:")
            print(f"Training set: {self.X_train.shape}")
            print(f"Testing set: {self.X_test.shape}")
            
        except FileNotFoundError:
            print(f"Processed data file not found at {data_path}")
            print("Please run data preprocessing first.")
    
    def initialize_models(self):
        """
        Initialize all machine learning models.
        """
        print("\n=== INITIALIZING MODELS ===")
        
        # Basic models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(random_state=42)
        
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMClassifier(random_state=42)
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"- {name}")
    
    def train_models(self):
        """
        Train all models and evaluate their performance.
        """
        if self.X_train is None:
            print("No training data available. Please load data first.")
            return
        
        print("\n=== TRAINING MODELS ===")
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Find best model
        self.find_best_model()
    
    def find_best_model(self, metric='f1_score'):
        """
        Find the best performing model based on specified metric.
        
        Args:
            metric (str): Metric to use for comparison
        """
        if not self.results:
            print("No results available. Please train models first.")
            return
        
        best_score = 0
        best_model_name = None
        
        for name, result in self.results.items():
            if result[metric] > best_score:
                best_score = result[metric]
                best_model_name = name
        
        self.best_model = self.results[best_model_name]['model']
        self.best_score = best_score
        
        print(f"\n=== BEST MODEL ===")
        print(f"Model: {best_model_name}")
        print(f"Best {metric}: {best_score:.4f}")
    
    def cross_validation_evaluation(self, cv=5):
        """
        Perform cross-validation evaluation for all models.
        
        Args:
            cv (int): Number of cross-validation folds
        """
        if self.X_train is None:
            print("No training data available. Please load data first.")
            return
        
        print(f"\n=== CROSS-VALIDATION EVALUATION (CV={cv}) ===")
        
        cv_results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name} with cross-validation...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1')
            
            cv_results[name] = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scores': cv_scores
            }
            
            print(f"{name} - CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_results
    
    def hyperparameter_tuning(self, model_name, param_grid, cv=5, n_iter=20):
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name (str): Name of the model to tune
            param_grid (dict): Parameter grid for tuning
            cv (int): Number of cross-validation folds
            n_iter (int): Number of iterations for randomized search
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
        
        print(f"\n=== HYPERPARAMETER TUNING FOR {model_name.upper()} ===")
        
        model = self.models[model_name]
        
        # Use RandomizedSearchCV for faster tuning
        random_search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv, scoring='f1',
            random_state=42, n_jobs=-1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        
        # Update the model with best parameters
        self.models[model_name] = random_search.best_estimator_
        
        return random_search.best_estimator_
    
    def get_hyperparameter_grids(self):
        """
        Get predefined hyperparameter grids for different models.
        
        Returns:
            dict: Dictionary of parameter grids
        """
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
        
        if XGBOOST_AVAILABLE:
            param_grids['XGBoost'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        if LIGHTGBM_AVAILABLE:
            param_grids['LightGBM'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            }
        
        return param_grids
    
    def tune_all_models(self):
        """
        Perform hyperparameter tuning for all models.
        """
        print("\n=== HYPERPARAMETER TUNING FOR ALL MODELS ===")
        
        param_grids = self.get_hyperparameter_grids()
        
        for model_name in self.models.keys():
            if model_name in param_grids:
                print(f"\nTuning {model_name}...")
                self.hyperparameter_tuning(model_name, param_grids[model_name])
            else:
                print(f"Skipping {model_name} - no parameter grid defined")
    
    def plot_results_comparison(self):
        """
        Plot comparison of model results.
        """
        if not self.results:
            print("No results available. Please train models first.")
            return
        
        # Prepare data for plotting
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color='skyblue', alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Remove the last subplot if not needed
        if len(metrics) < 6:
            axes[-1].remove()
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Model comparison plot saved as 'model_comparison.png'")
    
    def plot_roc_curves(self):
        """
        Plot ROC curves for all models.
        """
        if not self.results:
            print("No results available. Please train models first.")
            return
        
        plt.figure(figsize=(10, 8))
        
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            auc = result['roc_auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("ROC curves plot saved as 'roc_curves.png'")
    
    def plot_confusion_matrices(self):
        """
        Plot confusion matrices for all models.
        """
        if not self.results:
            print("No results available. Please train models first.")
            return
        
        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (name, result) in enumerate(self.results.items()):
            row = i // n_cols
            col = i % n_cols
            
            cm = confusion_matrix(self.y_test, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col])
            axes[row, col].set_title(f'{name}\nConfusion Matrix')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
        
        # Remove empty subplots
        for i in range(n_models, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].remove()
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Confusion matrices plot saved as 'confusion_matrices.png'")
    
    def save_models(self, directory='models/'):
        """
        Save all trained models to disk.
        
        Args:
            directory (str): Directory to save models
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        print(f"\n=== SAVING MODELS TO {directory} ===")
        
        for name, result in self.results.items():
            model_path = os.path.join(directory, f'{name.lower().replace(" ", "_")}.pkl')
            joblib.dump(result['model'], model_path)
            print(f"Saved {name} to {model_path}")
        
        # Save best model separately
        if self.best_model is not None:
            best_model_path = os.path.join(directory, 'best_model.pkl')
            joblib.dump(self.best_model, best_model_path)
            print(f"Saved best model to {best_model_path}")
    
    def generate_report(self):
        """
        Generate a comprehensive model evaluation report.
        """
        if not self.results:
            print("No results available. Please train models first.")
            return
        
        print("\n" + "="*60)
        print("DIABETES DETECTION MODEL EVALUATION REPORT")
        print("="*60)
        
        # Create results DataFrame
        report_data = []
        for name, result in self.results.items():
            report_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1_score']:.4f}",
                'ROC-AUC': f"{result['roc_auc']:.4f}"
            })
        
        report_df = pd.DataFrame(report_data)
        print("\nModel Performance Summary:")
        print(report_df.to_string(index=False))
        
        # Best model information
        if self.best_model is not None:
            print(f"\nBest Model: {self.best_model.__class__.__name__}")
            print(f"Best F1-Score: {self.best_score:.4f}")
        
        print("\n" + "="*60)
    
    def run_complete_pipeline(self):
        """
        Run the complete model training pipeline.
        """
        print("=== DIABETES DETECTION MODEL TRAINING PIPELINE ===")
        
        # Load data
        self.load_data()
        
        # Initialize models
        self.initialize_models()
        
        # Train models
        self.train_models()
        
        # Cross-validation evaluation
        cv_results = self.cross_validation_evaluation()
        
        # Hyperparameter tuning
        self.tune_all_models()
        
        # Retrain with tuned models
        self.train_models()
        
        # Generate plots
        self.plot_results_comparison()
        self.plot_roc_curves()
        self.plot_confusion_matrices()
        
        # Generate report
        self.generate_report()
        
        # Save models
        self.save_models()
        
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")

def main():
    """
    Main function to run the model training pipeline.
    """
    trainer = DiabetesModelTrainer()
    trainer.run_complete_pipeline()

if __name__ == "__main__":
    main()
