import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    A comprehensive model evaluation class for diabetes detection models.
    """
    
    def __init__(self, model_path=None, data_path=None):
        """
        Initialize the model evaluator.
        
        Args:
            model_path (str): Path to the trained model
            data_path (str): Path to the test data
        """
        self.model = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
        if data_path:
            self.load_test_data(data_path)
    
    def load_model(self, model_path):
        """
        Load a trained model from file.
        
        Args:
            model_path (str): Path to the model file
        """
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            print(f"Model type: {type(self.model).__name__}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def load_test_data(self, data_path):
        """
        Load test data from file.
        
        Args:
            data_path (str): Path to the test data file
        """
        try:
            import pickle
            with open(data_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            self.X_test = data_dict['X_test']
            self.y_test = data_dict['y_test']
            self.feature_names = self.X_test.columns.tolist()
            
            print(f"Test data loaded successfully:")
            print(f"Test set shape: {self.X_test.shape}")
            print(f"Number of features: {len(self.feature_names)}")
            
        except Exception as e:
            print(f"Error loading test data: {e}")
    
    def predict(self, X=None):
        """
        Make predictions using the loaded model.
        
        Args:
            X: Input features (uses X_test if None)
        
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.model is None:
            print("No model loaded. Please load a model first.")
            return None, None
        
        if X is None:
            X = self.X_test
        
        try:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
            return predictions, probabilities
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None, None
    
    def calculate_metrics(self, y_true=None, y_pred=None, y_prob=None):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
        
        Returns:
            dict: Dictionary of metrics
        """
        if y_true is None:
            y_true = self.y_test
        
        if y_pred is None or y_prob is None:
            y_pred, y_prob = self.predict()
        
        if y_pred is None:
            return None
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, log_loss
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'average_precision': average_precision_score(y_true, y_prob),
            'log_loss': log_loss(y_true, y_prob)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true=None, y_pred=None, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        if y_true is None:
            y_true = self.y_test
        
        if y_pred is None:
            y_pred, _ = self.predict()
        
        if y_pred is None:
            return
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Diabetic', 'Diabetic'],
                   yticklabels=['Non-Diabetic', 'Diabetic'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true=None, y_prob=None, save_path=None):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            save_path: Path to save the plot
        """
        if y_true is None:
            y_true = self.y_test
        
        if y_prob is None:
            _, y_prob = self.predict()
        
        if y_prob is None:
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true=None, y_prob=None, save_path=None):
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            save_path: Path to save the plot
        """
        if y_true is None:
            y_true = self.y_test
        
        if y_prob is None:
            _, y_prob = self.predict()
        
        if y_prob is None:
            return
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, method='permutation', save_path=None):
        """
        Plot feature importance.
        
        Args:
            method (str): Method for feature importance ('permutation' or 'model')
            save_path: Path to save the plot
        """
        if self.model is None or self.X_test is None:
            print("Model or test data not loaded.")
            return
        
        if method == 'permutation':
            # Permutation importance
            perm_importance = permutation_importance(
                self.model, self.X_test, self.y_test, 
                n_repeats=10, random_state=42, n_jobs=-1
            )
            importance_scores = perm_importance.importances_mean
            importance_std = perm_importance.importances_std
        elif method == 'model':
            # Model-specific feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance_scores = self.model.feature_importances_
                importance_std = np.zeros_like(importance_scores)
            elif hasattr(self.model, 'coef_'):
                importance_scores = np.abs(self.model.coef_[0])
                importance_std = np.zeros_like(importance_scores)
            else:
                print("Model doesn't have feature importance attribute.")
                return
        else:
            print("Invalid method. Use 'permutation' or 'model'.")
            return
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores,
            'std': importance_std
        }).sort_values('importance', ascending=True)
        
        # Plot
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(feature_importance_df))
        plt.barh(y_pos, feature_importance_df['importance'], 
                xerr=feature_importance_df['std'], capsize=5)
        plt.yticks(y_pos, feature_importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance ({method.title()} Method)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        
        return feature_importance_df
    
    def plot_prediction_distribution(self, save_path=None):
        """
        Plot distribution of prediction probabilities.
        
        Args:
            save_path: Path to save the plot
        """
        _, y_prob = self.predict()
        
        if y_prob is None:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot histograms for each class
        plt.hist(y_prob[self.y_test == 0], bins=30, alpha=0.7, 
                label='Non-Diabetic', color='blue')
        plt.hist(y_prob[self.y_test == 1], bins=30, alpha=0.7, 
                label='Diabetic', color='red')
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction distribution saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, save_path='model_dashboard.html'):
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            save_path: Path to save the HTML dashboard
        """
        if self.model is None or self.X_test is None:
            print("Model or test data not loaded.")
            return
        
        y_pred, y_prob = self.predict()
        
        if y_pred is None:
            return
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC Curve', 'Precision-Recall Curve', 
                          'Confusion Matrix', 'Feature Importance'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={metrics["roc_auc"]:.3f})'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')),
            row=1, col=1
        )
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode='lines', 
                      name=f'PR (AP={metrics["average_precision"]:.3f})'),
            row=1, col=2
        )
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        fig.add_trace(
            go.Heatmap(z=cm, x=['Non-Diabetic', 'Diabetic'], 
                      y=['Non-Diabetic', 'Diabetic'], 
                      colorscale='Blues', showscale=True),
            row=2, col=1
        )
        
        # Feature Importance
        feature_importance_df = self.plot_feature_importance(method='permutation')
        fig.add_trace(
            go.Bar(x=feature_importance_df['importance'], 
                  y=feature_importance_df['feature'], 
                  orientation='h', name='Feature Importance'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Diabetes Detection Model Dashboard",
            height=800,
            showlegend=True
        )
        
        # Save dashboard
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
    
    def generate_detailed_report(self, save_path='model_evaluation_report.txt'):
        """
        Generate a detailed evaluation report.
        
        Args:
            save_path: Path to save the report
        """
        if self.model is None or self.X_test is None:
            print("Model or test data not loaded.")
            return
        
        y_pred, y_prob = self.predict()
        
        if y_pred is None:
            return
        
        metrics = self.calculate_metrics()
        
        # Generate classification report
        class_report = classification_report(self.y_test, y_pred, 
                                           target_names=['Non-Diabetic', 'Diabetic'])
        
        # Create report content
        report_content = f"""
DIABETES DETECTION MODEL EVALUATION REPORT
{'='*60}

MODEL INFORMATION:
- Model Type: {type(self.model).__name__}
- Test Set Size: {len(self.y_test)}
- Number of Features: {len(self.feature_names)}

PERFORMANCE METRICS:
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1_score']:.4f}
- ROC-AUC: {metrics['roc_auc']:.4f}
- Average Precision: {metrics['average_precision']:.4f}
- Log Loss: {metrics['log_loss']:.4f}

CLASSIFICATION REPORT:
{class_report}

CONFUSION MATRIX:
{confusion_matrix(self.y_test, y_pred)}

FEATURE NAMES:
{', '.join(self.feature_names)}

MODEL PARAMETERS:
{self.model.get_params()}
"""
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report_content)
        
        print(f"Detailed report saved to {save_path}")
        print(report_content)
    
    def analyze_prediction_errors(self):
        """
        Analyze cases where the model made incorrect predictions.
        """
        if self.model is None or self.X_test is None:
            print("Model or test data not loaded.")
            return
        
        y_pred, y_prob = self.predict()
        
        if y_pred is None:
            return
        
        # Find incorrect predictions
        incorrect_mask = y_pred != self.y_test
        incorrect_indices = np.where(incorrect_mask)[0]
        
        if len(incorrect_indices) == 0:
            print("No incorrect predictions found!")
            return
        
        print(f"\n=== PREDICTION ERROR ANALYSIS ===")
        print(f"Total incorrect predictions: {len(incorrect_indices)}")
        print(f"Error rate: {len(incorrect_indices) / len(self.y_test):.2%}")
        
        # Analyze error types
        false_positives = (y_pred == 1) & (self.y_test == 0)
        false_negatives = (y_pred == 0) & (self.y_test == 1)
        
        print(f"False Positives: {false_positives.sum()}")
        print(f"False Negatives: {false_negatives.sum()}")
        
        # Create error analysis DataFrame
        error_df = pd.DataFrame({
            'True_Label': self.y_test[incorrect_indices],
            'Predicted_Label': y_pred[incorrect_indices],
            'Predicted_Probability': y_prob[incorrect_indices],
            'Error_Type': ['False Positive' if fp else 'False Negative' 
                          for fp in false_positives[incorrect_indices]]
        })
        
        # Add feature values for incorrect predictions
        for feature in self.feature_names:
            error_df[feature] = self.X_test.iloc[incorrect_indices][feature].values
        
        print("\nError Analysis Summary:")
        print(error_df.groupby('Error_Type').agg({
            'Predicted_Probability': ['mean', 'std', 'min', 'max']
        }))
        
        return error_df
    
    def run_complete_evaluation(self, model_path=None, data_path=None):
        """
        Run complete model evaluation pipeline.
        
        Args:
            model_path: Path to the model file
            data_path: Path to the test data file
        """
        print("=== COMPLETE MODEL EVALUATION PIPELINE ===")
        
        # Load model and data
        if model_path:
            self.load_model(model_path)
        if data_path:
            self.load_test_data(data_path)
        
        if self.model is None or self.X_test is None:
            print("Model or test data not available.")
            return
        
        # Calculate and display metrics
        metrics = self.calculate_metrics()
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Generate plots
        self.plot_confusion_matrix(save_path='confusion_matrix_eval.png')
        self.plot_roc_curve(save_path='roc_curve_eval.png')
        self.plot_precision_recall_curve(save_path='pr_curve_eval.png')
        self.plot_feature_importance(save_path='feature_importance_eval.png')
        self.plot_prediction_distribution(save_path='prediction_dist_eval.png')
        
        # Create interactive dashboard
        self.create_interactive_dashboard()
        
        # Generate detailed report
        self.generate_detailed_report()
        
        # Analyze prediction errors
        error_df = self.analyze_prediction_errors()
        
        print("\n=== EVALUATION COMPLETED ===")
        print("All plots and reports have been generated.")

def main():
    """
    Main function to demonstrate model evaluation.
    """
    # Example usage
    evaluator = ModelEvaluator()
    
    # You can load a specific model and data
    # evaluator.load_model('models/best_model.pkl')
    # evaluator.load_test_data('data/processed_data.pkl')
    
    # Run complete evaluation
    # evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main()
