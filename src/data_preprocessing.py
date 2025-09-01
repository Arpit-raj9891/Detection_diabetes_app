import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    A comprehensive data preprocessing class for diabetes detection dataset.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            data_path (str): Path to the diabetes dataset CSV file
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_selector = None
        
    def load_data(self, data_path=None):
        """
        Load the diabetes dataset.
        
        Args:
            data_path (str): Path to the dataset file
        """
        if data_path:
            self.data_path = data_path
            
        if self.data_path:
            try:
                self.data = pd.read_csv(self.data_path)
                print(f"Data loaded successfully. Shape: {self.data.shape}")
                return self.data
            except FileNotFoundError:
                print("Dataset file not found. Creating synthetic data...")
                return self.create_synthetic_data()
        else:
            print("Creating synthetic diabetes dataset...")
            return self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """
        Create synthetic diabetes dataset for demonstration purposes.
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic features based on typical diabetes dataset characteristics
        pregnancies = np.random.poisson(3, n_samples)
        glucose = np.random.normal(120, 30, n_samples)
        blood_pressure = np.random.normal(70, 12, n_samples)
        skin_thickness = np.random.normal(20, 10, n_samples)
        insulin = np.random.normal(80, 40, n_samples)
        bmi = np.random.normal(32, 7, n_samples)
        diabetes_pedigree = np.random.exponential(0.5, n_samples)
        age = np.random.normal(33, 12, n_samples)
        
        # Create target variable based on some rules
        # Higher glucose, BMI, age, and insulin levels increase diabetes risk
        diabetes_risk = (
            (glucose - 120) / 30 * 0.3 +
            (bmi - 32) / 7 * 0.2 +
            (age - 33) / 12 * 0.15 +
            (insulin - 80) / 40 * 0.1 +
            np.random.normal(0, 0.2, n_samples)
        )
        
        outcome = (diabetes_risk > 0.5).astype(int)
        
        # Create DataFrame
        self.data = pd.DataFrame({
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
        
        # Add some missing values to make it more realistic
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            mask = np.random.random(n_samples) < 0.05
            self.data.loc[mask, col] = np.nan
        
        print(f"Synthetic data created. Shape: {self.data.shape}")
        return self.data
    
    def explore_data(self):
        """
        Perform exploratory data analysis.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"\nColumns: {list(self.data.columns)}")
        
        print("\n=== DATA TYPES ===")
        print(self.data.dtypes)
        
        print("\n=== MISSING VALUES ===")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        print("\n=== BASIC STATISTICS ===")
        print(self.data.describe())
        
        print("\n=== TARGET DISTRIBUTION ===")
        print(self.data['Outcome'].value_counts())
        print(f"Diabetes rate: {self.data['Outcome'].mean():.2%}")
        
        # Create correlation matrix
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()
        
        # Feature distributions
        fig, axes = plt.subplots(2, 4, figsize=(15, 10))
        features = [col for col in self.data.columns if col != 'Outcome']
        
        for i, feature in enumerate(features):
            row = i // 4
            col = i % 4
            self.data[feature].hist(ax=axes[row, col], bins=20)
            axes[row, col].set_title(f'{feature} Distribution')
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png')
        plt.close()
        
        print("\nVisualization files saved: correlation_matrix.png, feature_distributions.png")
    
    def handle_missing_values(self, strategy='median'):
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Imputation strategy ('mean', 'median', 'most_frequent')
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print(f"\n=== HANDLING MISSING VALUES (Strategy: {strategy}) ===")
        
        # Check for missing values
        missing_before = self.data.isnull().sum()
        print("Missing values before imputation:")
        print(missing_before[missing_before > 0])
        
        # Impute missing values
        self.imputer = SimpleImputer(strategy=strategy)
        features = [col for col in self.data.columns if col != 'Outcome']
        
        self.data[features] = self.imputer.fit_transform(self.data[features])
        
        # Check after imputation
        missing_after = self.data.isnull().sum()
        print("\nMissing values after imputation:")
        print(missing_after[missing_after > 0])
    
    def feature_engineering(self):
        """
        Perform feature engineering to create new features.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("\n=== FEATURE ENGINEERING ===")
        
        # Create new features
        self.data['Glucose_BMI_Ratio'] = self.data['Glucose'] / self.data['BMI']
        self.data['Age_BMI_Product'] = self.data['Age'] * self.data['BMI']
        self.data['Insulin_Glucose_Ratio'] = self.data['Insulin'] / self.data['Glucose']
        
        # Create age groups
        self.data['Age_Group'] = pd.cut(
            self.data['Age'], 
            bins=[0, 30, 45, 60, 100], 
            labels=['Young', 'Middle', 'Senior', 'Elderly']
        )
        
        # Create BMI categories
        self.data['BMI_Category'] = pd.cut(
            self.data['BMI'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        
        # Convert categorical to numerical
        self.data['Age_Group_Encoded'] = self.data['Age_Group'].cat.codes
        self.data['BMI_Category_Encoded'] = self.data['BMI_Category'].cat.codes
        
        print("New features created:")
        print("- Glucose_BMI_Ratio")
        print("- Age_BMI_Product")
        print("- Insulin_Glucose_Ratio")
        print("- Age_Group (categorical)")
        print("- BMI_Category (categorical)")
        print("- Age_Group_Encoded")
        print("- BMI_Category_Encoded")
        
        print(f"\nUpdated dataset shape: {self.data.shape}")
    
    def feature_selection(self, k=10):
        """
        Perform feature selection using statistical tests.
        
        Args:
            k (int): Number of top features to select
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print(f"\n=== FEATURE SELECTION (Top {k} features) ===")
        
        # Prepare features and target
        features = [col for col in self.data.columns if col not in ['Outcome', 'Age_Group', 'BMI_Category']]
        X = self.data[features]
        y = self.data['Outcome']
        
        # Perform feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        feature_scores = self.feature_selector.scores_[self.feature_selector.get_support()]
        
        print("Selected features and their F-scores:")
        for feature, score in zip(selected_features, feature_scores):
            print(f"{feature}: {score:.2f}")
        
        # Update data with selected features
        self.data = self.data[selected_features + ['Outcome']]
        print(f"\nFinal dataset shape after feature selection: {self.data.shape}")
    
    def scale_features(self, scaler_type='standard'):
        """
        Scale features using specified scaler.
        
        Args:
            scaler_type (str): Type of scaler ('standard', 'minmax')
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print(f"\n=== FEATURE SCALING ({scaler_type}) ===")
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        features = [col for col in self.data.columns if col != 'Outcome']
        self.data[features] = self.scaler.fit_transform(self.data[features])
        
        print("Features scaled successfully.")
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print(f"\n=== DATA SPLITTING (Test size: {test_size}) ===")
        
        # Prepare features and target
        features = [col for col in self.data.columns if col != 'Outcome']
        X = self.data[features]
        y = self.data['Outcome']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
        print(f"Training target distribution: {self.y_train.value_counts().to_dict()}")
        print(f"Testing target distribution: {self.y_test.value_counts().to_dict()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_processed_data(self):
        """
        Get the processed training and testing data.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.X_train is None:
            print("Data not split yet. Please run split_data() first.")
            return None
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_processed_data(self, filepath='data/processed_data.pkl'):
        """
        Save the processed data to a file.
        
        Args:
            filepath (str): Path to save the processed data
        """
        import pickle
        
        if self.X_train is None:
            print("No processed data to save.")
            return
        
        data_dict = {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_selector': self.feature_selector
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"Processed data saved to {filepath}")

def main():
    """
    Main function to demonstrate data preprocessing pipeline.
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    data = preprocessor.load_data()
    
    # Explore data
    preprocessor.explore_data()
    
    # Handle missing values
    preprocessor.handle_missing_values()
    
    # Feature engineering
    preprocessor.feature_engineering()
    
    # Feature selection
    preprocessor.feature_selection(k=8)
    
    # Scale features
    preprocessor.scale_features()
    
    # Split data
    preprocessor.split_data()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    print("\n=== DATA PREPROCESSING COMPLETED ===")

if __name__ == "__main__":
    main()
