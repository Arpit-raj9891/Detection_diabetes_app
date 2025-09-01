# Diabetes Detection using Machine Learning

This project implements a comprehensive machine learning solution for diabetes detection using various algorithms and provides both a command-line interface and a web-based application.

## Features

- **Three Core ML Algorithms**: Logistic Regression, Random Forest, and Support Vector Machine (SVM)
- **Data Preprocessing**: Feature scaling, handling missing values, and feature engineering
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and ROC curves
- **Web Interface**: Streamlit-based web application for easy interaction
- **Model Persistence**: Save and load trained models
- **Data Visualization**: Interactive plots and charts for data analysis

## Project Structure

```
detection of diabetes/
├── data/
│   └── diabetes_dataset.csv
├── models/
│   └── (saved model files)
├── notebooks/
│   └── diabetes_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── app.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone or download this project
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Analysis and Model Training

Run the main training script:
```bash
python src/model_training.py
```

### 2. Web Application

Launch the Streamlit web interface:
```bash
streamlit run app.py
```

### 3. Jupyter Notebook

For detailed analysis, open the Jupyter notebook:
```bash
jupyter notebook notebooks/diabetes_analysis.ipynb
```

## Dataset

The project uses a diabetes dataset with the following features:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)²)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age in years
- Outcome: Target variable (0 = non-diabetic, 1 = diabetic)

## Model Performance

The project implements three core algorithms:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Each model is evaluated using:
- Accuracy Score
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix

## Contributing

Feel free to contribute to this project by:
- Adding new algorithms
- Improving the web interface
- Enhancing data preprocessing
- Adding new evaluation metrics

##Project deployed at: https://detectiondiabetesapp-5ehhcaa3mztpmqdj6r8uzx.streamlit.app/

## License

This project is open source and available under the MIT License.
"# Diabetes_detection1" 
