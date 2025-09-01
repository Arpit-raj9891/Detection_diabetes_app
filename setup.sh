#!/usr/bin/env bash

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories if they don't exist
mkdir -p models
mkdir -p data

# Check if models exist, if not, create a simple message
if [ ! -f "models/logistic_regression_model.pkl" ]; then
    echo "Models not found. Please ensure models are trained and saved."
fi

echo "Setup complete!"
