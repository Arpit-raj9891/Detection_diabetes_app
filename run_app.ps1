# Diabetes Detection App Launcher
Write-Host "🩺 Starting Diabetes Detection App..." -ForegroundColor Green
Write-Host ""

# Navigate to project directory
Set-Location "C:\Users\Arvind\Desktop\detection of diabetes"
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# Check if models exist
if (Test-Path "models\logistic_regression_model.pkl") {
    Write-Host "✅ Models found - ready to run!" -ForegroundColor Green
} else {
    Write-Host "⚠️ No models found. Running training first..." -ForegroundColor Yellow
    python simple_train.py
}

Write-Host ""
Write-Host "🚀 Launching Streamlit app..." -ForegroundColor Cyan
Write-Host "App will open in your browser at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

# Run the app
streamlit run app.py
