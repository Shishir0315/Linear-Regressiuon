@echo off
echo ==========================================
echo Regression Pro - Deployment Setup
echo ==========================================
echo.

echo Checking for Python...
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python/py.exe not found! Please install Python 3.11+.
    pause
    exit /b
)

echo Installing dependencies from requirements.txt...
py -m pip install -r requirements.txt

echo.
echo Setup Complete!
echo.
echo To start the web server, run: py app.py
echo Then open http://127.0.0.1:5000 in your browser.
echo.
pause
