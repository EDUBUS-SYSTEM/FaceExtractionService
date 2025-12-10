@echo off
title EduBus Face Extraction Service
color 0A

echo ==================================================
echo   EDU-BUS FACE EXTRACTION SERVICE
echo   Starting in PRODUCTION mode...
echo ==================================================

cd /d "%~dp0"

:: Check if venv exists (optional, assumes python is available)
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python.
    pause
    exit /b
)

:: Install Waitress if missing
pip show waitress >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing Waitress...
    pip install waitress
)

:: Run Server
echo 🚀 Starting Server...
python server.py

pause
