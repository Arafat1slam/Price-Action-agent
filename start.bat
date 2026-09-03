@echo off
title Live Price Action Scanner
cd /d %~dp0

echo ========================================================
echo        Starting Live Price Action Scanner Engine
echo ========================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH!
    echo Please install Python 3.10+ from python.org and check "Add Python to PATH".
    pause
    exit /b 1
)

:: Create virtual environment if it does not exist
if not exist "venv\Scripts\activate.bat" (
    echo [SETUP] Creating Python virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install / Update dependencies
echo [SETUP] Checking dependencies...
pip install -r requirements.txt --quiet

:: Run Main Application
echo [LAUNCH] Starting Scanner CLI...
echo.
python main.py

pause
