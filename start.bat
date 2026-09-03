@echo off
title Live Price Action Assistant - AI Cockpit v2.0
cd /d %~dp0

:: Enable Windows UTF-8 code page & VT100 processing
chcp 65001 >nul 2>&1
reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul 2>&1

echo ========================================================
echo       AI Price Action Assistant - Cockpit v2.0
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
