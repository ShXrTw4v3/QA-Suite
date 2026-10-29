@echo off
TITLE Setup Micro-QA Environment (Windows)
cd /d "%~dp0"

echo ==============================================================================
echo    Micro-QA + TRS-398 Suite - Windows Automated Setup
echo ==============================================================================
echo.

:: Check for Python installation
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in PATH.
    echo Please install Python 3.9 - 3.11 (64-bit) and check "Add Python to PATH".
    echo Download: https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [1/3] Creating virtual environment (venv)...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment already exists.
)

:: Upgrade pip
echo [2/3] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip

:: Install requirements
echo [3/3] Installing Python dependencies from requirements.txt...
venv\Scripts\python.exe -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARNING] Online installation encountered an issue.
    echo If this workstation is offline / on an isolated research LAN,
    echo run: pip install --no-index --find-links=wheels -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ==============================================================================
echo    Setup completed successfully!
echo    To launch on local hospital intranet: double-click start_hospital_server.bat
echo    To launch locally only: double-click run_standalone_windows.bat
echo ==============================================================================
pause
