@echo off
TITLE Micro-QA + TRS-398 Standalone Workstation
cd /d "%~dp0"

echo ==============================================================================
echo    Starting Micro-QA + TRS-398 in Standalone Mode (Localhost Only)
echo ==============================================================================
echo.

IF EXIST "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=false --browser.gatherUsageStats=false
) ELSE (
    python -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=false --browser.gatherUsageStats=false
)

pause
