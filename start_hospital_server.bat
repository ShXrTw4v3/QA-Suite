@echo off
TITLE Micro-QA + TRS-398 Hospital Network Server
cd /d "%~dp0"

IF EXIST "venv\Scripts\python.exe" (
    venv\Scripts\python.exe hospital_server.py --browser %*
) ELSE (
    python hospital_server.py --browser %*
)
pause
