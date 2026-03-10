@echo off
REM The Hobbit Scholar - Main Chatbot Launcher
REM This batch file allows you to run the chatbot from anywhere

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Change to the project directory (important for relative paths)
cd /d "%SCRIPT_DIR%"

REM Run the Python script using the virtual environment
"%SCRIPT_DIR%venv\Scripts\python.exe" "%SCRIPT_DIR%main.py" %*
