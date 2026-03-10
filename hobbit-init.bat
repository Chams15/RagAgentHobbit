@echo off
REM The Hobbit Scholar - Database Initializer
REM This batch file allows you to initialize the database from anywhere

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Change to the project directory (important for relative paths)
cd /d "%SCRIPT_DIR%"

REM Run the Python script using the virtual environment
"%SCRIPT_DIR%venv\Scripts\python.exe" "%SCRIPT_DIR%vector.py" %*
