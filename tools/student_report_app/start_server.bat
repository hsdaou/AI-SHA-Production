@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=C:\StudentReportApp\venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Python environment not found. Run setup.bat first.
    pause
    exit /b 1
)

if exist "C:\StudentReportApp\Houssam Report.csv" (
    set "STUDENT_REPORT_DATA_FILE=C:\StudentReportApp\Houssam Report.csv"
)

echo Starting Student Report Viewer at http://localhost:5000
echo Keep this window open while using the app.
echo.
"%PYTHON_EXE%" start_server.py
echo.
echo The server has stopped.
pause
endlocal
