@echo off
cd /d "%~dp0"
echo Installing required libraries...
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
echo.
echo Done. If you saw no red errors, the installation succeeded.
pause
