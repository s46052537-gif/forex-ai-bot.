@echo off
cd /d "%~dp0"
echo Make sure MetaTrader 5 is open, logged in to your DEMO account,
echo and the "Algo Trading" button is green.
py gui.py
pause
