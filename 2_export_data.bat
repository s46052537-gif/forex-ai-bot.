@echo off
cd /d "%~dp0"
echo Make sure MetaTrader 5 is open and logged in to your DEMO account.
py export_history.py --bars 50000
pause
