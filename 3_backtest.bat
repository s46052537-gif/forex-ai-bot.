@echo off
cd /d "%~dp0"
echo ===== EURUSD =====
py backtest.py --csv data\EURUSD_H1.csv --symbol EURUSD --ai
echo.
echo ===== GBPUSD =====
py backtest.py --csv data\GBPUSD_H1.csv --symbol GBPUSD --ai
echo.
echo ===== USDJPY =====
py backtest.py --csv data\USDJPY_H1.csv --symbol USDJPY --ai
pause
