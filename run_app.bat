@echo off
cd /d "%~dp0"
title AI Video Assistant
echo ========================================================
echo   Launching AI Video Assistant (Streamlit Web App)...
echo   URL: http://localhost:8502
echo ========================================================
call .\.venv\Scripts\activate.bat
python -m streamlit run app.py --server.port 8502 --server.address localhost
pause
