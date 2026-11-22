@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found.
  echo Follow README.md to create .venv and install requirements.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m streamlit run app.py
pause
