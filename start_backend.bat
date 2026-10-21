@echo off
echo ============================================
echo  DrugMirror - Backend (FastAPI + 4 models)
echo ============================================
cd /d "%~dp0backend"

if exist ".venv\Scripts\python.exe" (
  echo [INFO] Using backend\.venv
  .venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
) else (
  echo [INFO] Using system Python
  python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
)
