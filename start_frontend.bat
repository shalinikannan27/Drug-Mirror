@echo off
echo ============================================
echo  DrugMirror - Frontend (simple HTML server)
echo ============================================
echo Open in browser: http://localhost:3000
cd /d "%~dp0frontend"
python -m http.server 3000
