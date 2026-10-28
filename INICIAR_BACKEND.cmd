@echo off
setlocal
cd /d "%~dp0backend"
if not exist ".venv\Scripts\python.exe" (
 echo Ejecuta INSTALAR.cmd primero.
 pause
 exit /b 1
)
set STORAGE=demo
set AUTH_REQUIRED=1
set RENTCONTROL_API_KEY=
set CORS_ORIGINS=http://localhost:8081,http://127.0.0.1:8081
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
