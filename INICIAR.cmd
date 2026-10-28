@echo off
setlocal
cd /d "%~dp0"
if not exist "backend\.venv\Scripts\python.exe" goto install
if not exist "RentControlApp\node_modules\.bin\expo.cmd" goto install
start "RentControl - Python" cmd /c call "%~dp0INICIAR_BACKEND.cmd"
start "RentControl - App" cmd /c call "%~dp0INICIAR_APP.cmd"
exit /b 0
:install
echo Primero ejecuta INSTALAR.cmd y espera a que termine.
pause
exit /b 1
