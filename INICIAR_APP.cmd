@echo off
setlocal
cd /d "%~dp0RentControlApp"
if not exist "node_modules\.bin\expo.cmd" (
 echo Ejecuta INSTALAR.cmd primero.
 pause
 exit /b 1
)
set EXPO_PUBLIC_API_URL=http://localhost:8000
set EXPO_PUBLIC_API_KEY=
call npm.cmd run web -- --port 8081 --clear
pause
