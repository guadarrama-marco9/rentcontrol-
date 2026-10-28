@echo off
setlocal
cd /d "%~dp0"
title RentControl - Instalar
where python >nul 2>nul
if errorlevel 1 goto python_error
python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"
if errorlevel 1 goto python_error
where npm.cmd >nul 2>nul
if errorlevel 1 goto node_error
node -e "const [a,b]=process.versions.node.split('.').map(Number);process.exit(a>22||(a===22&&b>=13)?0:1)"
if errorlevel 1 goto node_error
echo [1/4] Preparando Python...
if not exist "backend\.venv\Scripts\python.exe" python -m venv "backend\.venv"
if errorlevel 1 goto fail
echo [2/4] Instalando dependencias de Python...
"backend\.venv\Scripts\python.exe" -m pip install -r "backend\requirements.txt"
if errorlevel 1 goto fail
echo [3/4] Instalando la aplicacion...
cd RentControlApp
call npm.cmd ci
if errorlevel 1 goto fail
echo [4/4] Comprobando TypeScript...
call npm.cmd run typecheck
if errorlevel 1 goto fail
echo.
echo Instalacion terminada. Ejecuta INICIAR.cmd para abrir RentControl.
pause
exit /b 0
:python_error
echo Se requiere Python 3.11 o superior disponible con el comando python.
pause
exit /b 1
:node_error
echo Se requiere Node.js 22.13 o superior y npm disponibles en la terminal.
pause
exit /b 1
:fail
echo.
echo La instalacion no termino. Revisa el error que aparece arriba.
pause
exit /b 1
