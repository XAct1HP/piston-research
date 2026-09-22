@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM  Do not write .pyc files. This project lives in OneDrive, which rewrites
REM  file timestamps when it syncs, and Python decides whether cached
REM  bytecode is stale by comparing timestamps. A synced .pyc that looks
REM  newer than its source means the old code keeps running however many
REM  times you restart -- which is indistinguishable from "the folder was
REM  never updated". The startup cost of skipping the cache is a fraction of
REM  a second; the cost of debugging the other thing is an evening.
set PYTHONDONTWRITEBYTECODE=1

REM  No arguments: start the browser front end.
REM  Any arguments: pass them straight through to the command line tool.

REM ---------------------------------------------------------------- python
REM  Windows reaches Python in several ways and "python" is the least
REM  reliable of them: the py launcher ships with the python.org installer
REM  and works even when PATH does not, while a bare "python" is often the
REM  Microsoft Store stub, which exits without doing anything.

set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY ( python --version >nul 2>&1 && set "PY=python" )
if not defined PY ( python3 --version >nul 2>&1 && set "PY=python3" )

if not defined PY (
  echo.
  echo   No Python interpreter found.
  echo.
  echo   Install Python 3.10 or newer from https://www.python.org/downloads/
  echo   and tick "Add python.exe to PATH" in the installer.
  echo.
  echo   If Python is already installed:
  echo     - open a NEW terminal window, because PATH changes need one
  echo     - or turn off the Microsoft Store stubs under
  echo       Settings ^> Apps ^> Advanced app settings ^> App execution aliases
  echo       switch OFF python.exe and python3.exe
  echo.
  exit /b 1
)

for /f "tokens=*" %%V in ('%PY% --version 2^>^&1') do set "PYVER=%%V"

REM ---------------------------------------------------------- dependencies
%PY% -c "import numpy" >nul 2>&1
if errorlevel 1 (
  echo.
  echo   Found %PYVER% ^(as: %PY%^) but its packages are not installed yet.
  echo.
  echo   Run this once:
  echo       %PY% -m pip install -r requirements.txt
  echo.
  exit /b 1
)

if "%~1"=="" (
  %PY% -c "import fastapi, uvicorn" >nul 2>&1
  if errorlevel 1 (
    echo.
    echo   The front end needs fastapi and uvicorn, which are not installed.
    echo.
    echo       %PY% -m pip install -r requirements.txt
    echo.
    echo   Or use the command line instead, which does not need them:
    echo       run.bat loads examples\ls3.json
    echo.
    exit /b 1
  )
)

%PY% -c "import cadquery" >nul 2>&1
if errorlevel 1 (
  echo   Note: cadquery is not installed, so the 3D viewport and the geometry
  echo   commands will not work. Everything else does. To add it:
  echo       %PY% -m pip install cadquery
  echo.
)

%PY% -c "import skfem, tetgen" >nul 2>&1
if errorlevel 1 (
  echo   Note: scikit-fem and tetgen are not installed, so the Stress button
  echo   and the fea command will not work. Everything else does. To add them:
  echo       %PY% -m pip install scikit-fem tetgen scipy
  echo.
)

REM ------------------------------------------------------------------ key
if "%~1"=="" (
  if not exist ".env" (
    if "%ANTHROPIC_API_KEY%"=="" (
      echo.
      echo   Note: no Anthropic API key, so the Assistant pane will be off.
      echo   To turn it on: copy .env.example to .env and put your key in it.
      echo       copy .env.example .env
      echo       notepad .env
      echo   Everything else works without it.
      echo.
    )
  )
)

REM ------------------------------------------------------------------- go
if "%~1"=="" (
  echo.
  echo   Piston System Research Tool
  echo   %PYVER% via "%PY%"
  echo   http://127.0.0.1:8000/   ^(Ctrl-C to stop^)
  echo.
  echo   For the command line instead:  run.bat loads examples\ls3.json
  echo.
  %PY% -m psrt serve examples\ls3.json
) else (
  %PY% -m psrt %*
)
exit /b %errorlevel%
