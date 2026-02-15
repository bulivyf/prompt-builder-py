@echo off
setlocal enabledelayedexpansion

REM Load .env if present (simple parser: KEY=VALUE)
if exist .env (
  for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" (
      set "K=%%A"
      set "V=%%B"
      if not "!K:~0,1!"=="#" (
        set "!K!=!V!"
      )
    )
  )
)

if "%HOST%"=="" set HOST=127.0.0.1
if "%PORT%"=="" set PORT=8000

echo Starting Prompt Builder on http://%HOST%:%PORT%
echo Press CTRL+C to stop.

python -m uvicorn app.main:app --host %HOST% --port %PORT% --reload
