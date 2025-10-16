@echo off
REM Ensure local Python paths are available (Cursor sessions trim PATH)
set "PY_LAUNCHER=%LocalAppData%\Programs\Python\Launcher"
set "PY_ROOT=%LocalAppData%\Programs\Python\Python313"
if exist "%PY_LAUNCHER%\py.exe" (
    set "PATH=%PY_LAUNCHER%;%PY_ROOT%;%PY_ROOT%\Scripts;%PATH%"
)

REM Create venv if it doesn't exist
if not exist venv (
    REM python -m venv venv
    if exist "%PY_ROOT%\python.exe" (
        "%PY_ROOT%\python.exe" -m venv venv
    ) else (
        python -m venv venv
    )
)

REM Activate venv
call venv\Scripts\activate

REM Install dependencies
REM pip install --upgrade pip
python -m pip install --upgrade pip
REM pip install -r requirements.txt
python -m pip install -r requirements.txt

REM Run pipeline only (no dev server) — performs delete + compile sequentially
python pipeline_runner.py --config pipeline_config.json --no-serve
if errorlevel 1 (
    echo [run_app] Pipeline failed; not starting server.
    exit /b 1
)

REM Serve via Waitress (prod-like on Windows)
set DASH_HOST=0.0.0.0
set DASH_PORT=8050
set APP_ENV=development
set ENABLE_HTTPS=0
set BEHIND_PROXY=0
waitress-serve --listen=%DASH_HOST%:%DASH_PORT% app:server
