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

REM Run the pipeline + app orchestrator
REM python app.py
python pipeline_runner.py --config pipeline_config.json
