@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "BASE=C:\Users\kaushikb\Documents\Work\Git\Office-work-"
set "JOB_BAT=%BASE%\run_pipeline_commit.bat"
set "LOG=%BASE%\scheduler.log"
set "LOCK=%temp%\dashboard_scheduler.lock"

set "T1=10:30"
set "T2=11:30"
set "T3=12:02"
set "T4=16:00"

echo [%DATE% %TIME%] ===== Scheduler launch =====>> "%LOG%"
echo BASE=%BASE%>> "%LOG%"
echo JOB_BAT=%JOB_BAT%>> "%LOG%"

REM --- Validate job exists (NO parentheses block) ---
if not exist "%JOB_BAT%" goto :job_missing

REM --- Auto-clean lock (safe for single-user scheduler) ---
if exist "%LOCK%" (
  echo [%DATE% %TIME%] NOTE: Removing stale lock %LOCK%>> "%LOG%"
  del "%LOCK%" 2>nul
)

echo running > "%LOCK%"

echo [%DATE% %TIME%] Scheduler running (keep window open).>> "%LOG%"
echo Scheduler running. Logs: %LOG%

set "LAST_RUN_1="
set "LAST_RUN_2="
set "LAST_RUN_3="
set "LAST_RUN_4="

:loop
REM Parse HH and MM from %TIME% safely
set "HH_RAW="
set "MN_RAW="
for /f "tokens=1-2 delims=:. " %%a in ("%TIME%") do (
  set "HH_RAW=%%a"
  set "MN_RAW=%%b"
)

set "HH_RAW=%HH_RAW: =%"
set "MN_RAW=%MN_RAW: =%"

set /a HHN=100%HH_RAW% %% 100
set /a MNN=100%MN_RAW% %% 100

if %HHN% LSS 10 (set "HH=0%HHN%") else set "HH=%HHN%"
if %MNN% LSS 10 (set "MN=0%MNN%") else set "MN=%MNN%"

set "NOW=%HH%:%MN%"
set "TODAY=%DATE%"

call :maybe_run "%T1%" LAST_RUN_1
call :maybe_run "%T2%" LAST_RUN_2
call :maybe_run "%T3%" LAST_RUN_3
call :maybe_run "%T4%" LAST_RUN_4

timeout /t 20 /nobreak >nul
goto loop

:maybe_run
set "TARGET=%~1"
set "VAR=%~2"
set "STAMP=%TODAY%_%TARGET%"

if "%NOW%"=="%TARGET%" (
  if not "!%VAR%!"=="%STAMP%" (
    echo [%DATE% %TIME%] Trigger %TARGET% starting job>> "%LOG%"
    call "%JOB_BAT%"
    echo [%DATE% %TIME%] Job finished errorlevel=!ERRORLEVEL!>> "%LOG%"
    set "%VAR%=%STAMP%"
  )
)
exit /b 0

:job_missing
echo [%DATE% %TIME%] ERROR: JOB_BAT not found: %JOB_BAT%>> "%LOG%"
echo ERROR: Cannot find %JOB_BAT%
pause
exit /b 1
