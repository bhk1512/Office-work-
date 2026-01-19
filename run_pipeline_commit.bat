@echo off
setlocal

REM Optional: set VM_SSH_KEY to a specific private key path.
REM set "VM_SSH_KEY=C:\Users\kaushikb\.ssh\id_ed25519"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_pipeline_commit.ps1"
set "exitcode=%ERRORLEVEL%"
endlocal & exit /b %exitcode%
