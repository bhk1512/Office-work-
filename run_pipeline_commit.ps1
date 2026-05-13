$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot

$runLog = Join-Path $scriptRoot "run_pipeline_commit.log"
function Write-RunLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$stamp] $Message"
    Write-Host $line
    Add-Content -Path $runLog -Value $line
}

function Invoke-CommandWithRetry {
    param(
        [scriptblock]$Command,
        [string]$StepName,
        [int]$MaxAttempts = 3,
        [int]$DelaySeconds = 10
    )

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Write-RunLog "$StepName (attempt $attempt/$MaxAttempts)"
        try {
            & $Command
            return
        }
        catch {
            if ($attempt -ge $MaxAttempts) {
                throw
            }
            Write-RunLog "$StepName failed: $($_.Exception.Message). Retrying in $DelaySeconds second(s)."
            Start-Sleep -Seconds $DelaySeconds
        }
    }
}

$pyLauncher = Join-Path $env:LocalAppData "Programs\\Python\\Launcher"
$pyRoot = Join-Path $env:LocalAppData "Programs\\Python\\Python313"
$basePython = "python"
if (Test-Path (Join-Path $pyRoot "python.exe")) {
    $basePython = Join-Path $pyRoot "python.exe"
}

$venvPython = Join-Path $scriptRoot "venv\\Scripts\\python.exe"
if (-not (Test-Path $venvPython)) {
    & $basePython -m venv venv
}
if (-not (Test-Path $venvPython)) {
    throw "venv python not found at $venvPython"
}

Write-RunLog "===== Pipeline run started ====="

Invoke-CommandWithRetry -StepName "pip upgrade" -MaxAttempts 2 -DelaySeconds 8 -Command {
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
}
Invoke-CommandWithRetry -StepName "pip install requirements" -MaxAttempts 2 -DelaySeconds 8 -Command {
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install requirements.txt failed." }
}

Write-RunLog "Running outlook_dpr_watcher.py"
& $venvPython outlook_dpr_watcher.py
if ($LASTEXITCODE -ne 0) { throw "outlook_dpr_watcher.py failed." }

Write-RunLog "Running pipeline_runner.py --no-serve"
& $venvPython pipeline_runner.py --config pipeline_config.json --no-serve
if ($LASTEXITCODE -ne 0) { throw "pipeline_runner.py failed." }

Write-RunLog "Staging git changes"
git add -A
if ($LASTEXITCODE -ne 0) { throw "git add failed." }

git diff --cached --quiet
$hasStagedChanges = $LASTEXITCODE -ne 0
if ($hasStagedChanges) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $commitMessage = "Auto pipeline update $timestamp"
    Write-RunLog "Creating commit: $commitMessage"
    git commit -m $commitMessage
    if ($LASTEXITCODE -ne 0) { throw "git commit failed." }
} else {
    Write-RunLog "No changes to commit."
}

$sshDir = Join-Path $env:USERPROFILE ".ssh"
$vmKey = $env:VM_SSH_KEY
if ([string]::IsNullOrWhiteSpace($vmKey)) {
    $candidateKeys = @(
        (Join-Path $sshDir "id_ed25519"),
        (Join-Path $sshDir "id_rsa")
    )
    $vmKey = $candidateKeys | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if ([string]::IsNullOrWhiteSpace($vmKey)) {
    throw "No SSH key found. Create one and add it to the vm server, or set VM_SSH_KEY to a private key path."
}

$prevSshCommand = $env:GIT_SSH_COMMAND
try {
    if ([string]::IsNullOrWhiteSpace($env:VM_SSH_KEY)) {
        $env:GIT_SSH_COMMAND = "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    } else {
        $env:GIT_SSH_COMMAND = "ssh -i `"$vmKey`" -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    }
    Invoke-CommandWithRetry -StepName "git push vm main" -MaxAttempts 2 -DelaySeconds 6 -Command {
        git push vm main
        if ($LASTEXITCODE -ne 0) { throw "git push vm main failed." }
    }
}
finally {
    if ($null -ne $prevSshCommand) { $env:GIT_SSH_COMMAND = $prevSshCommand } else { Remove-Item Env:\\GIT_SSH_COMMAND -ErrorAction SilentlyContinue }
}

Invoke-CommandWithRetry -StepName "git push origin main (final step)" -MaxAttempts 3 -DelaySeconds 15 -Command {
    git push origin main
    if ($LASTEXITCODE -ne 0) { throw "git push origin main failed." }
}

Write-RunLog "===== Pipeline run completed successfully ====="
