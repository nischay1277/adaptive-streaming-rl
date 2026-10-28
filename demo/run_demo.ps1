$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Checkpoint = Join-Path $ProjectRoot "results\demo\ppo_adaptive.pt"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project virtual environment not found. Create it and install requirements.txt first."
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Trained demo checkpoint not found at $Checkpoint. Run the training command from README.md first."
}

$Url = "http://127.0.0.1:8123/"
Start-Process $Url
& $Python (Join-Path $ProjectRoot "demo_app.py") --checkpoint $Checkpoint --host 127.0.0.1 --port 8123
