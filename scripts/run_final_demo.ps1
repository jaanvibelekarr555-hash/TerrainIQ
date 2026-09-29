$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:PYTHONPATH = $ProjectRoot

Write-Host "==============================================="
Write-Host " ADAPTIVE LiDAR FINAL DEMO"
Write-Host " Real RELLIS-3D + Drivable Path + Vehicle"
Write-Host "==============================================="
Write-Host ""

python scripts\final_adaptive_lidar_demo.py

if ($LASTEXITCODE -ne 0) {
    throw "Demo generation failed."
}

$DemoFile = Join-Path $ProjectRoot "demo\index.html"

if (-not (Test-Path $DemoFile)) {
    throw "Demo output was not created: $DemoFile"
}

Write-Host ""
Write-Host "Opening final demo: $DemoFile"
Start-Process $DemoFile
