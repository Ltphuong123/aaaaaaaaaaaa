param([int]$Port = 8809)
$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$pythonExe = Join-Path $projectDir '.venv\Scripts\python.exe'
$runtimeDir = Join-Path $projectDir '.runtime'
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
& (Join-Path $projectDir 'start-voxcpm.ps1')
$backendReady = $false
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8808/' -TimeoutSec 2
        if ($response.StatusCode -eq 200) { $backendReady = $true; break }
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $backendReady) { throw 'VoxCPM did not start. Check .runtime\server-error.log.' }
$pidFile = Join-Path $runtimeDir 'voice-studio.pid'
if (Test-Path -LiteralPath $pidFile) {
    $studioProcessId = [int](Get-Content -LiteralPath $pidFile)
    $existing = Get-Process -Id $studioProcessId -ErrorAction SilentlyContinue
    if ($existing -and $existing.Path -eq $pythonExe) {
        Write-Output "Giong Nha is already running: http://127.0.0.1:$Port"
        exit 0
    }
}
$env:PYTHONUTF8 = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$studio = Start-Process -FilePath $pythonExe -ArgumentList @(
    '-m', 'uvicorn', 'voice_studio.app:app', '--host', '127.0.0.1', '--port', "$Port"
) -WorkingDirectory $projectDir -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runtimeDir 'voice-studio.log') `
    -RedirectStandardError (Join-Path $runtimeDir 'voice-studio-error.log')
Set-Content -LiteralPath $pidFile -Value $studio.Id
Write-Output "Giong Nha started: http://127.0.0.1:$Port"
