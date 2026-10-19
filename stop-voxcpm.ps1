$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $PSScriptRoot '.runtime\server.pid'
if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Output 'VoxCPM is not running.'
    exit 0
}
$serverProcessId = [int](Get-Content -LiteralPath $pidFile)
$server = Get-Process -Id $serverProcessId -ErrorAction SilentlyContinue
$expectedPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if ($server) {
    if ($server.Path -ne $expectedPython) { throw 'PID belongs to another program; refusing to stop it.' }
    # The Windows virtual environment launcher starts a child interpreter.
    & taskkill.exe /PID $serverProcessId /T /F | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not stop the VoxCPM process tree.' }
}
Remove-Item -LiteralPath $pidFile
Write-Output 'VoxCPM stopped.'
