$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $PSScriptRoot '.runtime\voice-studio.pid'
if (Test-Path -LiteralPath $pidFile) {
    $studioProcessId = [int](Get-Content -LiteralPath $pidFile)
    $studio = Get-Process -Id $studioProcessId -ErrorAction SilentlyContinue
    $expectedPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if ($studio) {
        if ($studio.Path -ne $expectedPython) { throw 'PID belongs to another program.' }
        & taskkill.exe /PID $studioProcessId /T /F | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Could not stop Giong Nha.' }
    }
    Remove-Item -LiteralPath $pidFile
}
& (Join-Path $PSScriptRoot 'stop-voxcpm.ps1')
Write-Output 'Giong Nha stopped.'
