param([int]$Port = 8808)

$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$runtimeDir = Join-Path $projectDir '.runtime'
$pythonExe = Join-Path $projectDir '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'The project virtual environment is missing.'
}
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

$pidFile = Join-Path $runtimeDir 'server.pid'
if (Test-Path -LiteralPath $pidFile) {
    $serverProcessId = [int](Get-Content -LiteralPath $pidFile)
    $existing = Get-Process -Id $serverProcessId -ErrorAction SilentlyContinue
    if ($existing -and $existing.Path -eq $pythonExe) {
        Write-Output "VoxCPM is already running: http://127.0.0.1:$Port"
        exit 0
    }
}

$ffmpegDir = Get-ChildItem -LiteralPath (Join-Path $projectDir 'tools') -Directory |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'bin\ffmpeg.exe') } |
    Select-Object -First 1
if (-not $ffmpegDir) { throw 'FFmpeg is missing from tools.' }
$env:PATH = (Join-Path $ffmpegDir.FullName 'bin') + ';' + $env:PATH
$env:HF_HOME = Join-Path $projectDir 'models\huggingface'
$env:MODELSCOPE_CACHE = Join-Path $projectDir 'models\modelscope'
$env:TORCH_HOME = Join-Path $projectDir 'models\torch'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '120'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:PYTHONUTF8 = '1'
# Windows does not include the Triton compiler used by torch.compile.
$env:TORCH_COMPILE_DISABLE = '1'
# Leave room for the Windows desktop and avoid the driver's slow shared-RAM fallback.
$env:VOXCPM_GPU_MEMORY_FRACTION = '0.85'
$env:VOXCPM_RELEASE_CUDA_CACHE = '1'
$env:VOXCPM_ASR_DEVICE = 'cpu'
# Reuse reference features on CPU; retain full cloning and all inference steps.
$env:VOXCPM_PROMPT_CACHE_SIZE = '16'
$env:VOXCPM_CUDA_ESTIMATOR_GRAPH = '1'
$env:VOXCPM_CUDA_TOKEN_GRAPHS = '1'
$env:VOXCPM_CUDA_CFM_GRAPH = '1'
$env:VOXCPM_ASYNC_INFERENCE = '1'
# Return denoiser weights to RAM so later long reads retain enough GPU memory.
$env:VOXCPM_OFFLOAD_DENOISER = '1'

$server = Start-Process -FilePath $pythonExe -ArgumentList @(
    '-u', 'app.py', '--host', '127.0.0.1', '--port', "$Port",
    '--device', 'cuda', '--model-id', 'pretrained_models/VoxCPM2'
) -WorkingDirectory $projectDir -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runtimeDir 'server.log') `
    -RedirectStandardError (Join-Path $runtimeDir 'server-error.log')
Set-Content -LiteralPath $pidFile -Value $server.Id
Write-Output "VoxCPM started (PID $($server.Id)): http://127.0.0.1:$Port"
Write-Output "Logs: $runtimeDir"
