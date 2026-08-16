param(
    [string]$WeightsPath = "",
    [string]$Device = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
$defaultWeightsPath = Join-Path $projectRoot "minimind\checkpoints\rag_epoch_2.pth"
$expectedSha256 = "62d6f5de6c6b83e60a3f4ba8254ba7bc7c13f8bab733ff9719af22ff5af17234"

if ([string]::IsNullOrWhiteSpace($WeightsPath)) {
    $WeightsPath = $defaultWeightsPath
}

if (-not (Test-Path -LiteralPath $WeightsPath -PathType Leaf)) {
    Write-Host "错误：未找到模型权重。请将 rag_epoch_2.pth 放到：$WeightsPath" -ForegroundColor Red
    exit 2
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "错误：未找到 conda 命令。请先安装 Conda，并确保 conda 已加入 PATH。" -ForegroundColor Red
    exit 3
}

$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$chatArguments = @(
    "run",
    "--no-capture-output",
    "-n",
    "minimind",
    "python",
    "-m",
    "minimind.trainer.chat_current_law_rag",
    "--weights",
    $WeightsPath,
    "--weights-sha256",
    $expectedSha256
)
if (-not [string]::IsNullOrWhiteSpace($Device)) {
    $chatArguments += @("--device", $Device)
}

Push-Location $projectRoot
try {
    & conda @chatArguments
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    Write-Host "错误：RAG-SFT v2 交互程序启动失败，退出码：$exitCode" -ForegroundColor Red
}
exit $exitCode
