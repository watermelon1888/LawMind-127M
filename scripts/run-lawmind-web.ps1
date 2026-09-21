$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $projectRoot "frontend"
$nodeModules = Join-Path $frontendRoot "node_modules"

if (-not (Test-Path -LiteralPath $nodeModules -PathType Container)) {
    throw "Frontend dependencies are missing. Run npm.cmd install in frontend first."
}

Push-Location $projectRoot
try {
    npm.cmd --prefix $frontendRoot run build
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend build failed."
    }

    conda run --no-capture-output -n minimind python -m api.server
    if ($LASTEXITCODE -ne 0) {
        throw "LawMind Web server exited with an error."
    }
}
finally {
    Pop-Location
}
