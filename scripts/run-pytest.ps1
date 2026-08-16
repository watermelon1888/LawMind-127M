[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pytestTempLeaf = 'transformer-pytest-{0}' -f (
    [System.Guid]::NewGuid().ToString('N')
)

if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $repoRoot
} else {
    $env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"
}

$pytestBaseTemp = Join-Path ([System.IO.Path]::GetTempPath()) $pytestTempLeaf
$systemTempRoot = [System.IO.Path]::GetFullPath(
    [System.IO.Path]::GetTempPath()
).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
$pytestTempRoot = [System.IO.Path]::GetFullPath(
    $pytestBaseTemp
).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
if (
    [string]::IsNullOrWhiteSpace($pytestTempLeaf) -or
    $pytestTempLeaf -eq 'transformer-pytest-' -or
    $pytestTempRoot -eq $systemTempRoot
) {
    throw (
        'Failed to create an isolated pytest temporary directory: ' +
        "leaf=$pytestTempLeaf, target=$pytestTempRoot, system=$systemTempRoot"
    )
}

& conda run --no-capture-output --cwd $repoRoot -n minimind `
    python -m pytest "--basetemp=$pytestBaseTemp" -p no:cacheprovider @PytestArgs

exit $LASTEXITCODE
