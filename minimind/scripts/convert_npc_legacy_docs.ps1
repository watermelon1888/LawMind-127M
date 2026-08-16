param(
    [Parameter(Mandatory = $true)]
    [string]$InputRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-InventoryPath {
    param(
        [string]$Root,
        [string]$RelativePath
    )

    if ([string]::IsNullOrWhiteSpace($RelativePath)) {
        throw "文档清单包含空路径"
    }
    $normalized = $RelativePath.Replace('/', [System.IO.Path]::DirectorySeparatorChar)
    $resolved = [System.IO.Path]::GetFullPath((Join-Path $Root $normalized))
    $prefix = $Root.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "文档清单路径越界: $RelativePath"
    }
    return $resolved
}

function Test-OpenXmlDocument {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    $archive = $null
    try {
        $archive = [System.IO.Compression.ZipFile]::OpenRead($Path)
        return $null -ne $archive.GetEntry("word/document.xml")
    }
    catch {
        return $false
    }
    finally {
        if ($null -ne $archive) {
            $archive.Dispose()
        }
    }
}

Add-Type -AssemblyName System.IO.Compression.FileSystem

$root = (Resolve-Path -LiteralPath $InputRoot).Path
$inventoryPath = Join-Path $root "document-inventory.jsonl"
if (-not (Test-Path -LiteralPath $inventoryPath -PathType Leaf)) {
    throw "缺少文档清单: $inventoryPath"
}

$records = @(
    Get-Content -LiteralPath $inventoryPath -Encoding UTF8 |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { $_ | ConvertFrom-Json }
)
$legacyRecords = @($records | Where-Object { $_.original_extension -eq ".doc" })
$manifestPath = Join-Path $root "conversion-manifest.jsonl"
$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)

if ($legacyRecords.Count -eq 0) {
    [System.IO.File]::WriteAllText($manifestPath, "", $utf8WithoutBom)
    Write-Output "[完成] 没有需要转换的旧 DOC 文档"
    exit 0
}

$word = $null
$writer = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $word.Options.SaveNormalPrompt = $false
    $writer = New-Object System.IO.StreamWriter($manifestPath, $false, $utf8WithoutBom)

    $processed = 0
    foreach ($record in $legacyRecords) {
        $sourcePath = Resolve-InventoryPath $root ([string]$record.original_path)
        $targetPath = Resolve-InventoryPath $root ([string]$record.converted_path)
        if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
            throw "缺少旧 DOC 原文件: $sourcePath"
        }
        $sourceHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($sourceHash -ne ([string]$record.sha256).ToLowerInvariant()) {
            throw "旧 DOC 哈希与文档清单不一致: $($record.document_id)"
        }

        $targetDirectory = Split-Path -Parent $targetPath
        [System.IO.Directory]::CreateDirectory($targetDirectory) | Out-Null
        $status = "skipped"
        if (Test-Path -LiteralPath $targetPath) {
            if (-not (Test-OpenXmlDocument $targetPath)) {
                throw "已存在的转换文件不是合法 DOCX: $targetPath"
            }
        }
        else {
            $temporaryPath = "$targetPath.partial.docx"
            if (Test-Path -LiteralPath $temporaryPath) {
                throw "存在未处理的转换临时文件: $temporaryPath"
            }
            $document = $null
            try {
                $document = $word.Documents.Open($sourcePath, $false, $true, $false)
                $document.SaveAs2($temporaryPath, 16)
            }
            finally {
                if ($null -ne $document) {
                    $document.Close(0)
                    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
                }
            }
            if (-not (Test-OpenXmlDocument $temporaryPath)) {
                throw "Microsoft Word 未生成合法 DOCX: $($record.document_id)"
            }
            Move-Item -LiteralPath $temporaryPath -Destination $targetPath
            $status = "converted"
        }

        $targetFile = Get-Item -LiteralPath $targetPath
        $result = [ordered]@{
            document_id = [string]$record.document_id
            status = $status
            original_path = [string]$record.original_path
            original_sha256 = $sourceHash
            converted_path = [string]$record.converted_path
            converted_sha256 = (Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant()
            converted_size = $targetFile.Length
        }
        $writer.WriteLine(($result | ConvertTo-Json -Compress))
        $writer.Flush()
        $processed += 1
        if ($processed % 100 -eq 0 -or $processed -eq $legacyRecords.Count) {
            Write-Output "[转换] $processed/$($legacyRecords.Count)"
        }
    }
}
finally {
    if ($null -ne $writer) {
        $writer.Dispose()
    }
    if ($null -ne $word) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}

Write-Output "[完成] 转换 $($legacyRecords.Count) 个旧 DOC 文档"
