[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root 'environments\piper-research\Scripts\python.exe'
$piperRuntime = 'C:\projects\nvda piper addon\.phase2h-runtime\Lib\site-packages'
$model = 'C:\projects\nvda piper addon\.phase2h-assets\en_US-lessac-low\en_US-lessac-low.onnx'
$config = 'C:\projects\nvda piper addon\.phase2h-assets\en_US-lessac-low\en_US-lessac-low.onnx.json'
$rewritten = Join-Path $root 'models\generated\lessac-duration-override.onnx'
$expected = @{
    $model = 'f7d01dde371555732c4c314111ac79672b1a5ce2fc19266ab42178fd8df7f375'
    $config = '45754dfdebb3b8661c3fc564713772deec6e064feeb5b4e9594857dc7305193a'
    $rewritten = 'a5697871afeff4fdfa5e8f515a4241a63843512a51e588bccaeb9cbd5f16e480'
}
foreach ($path in $expected.Keys) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing locked artifact: $path" }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected[$path]) { throw "Hash mismatch: $path`nExpected $($expected[$path])`nActual   $actual" }
}
if (-not (Test-Path -LiteralPath (Join-Path $piperRuntime 'piper\espeakbridge.pyd'))) { throw "Missing locked Piper phonemizer runtime: $piperRuntime" }
$env:PYTHONPATH = (Join-Path $root 'experiments\onnx-duration-override') + [IO.Path]::PathSeparator + $piperRuntime
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$output = Join-Path $root 'results\phase2ai'
$listening = Join-Path $root 'listening\phase2ai'
$answerKey = Join-Path $root 'results\phase2ai-DO-NOT-OPEN-answer-key.json'
New-Item -ItemType Directory -Force -Path $output, $listening | Out-Null
& $python -m pytest (Join-Path $root 'tests') -q
if ($LASTEXITCODE -ne 0) { throw "Phase 2AI tests failed: $LASTEXITCODE" }
& $python (Join-Path $root 'experiments\onnx-duration-override\phase2ai_experiment.py') $model $rewritten $config $output $listening $answerKey
if ($LASTEXITCODE -ne 0) { throw "Phase 2AI experiment failed: $LASTEXITCODE" }
Write-Host "Phase 2AI complete. Listening files: $listening"
