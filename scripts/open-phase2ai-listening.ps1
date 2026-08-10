[CmdletBinding()]
param()
$root = Split-Path -Parent $PSScriptRoot
$listening = Join-Path $root 'listening\phase2ai'
if (-not (Test-Path -LiteralPath $listening)) { throw "Phase 2AI listening directory does not exist. Run scripts\run-phase2ai.ps1 first." }
Start-Process explorer.exe -ArgumentList $listening
