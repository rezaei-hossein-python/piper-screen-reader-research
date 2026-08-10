[CmdletBinding()] param()
$root=Split-Path -Parent $PSScriptRoot;$p=Join-Path $root 'listening\phase2aj';if(!(Test-Path -LiteralPath $p)){throw 'Run run-phase2aj.ps1 first'};Start-Process explorer.exe -ArgumentList $p
