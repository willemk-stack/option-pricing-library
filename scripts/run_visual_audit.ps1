param(
    [switch]$UpdateSnapshots
)

function Get-PythonCommand {
    $venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    return "python"
}

$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."

$python = Get-PythonCommand
$mode = if ($UpdateSnapshots) { "update" } else { "verify" }
& $python .\scripts\run_local_visual_regression.py $mode
