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
if ($UpdateSnapshots) {
    & $python .\scripts\run_ci_visual_regression.py update
    exit $LASTEXITCODE
}

& $python .\scripts\run_ci_visual_regression.py verify --tests smoke.spec.ts dom-audits.spec.ts math-audits.spec.ts a11y.spec.ts
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python .\scripts\run_ci_visual_regression.py verify --skip-build
exit $LASTEXITCODE
