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

& $python .\scripts\run_docs_browser_audits.py verify --build --tests smoke.spec.ts dom-audits.spec.ts math-audits.spec.ts --project chromium-375 --project chromium-1280 --findings-json .\artifacts\docs-audit\findings.json
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python .\scripts\run_docs_browser_audits.py verify --skip-build --tests a11y.spec.ts --project chromium-1280 --findings-json .\artifacts\docs-audit\a11y-findings.json
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python .\scripts\run_ci_visual_regression.py verify --skip-build
exit $LASTEXITCODE
