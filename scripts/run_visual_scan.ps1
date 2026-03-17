function Get-PythonCommand {
    $venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    return "python"
}

$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."

$env:MPLBACKEND = "Agg"
$env:DOCS_BASE_URL = "http://127.0.0.1:8000/option-pricing-library/"
$python = Get-PythonCommand

Write-Host "Rendering diagrams..."
& $python .\scripts\render_d2_diagrams.py

Write-Host "Building generated visual assets..."
& $python .\scripts\build_visual_artifacts.py all --profile ci

Write-Host "Building MkDocs site strictly..."
& $python -m mkdocs build --strict

Write-Host "Running existing generated asset integrity scan..."
& $python .\scripts\visual_audit\check_svg_assets.py

Write-Host "Writing static visual state report..."
& $python .\scripts\visual_audit\report_visual_state.py

Push-Location .\tests\visual
try {
    $env:SKIP_DOCS_PREBUILD = "1"
    Write-Host "Running targeted embedded-panel rendering scan..."
    npx playwright test embedded-panels.spec.ts
}
finally {
    Remove-Item Env:SKIP_DOCS_PREBUILD -ErrorAction SilentlyContinue
    Pop-Location
}
