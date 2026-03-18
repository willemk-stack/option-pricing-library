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

$config = Get-Content .\scripts\visual_audit\docs_visual_config.json | ConvertFrom-Json
$env:MPLBACKEND = "Agg"
$env:DOCS_BASE_URL = $config.docs_base_url
$env:DOCS_VISUAL_BUILD_PROFILE = $config.build_profile
$python = Get-PythonCommand

Write-Host "Rendering diagrams..."
& $python .\scripts\render_d2_diagrams.py

Write-Host "Building generated visual assets..."
& $python .\scripts\build_visual_artifacts.py all --profile $config.build_profile

Write-Host "Building MkDocs site strictly..."
& $python -m mkdocs build --strict

Write-Host "Checking generated asset integrity..."
& $python .\scripts\visual_audit\check_svg_assets.py

Push-Location .\tests\visual
try {
    $env:SKIP_DOCS_PREBUILD = "1"
    $env:SERVE_PREBUILT_SITE = "1"
    if ($UpdateSnapshots) {
        Write-Host "Refreshing Playwright screenshot baselines..."
        npx playwright test pages.spec.ts --update-snapshots
    }
    else {
        Write-Host "Running full Playwright suite..."
        npx playwright test
    }
}
finally {
    Remove-Item Env:SKIP_DOCS_PREBUILD -ErrorAction SilentlyContinue
    Remove-Item Env:SERVE_PREBUILT_SITE -ErrorAction SilentlyContinue
    Pop-Location
}