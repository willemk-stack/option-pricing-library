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

if ($env:SKIP_DOCS_PREBUILD -ne "1") {
	Write-Host "Rendering diagrams..."
	& $python .\scripts\render_d2_diagrams.py

	Write-Host "Building generated visual assets..."
	& $python .\scripts\build_visual_artifacts.py all --profile ci
}

Write-Host "Starting MkDocs server..."
& $python -m mkdocs serve -a 127.0.0.1:8000
