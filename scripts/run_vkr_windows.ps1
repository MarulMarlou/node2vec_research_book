# VKR Windows launcher. Run from repo root, on the feature/vkr-stage1-typed-graph-holdout branch.
# Examples:
# powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode quick
# powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode main
# powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode app
# powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode study-quick
param(
    [ValidateSet("quick", "main", "study-quick", "study-full", "compare-quick", "compare-full", "tests", "app")]
    [string]$Mode = "quick",
    [switch]$SkipInstall
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "src\vkr_pipeline.py")) {
    throw "VKR files not found. Switch to feature/vkr-stage1-typed-graph-holdout first."
}
if (-not (Test-Path "data\raw")) {
    New-Item -Type Directory -Path "data\raw" -Force | Out-Null
}
$python = Join-Path (Get-Location).Path ".venv-vkr\Scripts\python.exe"
if (-not (Test-Path $python)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python launcher 'py' missing. Install Python 3.11 from python.org."
    }
    & py -3.11 --version
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.11 required. Install it and rerun this script."
    }
    & py -3.11 -m venv .venv-vkr
    if ($LASTEXITCODE -ne 0) { throw "Cannot create virtual environment" }
}
if (-not $SkipInstall) {
    & $python -m pip install -r requirements-vkr-app.txt
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
}
if ($Mode -eq "tests") {
    & $python -m pytest -q tests
    if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
    exit 0
}
if ($Mode -eq "app") {
    if (-not (Test-Path "results\embeddings\vkr_n2v.npy")) {
        throw "Embeddings missing: run the 'quick' or 'main' mode first."
    }
    & $python -m streamlit run src/vkr_app.py
    exit $LASTEXITCODE
}
$required = @("data\raw\BX-Books.csv", "data\raw\BX-Book-Ratings.csv")
foreach ($file in $required) {
    if (-not (Test-Path $file)) {
        throw "Required Book-Crossing CSV missing: $file"
    }
}
if ($Mode -eq "compare-quick" -or $Mode -eq "compare-full") {
    $ComparisonMode = if ($Mode -eq "compare-quick") { "quick" } else { "full" }
    Write-Host "Node2Vec vs TransE comparative study ($ComparisonMode). Output: results/comparison"
    & $python -m src.vkr_compare_embeddings --mode $ComparisonMode --max-users 150 --max-books 1000
    if ($LASTEXITCODE -ne 0) { throw "Comparative study failed" }
    Write-Host "Comparison completed. See results/comparison/comparison_report.json and comparison_experiments.csv"
    exit 0
}
if ($Mode -eq "study-quick" -or $Mode -eq "study-full") {
    $StudyMode = if ($Mode -eq "study-quick") { "quick" } else { "full" }
    Write-Host "Node2Vec train/validation/test study ($StudyMode). Output: results/study"
    & $python -m src.vkr_embedding_study --mode $StudyMode --max-users 150 --max-books 1000
    if ($LASTEXITCODE -ne 0) { throw "Embedding study failed" }
    Write-Host "Study completed. See results/study/study_report.json and embedding_experiments.csv"
    exit 0
}
if ($Mode -eq "quick") {
    Write-Host "Quick run: 150 users, 1000 books; this replaces data/processed and results baseline."
    & $python -m src.vkr_pipeline --max-users 150 --max-books 1000 --dimensions 32 --walk-length 15 --num-walks 3 --epochs 2
} else {
    Write-Host "Main run: up to 1000 users, 5000 books. This may take longer on Windows."
    & $python -m src.vkr_pipeline --max-users 1000 --max-books 5000 --dimensions 64 --walk-length 20 --num-walks 4 --epochs 3
}
if ($LASTEXITCODE -ne 0) { throw "VKR pipeline failed" }
& $python -m src.vkr_graph_audit
if ($LASTEXITCODE -ne 0) { throw "Graph audit failed" }
Write-Host "Run succeeded. Open Streamlit:"
Write-Host "powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode app -SkipInstall"
