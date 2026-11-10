# VKR Windows launcher. Run from repo root, on the feature/vkr-stage1-typed-graph-holdout branch.
# Examples:
# powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode quick
# powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode main
# powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode app
# powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode study-quick
param(
    [ValidateSet("quick", "main", "study-quick", "study-full", "compare-quick", "compare-full", "genre", "faithfulness", "faithfulness-extended", "explanations", "explanations-depth4", "explanations-budget", "explanations-optimized", "explanations-paired", "tests", "app")]
    [string]$Mode = "quick",
    [switch]$SkipInstall,
    [switch]$UseGenres,
    [string]$ContactEmail = "",
    [ValidateRange(1, 25)]
    [int]$GenreBatchSize = 12
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
if ($Mode -eq "faithfulness") {
    Write-Host "Faithfulness path-ablation pilot: test-data-safe graph artifacts, 2 pairs, 2 matched controls"
    & $python -m src.vkr_faithfulness --max-pairs 2 --controls 2
    if ($LASTEXITCODE -ne 0) { throw "Faithfulness pilot failed" }
    Write-Host "See results/faithfulness/faithfulness_report.json and faithfulness_pairs.csv"
    exit 0
}
if ($Mode -eq "faithfulness-extended") {
    Write-Host "Extended train-only faithfulness ablation, 4 pairs, 20 degree-aware null controls per pair"
    & $python -m src.vkr_faithfulness --max-pairs 4 --controls 20 --output-dir results/faithfulness_extended
    if ($LASTEXITCODE -ne 0) { throw "Extended faithfulness study failed" }
    Write-Host "See results/faithfulness_extended/faithfulness_report.json and faithfulness_pairs.csv"
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
if ($Mode -eq "genre") {
    if ([string]::IsNullOrWhiteSpace($ContactEmail)) {
        throw "For Open Library requests provide -ContactEmail with your real contact email."
    }
    if (-not (Test-Path "data\\processed\\vkr_train_ratings.csv")) {
        throw "Training ratings missing: run -Mode quick or -Mode main first."
    }
    & $python -m src.vkr_genre_enrichment --train-csv data/processed/vkr_train_ratings.csv --output-csv data/raw/book_genres.csv --contact $ContactEmail --max-isbns $GenreBatchSize
    if ($LASTEXITCODE -ne 0) { throw "Small genre enrichment failed" }
    Write-Host "Genre facts saved in data/raw/book_genres.csv. Run quick/main with -UseGenres to add them to a new graph."
    exit 0
}
$genreArgs = @()
if ($UseGenres) {
    if (-not (Test-Path "data\\raw\\book_genres.csv")) {
        throw "Genre CSV missing; first run -Mode genre (small sample) or supply a verified data/raw/book_genres.csv."
    }
    $genreArgs = @("--genre-csv", "data/raw/book_genres.csv")
}
if ($Mode -eq "explanations-paired") {
    Write-Host "Paired user-level bootstrap after validation selection, fresh split seed 2082"
    & $python -m src.vkr_explanation_study --output-dir results/explanation_study_paired --max-users 150 --max-books 1000 --max-evaluated-users 80 --candidate-pool 25 --max-depth 4 --split-seed 2082 --max-paths 200 --max-expansions 5000 @genreArgs
    if ($LASTEXITCODE -ne 0) { throw "Paired explanation study failed" }
    Write-Host "See results/explanation_study_paired/explanation_study_report.json, explanation_test_paired_deltas.csv"
    exit 0
}
if ($Mode -eq "explanations-optimized") {
    Write-Host "Optimized typed 4-hop join on a fresh split seed 2072; test evaluated after validation selection"
    & $python -m src.vkr_explanation_study --output-dir results/explanation_study_optimized --max-users 150 --max-books 1000 --max-evaluated-users 40 --candidate-pool 25 --max-depth 4 --split-seed 2072 --max-paths 200 --max-expansions 5000 @genreArgs
    if ($LASTEXITCODE -ne 0) { throw "Optimized explanation study failed" }
    Write-Host "See results/explanation_study_optimized/explanation_study_report.json"
    exit 0
}
if ($Mode -eq "explanations-budget") {
    Write-Host "High-budget explanation method study on a fresh split 2062, depth4, 40 evaluation users"
    & $python -m src.vkr_explanation_study --output-dir results/explanation_study_budget --max-users 150 --max-books 1000 --max-evaluated-users 40 --candidate-pool 25 --max-depth 4 --split-seed 2062 --max-paths 200 --max-expansions 5000 @genreArgs
    if ($LASTEXITCODE -ne 0) { throw "High-budget explanation study failed" }
    Write-Host "See results/explanation_study_budget/explanation_study_report.json"
    exit 0
}
if ($Mode -eq "explanations-depth4") {
    Write-Host "Independent depth-4 explanation study (new split seed 2052). Output: results/explanation_study_depth4"
    & $python -m src.vkr_explanation_study --output-dir results/explanation_study_depth4 --max-users 150 --max-books 1000 --max-evaluated-users 40 --candidate-pool 25 --max-depth 4 --split-seed 2052 @genreArgs
    if ($LASTEXITCODE -ne 0) { throw "Depth-4 explanation study failed" }
    Write-Host "See results/explanation_study_depth4/explanation_study_report.json"
    exit 0
}
if ($Mode -eq "explanations") {
    Write-Host "Explanation study: fixed train-only Node2Vec, validation alpha selection, test once"
    & $python -m src.vkr_explanation_study --max-users 150 --max-books 1000 --max-evaluated-users 20 --candidate-pool 25 --max-depth 2 @genreArgs
    if ($LASTEXITCODE -ne 0) { throw "Explanation validation study failed" }
    Write-Host "See results/explanation_study/explanation_study_report.json and explanation_validation_grid.csv"
    exit 0
}
if ($Mode -eq "compare-quick" -or $Mode -eq "compare-full") {
    $ComparisonMode = if ($Mode -eq "compare-quick") { "quick" } else { "full" }
    Write-Host "Node2Vec vs TransE comparative study ($ComparisonMode). Output: results/comparison"
    & $python -m src.vkr_compare_embeddings --mode $ComparisonMode --max-users 150 --max-books 1000 @genreArgs
    if ($LASTEXITCODE -ne 0) { throw "Comparative study failed" }
    Write-Host "Comparison completed. See results/comparison/comparison_report.json and comparison_experiments.csv"
    exit 0
}
if ($Mode -eq "study-quick" -or $Mode -eq "study-full") {
    $StudyMode = if ($Mode -eq "study-quick") { "quick" } else { "full" }
    Write-Host "Node2Vec train/validation/test study ($StudyMode). Output: results/study"
    & $python -m src.vkr_embedding_study --mode $StudyMode --max-users 150 --max-books 1000 @genreArgs
    if ($LASTEXITCODE -ne 0) { throw "Embedding study failed" }
    Write-Host "Study completed. See results/study/study_report.json and embedding_experiments.csv"
    exit 0
}
if ($Mode -eq "quick") {
    Write-Host "Quick run: 150 users, 1000 books; this replaces data/processed and results baseline."
    & $python -m src.vkr_pipeline --max-users 150 --max-books 1000 --dimensions 32 --walk-length 15 --num-walks 3 --epochs 2 @genreArgs
} else {
    Write-Host "Main run: up to 1000 users, 5000 books. This may take longer on Windows."
    & $python -m src.vkr_pipeline --max-users 1000 --max-books 5000 --dimensions 64 --walk-length 20 --num-walks 4 --epochs 3 @genreArgs
}
if ($LASTEXITCODE -ne 0) { throw "VKR pipeline failed" }
& $python -m src.vkr_graph_audit
if ($LASTEXITCODE -ne 0) { throw "Graph audit failed" }
Write-Host "Run succeeded. Open Streamlit:"
Write-Host "powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode app -SkipInstall"
