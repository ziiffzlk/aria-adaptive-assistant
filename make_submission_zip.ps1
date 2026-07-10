# Builds aria-submission.zip on the Desktop containing only submittable files:
# excludes the personal database, API keys, the venv, caches, and the large
# model binaries (those are re-downloadable; see requirements.txt).
#
# Run from the project folder:  powershell -ExecutionPolicy Bypass -File make_submission_zip.ps1

$project = $PSScriptRoot
$dest    = Join-Path ([Environment]::GetFolderPath("Desktop")) "aria-submission.zip"
$staging = Join-Path $env:TEMP "aria-submission-staging"

$excludeFiles = @(
    "aria.db", ".env",
    "kokoro-v1.0.onnx", "voices-v1.0.bin", "aria_voice_reference.wav",
    "make_submission_zip.ps1"
)
$excludeDirs = @(".venv", "__pycache__", "models", ".claude")

if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Force $staging | Out-Null

# Copy the project preserving folder structure, skipping the exclusions.
Get-ChildItem $project -Recurse -File | Where-Object {
    $rel = $_.FullName.Substring($project.Length + 1)
    $parts = $rel.Split([IO.Path]::DirectorySeparatorChar)
    (-not ($parts | Where-Object { $excludeDirs -contains $_ })) -and
    ($excludeFiles -notcontains $_.Name)
} | ForEach-Object {
    $rel = $_.FullName.Substring($project.Length + 1)
    $target = Join-Path $staging $rel
    New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
    Copy-Item $_.FullName $target
}

if (Test-Path $dest) { Remove-Item $dest }
Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $dest
$count = (Get-ChildItem $staging -Recurse -File).Count
Remove-Item $staging -Recurse -Force

Write-Host "Created $dest with $count files."
Write-Host "Excluded: aria.db, .env, .venv, model binaries, caches."
