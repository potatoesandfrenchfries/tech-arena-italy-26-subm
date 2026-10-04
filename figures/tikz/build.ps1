# Build the TikZ slide diagrams to PDF, SVG and PNG next to the sources.
# Run from Windows PowerShell (MiKTeX/TeX Live is installed on the Windows side, not in WSL):
#   powershell -File figures\tikz\build.ps1 [name ...]
# Compiles in a local temp folder because pdflatex cannot use a UNC path as its working directory.
param([string[]]$Names)

$src = $PSScriptRoot
$work = Join-Path $env:TEMP "tikz_build"
New-Item -ItemType Directory -Force $work | Out-Null
Copy-Item (Join-Path $src "*.tex") $work -Force

if (-not $Names) { $Names = Get-ChildItem $work -Filter *.tex | Where-Object { $_.Name -ne "preamble.tex" } | ForEach-Object { $_.BaseName } }

foreach ($n in $Names) {
    Push-Location $work
    pdflatex -interaction=nonstopmode -halt-on-error "$n.tex" | Out-Null
    $ok = $LASTEXITCODE -eq 0
    if ($ok) {
        pdftocairo -svg "$n.pdf" "$n.svg"
        pdftocairo -png -r 300 -singlefile "$n.pdf" $n
        Copy-Item "$n.pdf", "$n.svg", "$n.png" $src -Force
        Write-Host "built $n"
    } else {
        Write-Host "FAILED $n (see $work\$n.log)"
        Get-Content "$n.log" | Select-String -Pattern "^!" -Context 0,3 | Select-Object -First 5
    }
    Pop-Location
}
