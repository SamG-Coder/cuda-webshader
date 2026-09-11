param([string]$VcVars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
node scripts/prepare-comparison.mjs
if ($LASTEXITCODE -ne 0) { throw 'Fixture generation failed' }
if (!(Test-Path -LiteralPath $VcVars)) { throw 'Pass -VcVars with the path to your Visual Studio vcvars64.bat' }
$buildCommand = 'call "' + $VcVars + '" && nvcc -O3 -std=c++17 -arch=native reports/comparison-inputs/native-generated.cu -o reports/native-benchmark.exe && nvcc -O3 -std=c++17 -arch=native tests/native-reference.cu -o reports/native-reference.exe'
& cmd.exe /d /c $buildCommand
if ($LASTEXITCODE -ne 0) { throw 'NVCC compilation failed' }
