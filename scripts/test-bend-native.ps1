param(
  [string]$VcVars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
)
$ErrorActionPreference = 'Stop'
$bendBuild = Join-Path $env:TEMP 'cuda-webshader-bend-native'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
  node scripts/prepare-bend-native.mjs $bendBuild
  if ($LASTEXITCODE -ne 0) { throw 'Bend native fixture generation failed' }
  if (!(Test-Path -LiteralPath $VcVars)) { throw 'Pass -VcVars pointing to vcvars64.bat' }
  $bendSource = Join-Path $bendBuild 'bend-native.cu'
  $bendExe = Join-Path $bendBuild 'bend-native.exe'
  $buildCommand = 'call "' + $VcVars + '" && nvcc -O2 --fmad=false -std=c++17 -arch=native "' + $bendSource + '" -o "' + $bendExe + '"'
  & cmd.exe /d /c $buildCommand
  if ($LASTEXITCODE -ne 0) { throw 'NVCC compilation failed' }
  $bendNativeOutput = & $bendExe
  $bendNativeExit = $LASTEXITCODE
  $bendNativeOutput | Write-Output
  [System.IO.File]::WriteAllLines((Join-Path (Get-Location) 'reports/bend-native.txt'), [string[]]$bendNativeOutput, [System.Text.UTF8Encoding]::new($false))
  if ($bendNativeExit -ne 0) { throw 'Native Bend runtime checks failed' }
} finally { Pop-Location }
