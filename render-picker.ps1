param([string]$Image)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
if ([string]::IsNullOrWhiteSpace($Image)) {
    $picker = New-Object System.Windows.Forms.OpenFileDialog
    $picker.Title = 'Choose an image for full-frame MegaGlitch'
    $picker.Filter = 'Images|*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff;*.gif|All files|*.*'
    $picker.InitialDirectory = Join-Path $PSScriptRoot 'preset'
    if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 0 }
    $Image = $picker.FileName
    $picker.Dispose()
}
$sourceImage = (Resolve-Path -LiteralPath $Image).Path
$save = New-Object System.Windows.Forms.SaveFileDialog
$save.Title = 'Save the 60-second 4K MegaGlitch video'
$save.Filter = 'MP4 video|*.mp4'
$save.InitialDirectory = Join-Path $PSScriptRoot 'outputs'
$save.FileName = [System.IO.Path]::GetFileNameWithoutExtension($sourceImage) + '-Full-Frame-4K-60s.mp4'
$save.OverwritePrompt = $true
if ($save.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 0 }
$exportPath = $save.FileName
$save.Dispose()
if (Test-Path -LiteralPath $exportPath) { throw 'Choose a new filename. Existing exports are preserved.' }
Write-Host 'Rendering full-frame liquid warps, RGB trails and broken bands with the included soundtrack.'
Write-Host 'The finished video will be saved to:' $exportPath
& (Join-Path $PSScriptRoot 'runtime/python.exe') -I (Join-Path $PSScriptRoot 'preset/render.py') --image $sourceImage --output $exportPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
[System.Windows.Forms.MessageBox]::Show("Export complete.`n$exportPath", 'MegaGlitch', 'OK', 'Information') | Out-Null
