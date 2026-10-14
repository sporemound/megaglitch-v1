param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,
    [Parameter(Mandatory = $true)]
    [string]$Output,
    [int]$DelaySeconds = 4,
    [string]$ExpectedTitle = ''
)

$ErrorActionPreference = 'Stop'
$Executable = (Resolve-Path -LiteralPath $Executable).Path
$Output = [IO.Path]::GetFullPath((Join-Path (Get-Location) $Output))
$Process = Start-Process -FilePath $Executable -WorkingDirectory (Split-Path $Executable) -PassThru -WindowStyle Normal
try {
    Start-Sleep -Seconds $DelaySeconds
    if ($Process.HasExited) { throw "UI exited early with $($Process.ExitCode)." }
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public struct MgRect { public int Left; public int Top; public int Right; public int Bottom; }
public static class MgWindow {
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr handle, out MgRect rect);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr handle, IntPtr hdc, uint flags);
}
'@
    $Deadline = (Get-Date).AddSeconds(5)
    do {
        $Process.Refresh()
        $Handle = $Process.MainWindowHandle
        if ($Handle -ne [IntPtr]::Zero) { break }
        Start-Sleep -Milliseconds 100
    } while ((Get-Date) -lt $Deadline)
    if ($Handle -eq [IntPtr]::Zero) { throw 'UI window handle was not available.' }
    if (-not [string]::IsNullOrWhiteSpace($ExpectedTitle) -and $Process.MainWindowTitle -notmatch $ExpectedTitle) {
        throw "UI title '$($Process.MainWindowTitle)' did not match '$ExpectedTitle'."
    }
    Write-Output "PID=$($Process.Id) TITLE=$($Process.MainWindowTitle)"
    Add-Type -AssemblyName System.Drawing
    $Rect = New-Object MgRect
    if (-not [MgWindow]::GetWindowRect($Handle, [ref]$Rect)) { throw 'Could not read UI window bounds.' }
    $Width = $Rect.Right - $Rect.Left
    $Height = $Rect.Bottom - $Rect.Top
    $Bitmap = New-Object System.Drawing.Bitmap $Width, $Height
    try {
        $Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)
        try {
            $Hdc = $Graphics.GetHdc()
            try {
                if (-not [MgWindow]::PrintWindow($Handle, $Hdc, 2)) { throw 'PrintWindow failed.' }
            }
            finally { $Graphics.ReleaseHdc($Hdc) }
            $Bitmap.Save($Output, [System.Drawing.Imaging.ImageFormat]::Png)
        }
        finally { $Graphics.Dispose() }
    }
    finally { $Bitmap.Dispose() }
}
finally {
    if (-not $Process.HasExited) {
        [void]$Process.CloseMainWindow()
        if (-not $Process.WaitForExit(5000)) { Stop-Process -Id $Process.Id -Force }
    }
}
Write-Output $Output
