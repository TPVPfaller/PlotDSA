@echo off
set "THIS_SCRIPT=%~f0"
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$text = Get-Content -Raw -LiteralPath $env:THIS_SCRIPT; $code = $text -split '# POWERSHELL START'; Invoke-Expression $code[-1]"
if errorlevel 1 pause
exit /b %errorlevel%

# POWERSHELL START

$VSCaptureWavePath = "C:\temp\VSCaptureWave\VSCaptureWave.exe"
$DsaPath = "C:\temp\VSCaptureWave\DSA.exe"
$BottomProcessName = "AnesthesiaUI"

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms

Add-Type @"
using System;
using System.Runtime.InteropServices;

public struct RECT
{
public int Left;
public int Top;
public int Right;
public int Bottom;
}

public static class WindowRect
{
[DllImport("user32.dll")]
public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
}
"@

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class NativeWindow
{
[DllImport("user32.dll", SetLastError=true)]
public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);


[DllImport("user32.dll", SetLastError=true)]
public static extern bool SetWindowPos(
    IntPtr hWnd,
    IntPtr hWndInsertAfter,
    int X,
    int Y,
    int cx,
    int cy,
    uint uFlags
);

[DllImport("user32.dll", SetLastError=true)]
public static extern int GetWindowLong(IntPtr hWnd, int nIndex);

[DllImport("user32.dll", SetLastError=true)]
public static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);

public static readonly IntPtr HWND_TOPMOST = new IntPtr(-1);
public static readonly IntPtr HWND_NOTOPMOST = new IntPtr(-2);

public const uint SWP_NOMOVE = 0x0002;
public const uint SWP_NOSIZE = 0x0001;

[DllImport("user32.dll")]
public static extern bool BringWindowToTop(IntPtr hWnd);

public const int SW_RESTORE = 9;

public const uint SWP_SHOWWINDOW = 0x0040;
public const uint SWP_FRAMECHANGED = 0x0020;

public const int GWL_STYLE = -16;

public const int WS_CAPTION   = 0x00C00000;
public const int WS_THICKFRAME = 0x00040000;
public const int WS_BORDER     = 0x00800000;


}
"@

function Start-ProcessIfMissing {
param(
[string]$ProcessName,
[string]$FilePath
)


if (-not (Get-Process -Name $ProcessName -ErrorAction SilentlyContinue)) {
    Write-Host "Starting $ProcessName..."
    Start-Process -FilePath $FilePath -WorkingDirectory (Split-Path $FilePath) | Out-Null
}
else {
    Write-Host "$ProcessName already running."
}


}

function Wait-MainWindowHandle {
param(
[string]$ProcessName,
[int]$TimeoutSeconds = 30
)

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)

do {
    $process = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowHandle -ne 0 } |
        Select-Object -First 1

    if ($process) {
        return $process.MainWindowHandle
    }

    Start-Sleep -Milliseconds 250
}
while ((Get-Date) -lt $deadline)

throw "Could not find visible window for process '$ProcessName'."


}

function Position-Window {
param(
[string]$ProcessName,
[int]$X,
[int]$Y,
[int]$Width,
[int]$Height
)


Write-Host "Positioning $ProcessName"

$handle = Wait-MainWindowHandle -ProcessName $ProcessName

[NativeWindow]::ShowWindow(
    $handle,
    [NativeWindow]::SW_RESTORE
) | Out-Null

Start-Sleep -Milliseconds 500


Start-Sleep -Milliseconds 100

[NativeWindow]::SetWindowPos(
    $handle,
    [IntPtr]::Zero,
    $X,
    $Y,
    $Width,
    $Height,
    [NativeWindow]::SWP_SHOWWINDOW -bor [NativeWindow]::SWP_FRAMECHANGED
) | Out-Null

[NativeWindow]::BringWindowToTop($handle) | Out-Null

Start-Sleep -Milliseconds 250

$rect = New-Object RECT
[WindowRect]::GetWindowRect($handle, [ref]$rect) | Out-Null



Write-Host ""
Write-Host "$ProcessName actual position:"
Write-Host "Left   = $($rect.Left)"
Write-Host "Top    = $($rect.Top)"
Write-Host "Right  = $($rect.Right)"
Write-Host "Bottom = $($rect.Bottom)"
Write-Host "Width  = $($rect.Right - $rect.Left)"
Write-Host "Height = $($rect.Bottom - $rect.Top)"
Write-Host ""

}

#

# Start applications

#

Start-ProcessIfMissing -ProcessName "VSCaptureWave" -FilePath $VSCaptureWavePath

$DsaWasRunning = $null -ne (Get-Process -Name "DSA" -ErrorAction SilentlyContinue)

Start-ProcessIfMissing -ProcessName "DSA" -FilePath $DsaPath

if (-not $DsaWasRunning) {
Write-Host "Waiting for DSA window..."
Wait-MainWindowHandle -ProcessName "DSA" -TimeoutSeconds 60 | Out-Null
Write-Host "DSA window detected."
}

#

# Screen layout

#

$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds

$topHeight = [int]($screen.Height * 0.25)
$bottomHeight = $screen.Height - $topHeight

Write-Host "Screen:"
Write-Host "  Width  = $($screen.Width)"
Write-Host "  Height = $($screen.Height)"


$Overlap = 8

$ExtraWidth = 16

Position-Window `
    -ProcessName "DSA" `
    -X -8 `
    -Y 0 `
    -Width ($screen.Width + $ExtraWidth) `
    -Height $topHeight

Position-Window `
    -ProcessName "AnesthesiaUI" `
    -X -8 `
    -Y ($screen.Y + $topHeight - $Overlap) `
    -Width ($screen.Width + $ExtraWidth) `
    -Height ($bottomHeight + $Overlap)

foreach ($proc in @("DSA","AnesthesiaUI"))
{
    $h = Wait-MainWindowHandle -ProcessName $proc

    [NativeWindow]::SetWindowPos(
        $h,
        [NativeWindow]::HWND_TOPMOST,
        0,0,0,0,
        [NativeWindow]::SWP_NOMOVE -bor [NativeWindow]::SWP_NOSIZE
    ) | Out-Null

    [NativeWindow]::SetWindowPos(
        $h,
        [NativeWindow]::HWND_NOTOPMOST,
        0,0,0,0,
        [NativeWindow]::SWP_NOMOVE -bor [NativeWindow]::SWP_NOSIZE
    ) | Out-Null
}

Write-Host "Layout complete."
