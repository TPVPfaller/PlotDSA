# PlotDSA - EEG Density Spectral Array (DSA) Viewer

PlotDSA is a PySide6 application for real-time EEG visualization, featuring a scrolling Density Spectral Array (DSA) heatmap and live raw EEG trace.

## Quick Start
1. **Install dependencies**: `python -m pip install -r requirements.txt`
2. **Run the app**: `python -m plotdsa.app.main`
3. **Optional emulator**: `python -m plotdsa.tools.emulator` (streams data from `Entropy_Data/` to the `EEG_DATA` LSL stream)

## Key Features
- **Real-time Visualization**: Heatmap (frequency vs. time) and raw EEG trace via the `EEG_DATA` LSL stream.
- **Configurable Settings**: Adjust analysis window, overlap, and max frequency in-app.
- **Data Export**: Saves computed PSD data to CSV under `C:\temp\VSCaptureWave`.

## Development
- **Requirements**: Python 3.11, Windows (primary target).
- **Tests**: Run `pytest -q`.
- **Build**: Use Nuitka to produce a Windows executable.

## Build Single-File Executable (Windows)

Install the build dependencies:
```bash
python -m pip install nuitka zstandard
```

Build the executable:
```powershell
python -m nuitka --onefile --standalone `
  --enable-plugin=pyside6 `
  --include-module=PySide6.QtOpenGL `
  --include-data-file=".venv\Lib\site-packages\pylsl\lib\lsl.dll=pylsl/lib/lsl.dll" `
  --windows-icon-from-ico=app_icon.ico `
  --windows-disable-console `
  --windows-product-version=1.0.0.0 `
  --windows-file-version=1.0.0.0 `
  --windows-product-name="DSA" `
  --windows-file-description="DSA Application" `
  --windows-company-name="Your Name or Company" `
  --output-filename=DSA.exe `
  plotdsa\app\main.py
```
