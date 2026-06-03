# Video Barcode Signal Extractor

Video Barcode Signal Extractor is a PyQt5 desktop application for extracting LED or barcode synchronization traces from videos and aligning them to external reference signals.

## Features

- Load single videos or ordered video lists.
- Draw an ROI over the synchronization light source.
- Extract raw, smoothed, and binary synchronization signals.
- Load a pre-extracted source signal CSV instead of extracting from video.
- Load reference CSV, TXT, or TSV traces.
- Align video or imported signal time to reference time with cross-correlation, edge pairing, or DTW-assisted workflows.
- Export aligned signal CSV files and metadata.

## Source Signal CSV Workflow

If the barcode or sync trace was already extracted elsewhere, use
`File > Open source signal CSV...` or the `Open signal CSV` toolbar button.
Choose its time and signal columns, then open the reference CSV and run
auto-align as usual. The imported signal replaces the video ROI trace for
thresholding, plotting, cross-correlation, edge alignment, DTW, and export.

## Install From Source

```powershell
py -3.11 -m pip install -r requirements.txt
```

## Run

```powershell
py -3.11 main.py
```

You can also pass a video path:

```powershell
py -3.11 main.py "C:\path\to\video.mp4"
```

## Test

Some Python environments auto-load unrelated pytest plugins. To test only this project:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
py -3.11 -m pytest -q
```

## Build Windows App

This repository includes a PyInstaller spec for creating a Windows release folder:

```powershell
py -3.11 -m PyInstaller --clean --noconfirm VideoBarcodeSignalExtractor.spec
```

The built application is written to:

```text
dist\VideoBarcodeSignalExtractor\
```

Run:

```powershell
dist\VideoBarcodeSignalExtractor\VideoBarcodeSignalExtractor.exe
```
