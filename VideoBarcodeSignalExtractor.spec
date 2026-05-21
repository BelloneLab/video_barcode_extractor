# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

hiddenimports = [
    "skimage.filters",
    "skimage.measure",
]

conda_bin = Path(r"C:\ProgramData\anaconda3\Library\bin")
extra_binaries = [
    (str(conda_bin / name), ".")
    for name in ("liblzma.dll", "libbz2.dll", "ffi.dll")
    if (conda_bin / name).exists()
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=extra_binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["av"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VideoBarcodeSignalExtractor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="VideoBarcodeSignalExtractor",
)
