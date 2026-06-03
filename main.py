"""Entry point for the Video Barcode Signal Extractor.

Prints brief diagnostic lines so you can tell from the terminal whether
the app got past each stage. If something fails, the last printed line
identifies where.
"""
import os
import sys
import traceback


_DLL_DIR_HANDLES = []


def _prepare_conda_dll_paths():
    """Make direct conda-env python.exe launches work on Windows.

    VS Code can run the selected interpreter directly without activating the
    Conda environment first. In that case compiled packages such as NumPy,
    SciPy, OpenCV, and Qt may abort the process because Windows cannot find
    the environment DLLs. Add the same directories Conda activation would put
    on PATH before importing the application.
    """
    if os.name != 'nt':
        return
    prefix = os.environ.get('CONDA_PREFIX') or sys.prefix
    if not prefix:
        return
    candidates = [
        prefix,
        os.path.join(prefix, 'Library', 'mingw-w64', 'bin'),
        os.path.join(prefix, 'Library', 'usr', 'bin'),
        os.path.join(prefix, 'Library', 'bin'),
        os.path.join(prefix, 'Scripts'),
    ]
    existing = [p for p in candidates if os.path.isdir(p)]
    if not existing:
        return

    current_path = os.environ.get('PATH', '')
    current_parts = [p for p in current_path.split(os.pathsep) if p]
    lower_parts = {p.lower() for p in current_parts}
    prepend = [p for p in existing if p.lower() not in lower_parts]
    if prepend:
        os.environ['PATH'] = os.pathsep.join(prepend + current_parts)

    add_dll_directory = getattr(os, 'add_dll_directory', None)
    if add_dll_directory is None:
        return
    for path in existing:
        try:
            _DLL_DIR_HANDLES.append(add_dll_directory(path))
        except OSError:
            pass


def _install_qt_handler():
    try:
        from PyQt5.QtCore import qInstallMessageHandler
        def _h(_mode, _ctx, msg):
            sys.stderr.write(f'[Qt] {msg}\n')
            sys.stderr.flush()
        qInstallMessageHandler(_h)
    except Exception as e:
        sys.stderr.write(f'(no Qt handler: {e})\n')


if __name__ == '__main__':
    print('[1/4] starting', flush=True)
    _prepare_conda_dll_paths()
    _install_qt_handler()
    try:
        print('[2/4] importing vbe.app', flush=True)
        from vbe.app import run
        print('[3/4] calling run()', flush=True)
        rc = run(sys.argv)
        print(f'[4/4] event loop returned {rc}', flush=True)
        sys.exit(rc)
    except SystemExit:
        raise
    except BaseException:
        print('--- launch failed: traceback follows ---', flush=True)
        traceback.print_exc()
        sys.exit(1)
