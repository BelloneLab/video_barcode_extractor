"""Entry point for the Video Barcode Signal Extractor.

Prints brief diagnostic lines so you can tell from the terminal whether
the app got past each stage. If something fails, the last printed line
identifies where.
"""
import sys
import traceback


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
