"""Main application window for the Video Barcode Signal Extractor.

The pure-logic pieces live in `vbe.core` (cache, extraction, alignment,
CSV loading). The Qt widgets live in `vbe.widgets`. This module wires
the user interface and slots together.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QPointF, QSettings, QSize, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QAction, QApplication, QButtonGroup, QCheckBox,
    QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenuBar,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton,
    QSizePolicy, QSlider, QSpinBox, QSplitter, QStatusBar, QStyle,
    QTabWidget, QToolBar, QVBoxLayout, QWidget)
from scipy.ndimage import gaussian_filter1d

from vbe.core.alignment import (
    DEFAULT_MIN_PEAK_R, aligned_time_for_video_times, alignment_metrics,
    default_align_model, edge_times, estimate_xcorr_offset, norm01,
    pair_edges, ransac_regression, sanitize_align_model,
    video_time_from_aligned_time)
from vbe.core.cache import (
    FrameCache, clear_signal_cache, load_cache_metadata,
    load_signal_cache, save_signal_cache, signal_cache_key,
    signal_cache_meta_path, write_cache_metadata)
from vbe.core.csv_loader import load_csv_robust
from vbe.core.dtw import dtw_alignment
from vbe.core.extraction import ExtractionWorker
from vbe.core.quality import (
    edge_interval_stats, saturated_fraction, signal_snr_db)
from vbe.core.thresholds import (
    METHODS as THRESHOLD_METHODS, auto_threshold)
from vbe.theme import (
    BG, BG2, BG3, BLUE, CYAN, FG, FG_MUTED, GREEN, MAUVE, OVERLAY, PEACH, RED,
    stylesheet)
from vbe.widgets.video_widget import VideoWidget

# Note: pyqtgraph configuration is applied inside `run()` after QApplication
# is constructed. Some pyqtgraph builds touch QObject internals when
# setConfigOptions is called, which can complain if it runs at import time
# (before QApplication exists).


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Video Barcode Signal Extractor')
        self.resize(1500, 880)
        self.setAcceptDrops(True)
        self._apply_style()

        # video state
        self._cap        = None
        self._video_path = None
        self._fps        = 30.0
        self._n_frames   = 0
        self._cache      = FrameCache(120)
        self._worker     = None
        self._settings   = QSettings('NeuroVideoTools', 'VideoBarcodeSignalExtractor')
        self._last_dir   = str(Path.home())   # remembered browse directory
        self._last_video_dir = self._last_dir
        self._last_csv_dir = self._last_dir
        self._last_project_dir = self._last_dir
        self._last_export_dir = self._last_dir
        self._file_panel_visible = True
        self._file_panel_width = 285

        # signal data
        self._sig_t   = None
        self._sig_v   = None
        self._smooth_cache_key = None
        self._smooth_cache_val = None
        self._csv_df  = None
        self._csv_path = None
        self._sig_cache_key = None   # key of currently loaded signal
        self._items = []             # paired video/reference list
        self._current_item = -1
        self._default_roi = None
        self._loading_item = False
        self._last_loaded_signal_key = None
        self._loading_from_cache = False
        self._align_model = {
            'method': 'cross_correlation',
            'offset': 0.0,
            'slope': 1.0,
            'intercept': 0.0,
            'video_edges': [],
            'ref_edges': [],
            'edge_pairs': 0,
        }

        self._load_app_settings()

        self._build_ui()
        self._build_menu_bar()
        self._build_log_dock()
        self._connect_signals()
        self._update_nav_state()

    def _default_offset(self) -> float:
        spin = getattr(self, '_offset_spin', None)
        return float(spin.value()) if spin is not None else 0.0

    def _default_align_model(self):
        return default_align_model(self._default_offset())

    def _sanitize_align_model(self, model):
        return sanitize_align_model(model, default_offset=self._default_offset())

    def _load_app_settings(self):
        home = str(Path.home())
        self._last_dir = self._settings.value('folders/last', home, type=str)
        self._last_video_dir = self._settings.value('folders/video', self._last_dir, type=str)
        self._last_csv_dir = self._settings.value('folders/csv', self._last_dir, type=str)
        self._last_project_dir = self._settings.value('folders/project', self._last_dir, type=str)
        self._last_export_dir = self._settings.value('folders/export', self._last_dir, type=str)
        self._file_panel_visible = self._settings.value('ui/file_panel_visible', True, type=bool)
        self._file_panel_width = self._settings.value('ui/file_panel_width', 285, type=int)
        roi_text = self._settings.value('roi/last', '', type=str)
        if roi_text:
            try:
                roi = json.loads(roi_text)
                if isinstance(roi, list) and len(roi) == 4:
                    self._default_roi = tuple(int(v) for v in roi)
            except Exception:
                self._default_roi = None

    def _save_app_settings(self):
        self._save_current_item_state()
        if hasattr(self, '_root_splitter') and self._file_panel_visible:
            sizes = self._root_splitter.sizes()
            if sizes and sizes[0] > 0:
                self._file_panel_width = sizes[0]
        self._settings.setValue('folders/last', self._last_dir)
        self._settings.setValue('folders/video', self._last_video_dir)
        self._settings.setValue('folders/csv', self._last_csv_dir)
        self._settings.setValue('folders/project', self._last_project_dir)
        self._settings.setValue('folders/export', self._last_export_dir)
        self._settings.setValue('ui/file_panel_visible', self._file_panel_visible)
        self._settings.setValue('ui/file_panel_width', self._file_panel_width)
        roi = self._vid.get_roi() if hasattr(self, '_vid') else self._default_roi
        if roi:
            self._settings.setValue('roi/last', json.dumps(list(roi)))
        self._settings.sync()

    def _apply_style(self):
        self.setStyleSheet(stylesheet())

    @staticmethod
    def _theme_plot(plot):
        """Apply Nord styling to a pyqtgraph PlotWidget."""
        plot.setBackground(BG)
        pi = plot.getPlotItem()
        for axis_name in ('left', 'bottom', 'right', 'top'):
            ax = pi.getAxis(axis_name)
            ax.setPen(pg.mkPen(OVERLAY, width=1))
            ax.setTextPen(pg.mkPen(FG_MUTED))
            ax.setStyle(tickFont=QFont('Segoe UI', 9))

    # ── menu bar / log dock / shortcuts dialog ───────────────────────────────
    @staticmethod
    def _add_menu_action(menu, text, callback, shortcut=None):
        """Add an action without relying on the 3-arg `addAction` overload,
        which some PyQt5 builds reject because the third positional argument
        is `member` (old-style slot name) rather than a shortcut string."""
        action = menu.addAction(text)
        action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(shortcut)
        return action

    def _build_menu_bar(self):
        mb = self.menuBar()
        add = self._add_menu_action

        m_file = mb.addMenu('&File')
        add(m_file, 'Open video...', self._open_video, 'Ctrl+O')
        add(m_file, 'Open video list...', self._open_video_list)
        add(m_file, 'Open reference CSV...', self._open_csv, 'Ctrl+R')
        add(m_file, 'Open CSV list...', self._open_csv_list)
        m_file.addSeparator()
        add(m_file, 'Save project...', self._save_project, 'Ctrl+S')
        add(m_file, 'Load project...', self._load_project)
        m_file.addSeparator()
        add(m_file, 'Quit', self.close, 'Ctrl+Q')

        m_extract = mb.addMenu('&Extract')
        add(m_extract, 'Extract current', self._extract_signal, 'Ctrl+E')
        add(m_extract, 'Batch extract all loaded items', self._batch_extract)
        m_extract.addSeparator()
        add(m_extract, 'Clear signal cache for current video',
            self._clear_signal_cache)

        m_align = mb.addMenu('&Align')
        add(m_align, 'Auto-align (cross-correlation, RANSAC if regression)',
            self._auto_align, 'Ctrl+L')

        m_export = mb.addMenu('E&xport')
        add(m_export, 'Export current signal CSV...', self._export_signal,
            'Ctrl+Shift+E')
        add(m_export, 'Batch export aligned CSVs...', self._batch_export)
        m_export.addSeparator()
        add(m_export, 'Load aligned CSV...', self._open_aligned_signal)

        m_view = mb.addMenu('&View')
        add(m_view, 'Toggle file panel',
            lambda: self._set_file_panel_visible(not self._file_panel_visible),
            'F2')
        add(m_view, 'Toggle log dock', self._toggle_log_dock, 'F3')

        m_help = mb.addMenu('&Help')
        add(m_help, 'Keyboard shortcuts...', self._show_shortcuts_dialog, 'F1')
        add(m_help, 'About', self._show_about)

    def _build_log_dock(self):
        self._log_dock = QDockWidget('Log', self)
        self._log_dock.setObjectName('LogDock')
        self._log_dock.setAllowedAreas(Qt.BottomDockWidgetArea
                                        | Qt.RightDockWidgetArea)
        self._log_view = QPlainTextEdit()
        self._log_view.setObjectName('LogPanel')
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(2000)
        self._log_dock.setWidget(self._log_view)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._log_dock)
        self._log_dock.setVisible(False)
        self._log('Application started.')

    def _toggle_log_dock(self):
        if hasattr(self, '_log_dock'):
            self._log_dock.setVisible(not self._log_dock.isVisible())

    def _show_shortcuts_dialog(self):
        rows = [
            ('Space', 'Play / Pause'),
            ('Left / Right', 'Seek one frame'),
            ('Ctrl+Left / Ctrl+Right', 'Previous / Next item'),
            ('Ctrl+O', 'Open video'),
            ('Ctrl+R', 'Open reference CSV'),
            ('Ctrl+E', 'Extract current'),
            ('Ctrl+L', 'Auto-align'),
            ('Ctrl+S', 'Save project'),
            ('Ctrl+Shift+E', 'Export signal CSV'),
            ('Ctrl+Q', 'Quit'),
            ('F1', 'This shortcuts dialog'),
            ('F2', 'Toggle file panel'),
            ('F3', 'Toggle log dock'),
            ('Scroll wheel on video', 'Zoom'),
            ('Ctrl+drag / Middle drag', 'Pan'),
        ]
        text = '\n'.join(f'{k:28s}  {v}' for k, v in rows)
        dlg = QDialog(self)
        dlg.setWindowTitle('Keyboard shortcuts')
        layout = QVBoxLayout(dlg)
        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setMinimumWidth(440)
        view.setMinimumHeight(360)
        layout.addWidget(view)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        layout.addWidget(bb)
        dlg.exec_()

    def _show_about(self):
        QMessageBox.information(
            self, 'About',
            'Video Barcode Signal Extractor\n'
            'A tool for extracting LED / barcode synchronization signals from video\n'
            'and aligning them to external reference traces.\n\n'
            'Refactored into modules with unit tests.')

    # ── PyAV-accurate decoder option ─────────────────────────────────────────
    @staticmethod
    def _pyav_available() -> bool:
        try:
            import av  # noqa: F401
            return True
        except Exception:
            return False

    def _build_file_panel(self):
        panel = QFrame()
        panel.setObjectName('SidePanel')
        panel.setMinimumWidth(270)
        panel.setMaximumWidth(380)

        vl = QVBoxLayout(panel)
        vl.setContentsMargins(10, 10, 10, 10)
        vl.setSpacing(8)

        head = QHBoxLayout()
        title = QLabel('Files')
        title.setObjectName('PanelTitle')
        head.addWidget(title)
        head.addStretch()
        self._file_count_badge = QLabel('0 files')
        self._file_count_badge.setObjectName('BadgeLabel')
        self._file_count_badge.setAlignment(Qt.AlignCenter)
        head.addWidget(self._file_count_badge)
        close_btn = QPushButton('×')
        close_btn.setFixedWidth(30)
        close_btn.setToolTip('Hide file navigation panel')
        close_btn.clicked.connect(lambda: self._set_file_panel_visible(False))
        head.addWidget(close_btn)
        vl.addLayout(head)

        self._file_filter = QLineEdit()
        self._file_filter.setPlaceholderText('Filter videos or CSVs')
        self._file_filter.setClearButtonEnabled(True)
        vl.addWidget(self._file_filter)

        self._file_list = QListWidget()
        self._file_list.setObjectName('FileList')
        self._file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._file_list.setUniformItemSizes(False)
        vl.addWidget(self._file_list, 1)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)

        def _panel_btn(text, icon, tip, cb):
            b = QPushButton(text)
            b.setIcon(self.style().standardIcon(icon))
            b.setToolTip(tip)
            b.clicked.connect(cb)
            return b

        self._side_prev_btn = _panel_btn(
            'Previous', QStyle.SP_ArrowBack,
            'Previous video/reference pair', self._prev_item)
        self._side_next_btn = _panel_btn(
            'Next', QStyle.SP_ArrowForward,
            'Next video/reference pair', self._next_item)
        nav_row.addWidget(self._side_prev_btn)
        nav_row.addWidget(self._side_next_btn)
        vl.addLayout(nav_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        action_row.addWidget(_panel_btn(
            'Video', QStyle.SP_DialogOpenButton,
            'Open one video', self._open_video))
        action_row.addWidget(_panel_btn(
            'List', QStyle.SP_FileDialogListView,
            'Open a list of video files', self._open_video_list))
        action_row.addWidget(_panel_btn(
            'CSVs', QStyle.SP_FileIcon,
            'Attach a list of reference CSV/TXT files by row order',
            self._open_csv_list))
        vl.addLayout(action_row)

        project_row = QHBoxLayout()
        project_row.setSpacing(6)
        project_row.addWidget(_panel_btn(
            'Save', QStyle.SP_DialogSaveButton,
            'Save project file with file list, ROI, columns and alignment settings',
            self._save_project))
        project_row.addWidget(_panel_btn(
            'Load', QStyle.SP_DirOpenIcon,
            'Load saved project file', self._load_project))
        vl.addLayout(project_row)

        info = QFrame()
        info.setObjectName('InfoPanel')
        info_l = QVBoxLayout(info)
        info_l.setContentsMargins(10, 9, 10, 9)
        info_l.setSpacing(4)
        current_lbl = QLabel('Current Selection')
        current_lbl.setObjectName('MutedLabel')
        info_l.addWidget(current_lbl)
        self._file_video_lbl = QLabel('No video selected')
        self._file_video_lbl.setWordWrap(True)
        self._file_csv_lbl = QLabel('Reference: none')
        self._file_csv_lbl.setWordWrap(True)
        self._file_state_lbl = QLabel('Load a video, open a project, or drag files here.')
        self._file_state_lbl.setObjectName('MutedLabel')
        self._file_state_lbl.setWordWrap(True)
        info_l.addWidget(self._file_video_lbl)
        info_l.addWidget(self._file_csv_lbl)
        info_l.addWidget(self._file_state_lbl)
        vl.addWidget(info)

        return panel

    # ── build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── toolbar ──────────────────────────────────────────────────────────
        tb = QToolBar('Main', movable=False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        def _btn(text, tip, cb, enabled=True):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setEnabled(enabled)
            b.clicked.connect(cb)
            tb.addWidget(b)
            return b

        # Minimal toolbar: file panel toggle, open video, navigation,
        # extract (primary), auto-align, export. Everything else lives in
        # the menu bar (File / Extract / Align / Export / View / Help).
        self._btn_files = QPushButton('Files')
        self._btn_files.setCheckable(True)
        self._btn_files.setChecked(self._file_panel_visible)
        self._btn_files.setToolTip('Show or hide the file navigation panel  (F2)')
        self._btn_files.clicked.connect(self._set_file_panel_visible)
        tb.addWidget(self._btn_files)
        tb.addSeparator()

        _btn('Open video', 'Open a video file  (Ctrl+O)', self._open_video)
        tb.addSeparator()

        self._btn_prev = _btn('◀', 'Previous video/reference pair',
                              self._prev_item, enabled=False)
        self._btn_prev.setFixedWidth(34)
        self._file_index_lbl = QLabel('0/0')
        self._file_index_lbl.setMinimumWidth(52)
        self._file_index_lbl.setAlignment(Qt.AlignCenter)
        self._file_index_lbl.setObjectName('TimeCode')
        tb.addWidget(self._file_index_lbl)
        self._file_nav_cb = QComboBox()
        self._file_nav_cb.setMinimumWidth(220)
        self._file_nav_cb.setToolTip('Jump to a loaded video/reference pair')
        tb.addWidget(self._file_nav_cb)
        self._btn_next = _btn('▶', 'Next video/reference pair',
                              self._next_item, enabled=False)
        self._btn_next.setFixedWidth(34)
        tb.addSeparator()

        self._btn_extract = _btn('Extract', 'Extract ROI signal  (Ctrl+E)',
                                 self._extract_signal, enabled=False)
        self._btn_extract.setObjectName('PrimaryButton')
        self._btn_xcorr = _btn('Auto-align',
                               'Align traces by cross-correlation  (Ctrl+L)',
                               self._auto_align, enabled=False)
        tb.addSeparator()
        self._btn_export = _btn('Export CSV', 'Export signal CSV  (Ctrl+Shift+E)',
                                self._export_signal)
        # The following toolbar buttons were removed in the redesign because
        # they all live in the menu bar now. We retain attribute references
        # that existing code touches, but they are not added to the toolbar.
        self._btn_batch_export = QPushButton('Batch export')
        self._btn_batch_export.clicked.connect(self._batch_export)
        self._btn_clrcache = QPushButton('Cache')
        self._btn_clrcache.clicked.connect(self._clear_signal_cache)

        # ── status bar ───────────────────────────────────────────────────────
        self._sb = QStatusBar()
        self._sb.setSizeGripEnabled(False)
        self.setStatusBar(self._sb)
        self._sb_lbl = QLabel('Load a video, open a project, or drag files here.')
        self._sb.addWidget(self._sb_lbl, 1)

        # Extraction progress complex (hidden when idle, shown during a run).
        # Lives on the right of the status bar so it does not displace the
        # main status text.
        self._progress_widget = QWidget()
        ph = QHBoxLayout(self._progress_widget)
        ph.setContentsMargins(0, 0, 0, 0)
        ph.setSpacing(10)

        self._progress_frame_lbl = QLabel('')
        self._progress_frame_lbl.setObjectName('TimeCode')
        self._progress_frame_lbl.setMinimumWidth(180)
        ph.addWidget(self._progress_frame_lbl)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(280)
        self._progress.setMinimumHeight(18)
        self._progress.setTextVisible(True)
        ph.addWidget(self._progress)

        self._progress_eta_lbl = QLabel('')
        self._progress_eta_lbl.setObjectName('TimeCode')
        self._progress_eta_lbl.setMinimumWidth(170)
        ph.addWidget(self._progress_eta_lbl)

        self._abort_btn = QPushButton('Abort')
        self._abort_btn.setFixedWidth(72)
        self._abort_btn.setToolTip('Abort the running extraction')
        self._abort_btn.clicked.connect(self._abort_extraction)
        ph.addWidget(self._abort_btn)

        self._progress_widget.setVisible(False)
        self._sb.addPermanentWidget(self._progress_widget)

        # ── central splitter (file queue, video, analysis) ────────────────
        self._root_splitter = QSplitter(Qt.Horizontal)
        root = self._root_splitter
        self.setCentralWidget(root)
        self._file_panel = self._build_file_panel()
        root.addWidget(self._file_panel)

        # ══════════════════════════════════════════
        # LEFT — video panel
        # ══════════════════════════════════════════
        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.setSpacing(3)

        self._vid = VideoWidget()
        lv.addWidget(self._vid, 1)

        # zoom row
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(6)
        for text, tip, fn in [
            ('+',    'Zoom in  (scroll up)',   lambda: self._vid.zoom_by(1.25)),
            ('−',    'Zoom out (scroll down)', lambda: self._vid.zoom_by(0.8)),
            ('Fit',  'Fit to window',          self._vid.fit_to_widget),
            ('1:1',  '100% zoom',              self._zoom100),
        ]:
            b = QPushButton(text); b.setToolTip(tip)
            b.setFixedWidth(46);  b.clicked.connect(fn)
            zoom_row.addWidget(b)
        zoom_row.addStretch()
        self._zoom_lbl = QLabel('zoom')
        self._zoom_lbl.setObjectName('TimeCode')
        zoom_row.addWidget(self._zoom_lbl)
        lv.addLayout(zoom_row)

        # frame row
        fr_row = QHBoxLayout()
        fr_row.addWidget(QLabel('Frame:'))
        self._fspin = QSpinBox()
        self._fspin.setMinimum(0); self._fspin.setFixedWidth(80)
        fr_row.addWidget(self._fspin)
        fr_row.addWidget(QLabel('/'))
        self._ftotal_lbl = QLabel('0')
        fr_row.addWidget(self._ftotal_lbl)
        fr_row.addStretch()
        self._time_lbl = QLabel('00:00.000 / 00:00.000')
        self._time_lbl.setObjectName('TimeCode')
        self._time_lbl.setMinimumWidth(150)
        self._time_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        fr_row.addWidget(self._time_lbl)
        lv.addLayout(fr_row)

        self._fslider = QSlider(Qt.Horizontal)
        self._fslider.setMinimum(0)
        lv.addWidget(self._fslider)

        # playback row
        play_row = QHBoxLayout()
        play_row.setSpacing(4)
        for text, tip, fn in [
            ('|◀',   'First frame',  lambda: self._seek(0)),
            ('-10s', 'Jump back 10 seconds',
             lambda: self._seek(self._fslider.value() - int(round(self._fps * 10)))),
            ('-1s',  'Jump back 1 second',
             lambda: self._seek(self._fslider.value() - int(round(self._fps)))),
            ('◀',    'Step back',    lambda: self._seek(self._fslider.value() - 1)),
            ('▶',    'Step forward', lambda: self._seek(self._fslider.value() + 1)),
            ('+1s',  'Jump forward 1 second',
             lambda: self._seek(self._fslider.value() + int(round(self._fps)))),
            ('+10s', 'Jump forward 10 seconds',
             lambda: self._seek(self._fslider.value() + int(round(self._fps * 10)))),
            ('▶|',   'Last frame',   lambda: self._seek(self._n_frames - 1)),
        ]:
            b = QPushButton(text); b.setToolTip(tip)
            b.setFixedWidth(48 if 's' in text else 38);  b.clicked.connect(fn)
            play_row.addWidget(b)

        self._btn_play = QPushButton('Play')
        self._btn_play.setCheckable(True)
        self._btn_play.setFixedWidth(72)
        self._btn_play.setToolTip('Play / Pause  (Space)')
        self._btn_play.clicked.connect(self._toggle_play)
        play_row.addWidget(self._btn_play)

        self._speed_cb = QComboBox()
        for s in ['0.25×', '0.5×', '1×', '2×', '4×']:
            self._speed_cb.addItem(s)
        self._speed_cb.setCurrentIndex(2)
        self._speed_cb.setFixedWidth(62)
        play_row.addWidget(self._speed_cb)
        play_row.addStretch()
        lv.addLayout(play_row)

        # ROI group
        roi_grp    = QGroupBox('Region of Interest  (click-drag on video)')
        roi_layout = QFormLayout(roi_grp)
        roi_layout.setSpacing(3)
        self._roi_lbl = QLabel('Not set')
        roi_layout.addRow('ROI (x,y,w,h):', self._roi_lbl)

        ch_row = QHBoxLayout()
        self._ch_cb = QComboBox()
        for c in ['Grayscale', 'Red', 'Green', 'Blue']:
            self._ch_cb.addItem(c)
        ch_row.addWidget(QLabel('Channel:'))
        ch_row.addWidget(self._ch_cb)
        ch_row.addStretch()
        btn_clr = QPushButton('Clear ROI')
        btn_clr.setFixedWidth(80)
        btn_clr.clicked.connect(self._clear_roi)
        ch_row.addWidget(btn_clr)
        roi_layout.addRow('', ch_row)

        # extraction range
        rng_row = QHBoxLayout()
        rng_row.addWidget(QLabel('Start frame:'))
        self._start_spin = QSpinBox(); self._start_spin.setMinimum(0)
        self._start_spin.setFixedWidth(80)
        rng_row.addWidget(self._start_spin)
        rng_row.addWidget(QLabel('End frame:'))
        self._end_spin = QSpinBox(); self._end_spin.setMinimum(1)
        self._end_spin.setFixedWidth(80)
        rng_row.addWidget(self._end_spin)
        rng_row.addStretch()
        roi_layout.addRow('Extract range:', rng_row)

        # workers row
        wrk_row = QHBoxLayout()
        wrk_row.addWidget(QLabel('Workers:'))
        self._workers_spin = QSpinBox()
        self._workers_spin.setMinimum(1)
        self._workers_spin.setMaximum(os.cpu_count() or 8)
        self._workers_spin.setValue(max(1, (os.cpu_count() or 2) // 2))
        self._workers_spin.setFixedWidth(48)
        self._workers_spin.setToolTip(
            f'Parallel processes for extraction\n'
            f'(CPU cores detected: {os.cpu_count()})')
        wrk_row.addWidget(self._workers_spin)
        self._workers_info = QLabel('')
        self._workers_info.setStyleSheet(f'color:{CYAN}; font-family:Consolas; font-size:11px;')
        wrk_row.addWidget(self._workers_info)
        wrk_row.addSpacing(12)
        wrk_row.addWidget(QLabel('Decoder:'))
        self._decoder_cb = QComboBox()
        self._decoder_cb.addItem('OpenCV (fast)')
        pyav_ok = self._pyav_available()
        self._decoder_cb.addItem('PyAV (accurate)'
                                  + ('' if pyav_ok else ' [install av]'))
        item_idx = self._decoder_cb.model().index(1, 0)
        if not pyav_ok:
            self._decoder_cb.model().itemFromIndex(item_idx).setEnabled(False)
        self._decoder_cb.setToolTip(
            'OpenCV: fast, uses keyframe seek + lead-in. Adequate for most videos.\n'
            'PyAV: frame-accurate seek via container PTS. Requires `pip install av`.')
        self._decoder_cb.setFixedWidth(170)
        wrk_row.addWidget(self._decoder_cb)
        wrk_row.addStretch()
        roi_layout.addRow('Parallel jobs:', wrk_row)

        self._roi_val_lbl = QLabel('—')
        roi_layout.addRow('Current ROI mean:', self._roi_val_lbl)

        lv.addWidget(roi_grp)
        root.addWidget(left)

        # ══════════════════════════════════════════
        # RIGHT — tabbed analysis panel
        # ══════════════════════════════════════════
        self._right_tabs = QTabWidget()
        self._right_tabs.setDocumentMode(True)
        # Use the global stylesheet's QTabBar styling (Nord underline).

        # ─────────────────────────────────────────
        # TAB 1 — Signal
        # ─────────────────────────────────────────
        sig_w  = QWidget()
        sig_vl = QVBoxLayout(sig_w)
        sig_vl.setContentsMargins(4, 4, 4, 4)
        sig_vl.setSpacing(2)

        self._trace_pw = pg.PlotWidget()
        self._theme_plot(self._trace_pw)
        self._trace_pw.setLabel('bottom', 'Time', units='s')
        self._trace_pw.setLabel('left', 'ROI mean intensity')
        self._trace_pw.showGrid(x=True, y=True, alpha=0.18)
        self._trace_pw.setDownsampling(auto=True, mode='peak')
        self._trace_pw.setClipToView(True)
        self._trace_pw.addLegend(offset=(10, 10))

        self._raw_curve  = self._trace_pw.plot(
            pen=pg.mkPen(BLUE, width=1.5), name='Raw signal')
        self._bin_curve  = self._trace_pw.plot(
            pen=pg.mkPen(GREEN, width=1.5), name='Binary (scaled)')

        self._thresh_line = pg.InfiniteLine(
            pos=128, angle=0, movable=True,
            pen=pg.mkPen(RED, width=1.5, style=Qt.DashLine),
            label='Thr {value:.1f}',
            labelOpts={'color': RED, 'position': 0.05})
        self._trace_pw.addItem(self._thresh_line)

        self._region = pg.LinearRegionItem(
            values=[0, 10],
            brush=pg.mkBrush(color=(137, 180, 250, 25)),
            pen=pg.mkPen(BLUE, width=0.8))
        self._trace_pw.addItem(self._region)

        self._cursor_v = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(PEACH, width=1.2, style=Qt.DotLine))
        self._trace_pw.addItem(self._cursor_v)

        sig_vl.addWidget(self._trace_pw, 1)

        # threshold controls – two rows inside a vertical layout
        th_grp  = QGroupBox('Threshold')
        th_vl   = QVBoxLayout(th_grp)
        th_vl.setSpacing(3)

        # row 1: mode radios + output combo + smooth + invert
        thl1 = QHBoxLayout()
        thl1.setSpacing(6)

        self._rad_auto   = QRadioButton('Auto')
        self._rad_manual = QRadioButton('Manual')
        self._rad_auto.setChecked(True)
        self._thr_mode_grp = QButtonGroup()
        self._thr_mode_grp.addButton(self._rad_auto,   0)
        self._thr_mode_grp.addButton(self._rad_manual, 1)
        thl1.addWidget(self._rad_auto)
        thl1.addWidget(self._rad_manual)
        thl1.addSpacing(8)

        thl1.addWidget(QLabel('Method:'))
        self._thr_method_cb = QComboBox()
        self._thr_method_cb.addItems(
            ['Auto (recommended)', 'Otsu', 'Triangle', 'Kapur', 'MAD', 'Percentile'])
        self._thr_method_cb.setToolTip(
            'Auto: pick Otsu or Triangle based on class balance.\n'
            'Otsu: classic between-class variance maximization (balanced data).\n'
            'Triangle: robust to skewed histograms (sparse pulses).\n'
            'Kapur: maximum-entropy threshold (principled for sparse).\n'
            'MAD: median + k * MAD of the off-state (robust statistical fence).\n'
            'Percentile: 95th percentile (tunable in Pass B).')
        self._thr_method_cb.setFixedWidth(150)
        thl1.addWidget(self._thr_method_cb)
        thl1.addSpacing(12)

        thl1.addWidget(QLabel('Output:'))
        self._out_mode_cb = QComboBox()
        self._out_mode_cb.addItems(['Binary (0/1)', 'Gated peaks'])
        self._out_mode_cb.setToolTip(
            'Binary: hard 0/1 per sample\n'
            'Gated peaks: keep raw amplitude where signal ≥ threshold')
        self._out_mode_cb.setFixedWidth(130)
        thl1.addWidget(self._out_mode_cb)
        thl1.addSpacing(12)

        thl1.addWidget(QLabel('Smooth σ:'))
        self._smooth_spin = QSpinBox()
        self._smooth_spin.setRange(0, 100)
        self._smooth_spin.setValue(0)
        self._smooth_spin.setToolTip('Gaussian σ (frames)')
        self._smooth_spin.setFixedWidth(52)
        thl1.addWidget(self._smooth_spin)

        self._invert_cb = QCheckBox('Invert')
        thl1.addWidget(self._invert_cb)
        thl1.addStretch()
        th_vl.addLayout(thl1)

        # row 2: threshold slider (active only in Manual mode)
        thl2 = QHBoxLayout()
        thl2.setSpacing(6)
        thl2.addWidget(QLabel('Threshold:'))
        self._th_slider = QSlider(Qt.Horizontal)
        self._th_slider.setRange(0, 255)
        self._th_slider.setValue(128)
        self._th_slider.setEnabled(False)   # starts in Auto mode
        thl2.addWidget(self._th_slider, 1)
        self._th_val_lbl = QLabel('128.0')
        self._th_val_lbl.setFixedWidth(50)
        thl2.addWidget(self._th_val_lbl)
        th_vl.addLayout(thl2)

        sig_vl.addWidget(th_grp)

        # ── quality readout ──────────────────────────────────────────────
        q_grp = QGroupBox('Signal quality')
        q_grp.setToolTip(
            'SNR: peak-to-peak amplitude over off-state noise floor (MAD).\n'
            'Edges: number of state transitions in the binarized signal.\n'
            'Edge-interval CV: std/mean of inter-edge intervals. Lower is cleaner.\n'
            'Saturation: fraction of samples pinned at the min or max value.')
        q_form = QFormLayout(q_grp)
        q_form.setSpacing(2)
        self._q_snr_lbl  = QLabel('—')
        self._q_snr_lbl.setObjectName('TimeCode')
        self._q_edges_lbl = QLabel('—')
        self._q_edges_lbl.setObjectName('TimeCode')
        self._q_cv_lbl   = QLabel('—')
        self._q_cv_lbl.setObjectName('TimeCode')
        self._q_sat_lbl  = QLabel('—')
        self._q_sat_lbl.setObjectName('TimeCode')
        q_form.addRow('SNR (dB):', self._q_snr_lbl)
        q_form.addRow('Edges:', self._q_edges_lbl)
        q_form.addRow('Edge interval CV:', self._q_cv_lbl)
        q_form.addRow('Saturated frac:', self._q_sat_lbl)
        sig_vl.addWidget(q_grp)

        self._right_tabs.addTab(sig_w, 'Signal')

        # ─────────────────────────────────────────
        # TAB 2 — Alignment
        # ─────────────────────────────────────────
        align_tab = QWidget()
        align_vl  = QVBoxLayout(align_tab)
        align_vl.setContentsMargins(4, 4, 4, 4)
        align_vl.setSpacing(3)

        align_split = QSplitter(Qt.Vertical)

        # ── overlay plot (extracted + CSV, both normalised) ───────────────
        self._cmp_pw = pg.PlotWidget()
        self._theme_plot(self._cmp_pw)
        self._cmp_pw.setLabel('bottom', 'Time', units='s')
        self._cmp_pw.setLabel('left', 'Stacked normalised amplitude')
        self._cmp_pw.showGrid(x=True, y=True, alpha=0.18)
        self._cmp_pw.setDownsampling(auto=True, mode='peak')
        self._cmp_pw.setClipToView(True)
        self._cmp_pw.addLegend(offset=(10, 10))
        self._cmp_pw.setXLink(self._trace_pw)

        self._ext_curve = self._cmp_pw.plot(
            pen=pg.mkPen(CYAN,   width=2.0), name='Extracted signal')
        self._csv_curve = self._cmp_pw.plot(
            pen=pg.mkPen(PEACH, width=2.0), name='CSV reference')

        self._cursor_v2 = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(PEACH, width=1.2, style=Qt.DotLine))
        self._cmp_pw.addItem(self._cursor_v2)

        self._cmp_sep_line = pg.InfiniteLine(
            pos=1.10, angle=0, movable=False,
            pen=pg.mkPen(OVERLAY, width=1.0, style=Qt.DashLine))
        self._cmp_pw.addItem(self._cmp_sep_line)
        self._cmp_pw.setYRange(-0.10, 2.35, padding=0)

        # Difference fill is hidden in stacked mode; overlap fill obscures timing.
        self._diff_curve = pg.FillBetweenItem(
            self._ext_curve, self._csv_curve,
            brush=pg.mkBrush(color=(205, 214, 244, 0)))
        self._cmp_pw.addItem(self._diff_curve)
        self._diff_curve.setVisible(False)

        align_split.addWidget(self._cmp_pw)

        # ── cross-correlation plot ─────────────────────────────────────────
        self._xcorr_pw = pg.PlotWidget()
        self._theme_plot(self._xcorr_pw)
        self._xcorr_pw.setLabel('bottom', 'Lag', units='s')
        self._xcorr_pw.setLabel('left', 'Cross-correlation')
        self._xcorr_pw.showGrid(x=True, y=True, alpha=0.18)
        self._xcorr_pw.setDownsampling(auto=True, mode='peak')
        self._xcorr_pw.setClipToView(True)
        self._xcorr_pw.addLegend(offset=(10, 10))

        self._xcorr_curve = self._xcorr_pw.plot(
            pen=pg.mkPen(MAUVE, width=1.5), name='Cross-correlation')
        self._align_fit_curve = self._xcorr_pw.plot(
            pen=pg.mkPen(GREEN, width=1.4), name='Fit / warp')
        self._xcorr_lag_line = pg.InfiniteLine(
            pos=0, angle=90, movable=False,
            pen=pg.mkPen(RED, width=1.5, style=Qt.DashLine),
            label='lag {value:.3f} s',
            labelOpts={'color': RED, 'position': 0.85})
        self._xcorr_pw.addItem(self._xcorr_lag_line)

        align_split.addWidget(self._xcorr_pw)
        align_split.setSizes([400, 200])
        align_vl.addWidget(align_split, 1)

        # ── alignment controls ─────────────────────────────────────────────
        al_grp = QGroupBox('Alignment & Column Selection')
        al_vl  = QVBoxLayout(al_grp)
        al_vl.setSpacing(3)

        # row 1: time column + display value column
        all_1 = QHBoxLayout(); all_1.setSpacing(6)
        all_1.addWidget(QLabel('Time col:'))
        self._csv_t_cb = QComboBox(); self._csv_t_cb.setMinimumWidth(120)
        self._csv_t_cb.setToolTip('Column interpreted as time (x-axis)')
        all_1.addWidget(self._csv_t_cb)
        all_1.addWidget(QLabel('Display col:'))
        self._csv_v_cb = QComboBox(); self._csv_v_cb.setMinimumWidth(120)
        self._csv_v_cb.setToolTip('Column plotted as the reference trace')
        all_1.addWidget(self._csv_v_cb)
        all_1.addWidget(QLabel('Method:'))
        self._align_method_cb = QComboBox()
        self._align_method_cb.addItems([
            'Cross-correlation',
            'Linear regression',
            'Edge interpolation',
            'DTW (nonlinear)',
        ])
        self._align_method_cb.setToolTip(
            'Cross-correlation: one global offset\n'
            'Linear regression: edge-paired affine time map (RANSAC)\n'
            'Edge interpolation: piecewise time map through matched edges\n'
            'DTW: nonlinear time warp on full waveforms; handles drift,\n'
            '     dropped frames, and skew without requiring landmark edges.')
        self._align_method_cb.setMinimumWidth(150)
        all_1.addWidget(self._align_method_cb)
        all_1.addStretch()
        al_vl.addLayout(all_1)

        # row 2: alignment column + offset + stats
        all_2 = QHBoxLayout(); all_2.setSpacing(6)
        all_2.addWidget(QLabel('Align col:'))
        self._csv_align_cb = QComboBox(); self._csv_align_cb.setMinimumWidth(120)
        self._csv_align_cb.setToolTip(
            'Column used for cross-correlation alignment\n'
            '(can differ from the display column)')
        all_2.addWidget(self._csv_align_cb)
        all_2.addWidget(QLabel('Offset (s):'))
        self._offset_spin = QDoubleSpinBox()
        self._offset_spin.setRange(-99999, 99999)
        self._offset_spin.setSingleStep(0.01)
        self._offset_spin.setDecimals(4)
        self._offset_spin.setFixedWidth(90)
        all_2.addWidget(self._offset_spin)
        all_2.addWidget(QLabel('Max lag:'))
        self._max_lag_spin = QDoubleSpinBox()
        self._max_lag_spin.setRange(0.05, 3600.0)
        self._max_lag_spin.setSingleStep(0.1)
        self._max_lag_spin.setDecimals(2)
        self._max_lag_spin.setValue(2.5)
        self._max_lag_spin.setFixedWidth(72)
        self._max_lag_spin.setToolTip(
            'Cross-correlation search limit around zero lag.\n'
            'Keep this small for repetitive barcode trains; increase only if the true offset is large.')
        all_2.addWidget(self._max_lag_spin)
        all_2.addSpacing(16)
        self._align_stats_lbl = QLabel('')
        self._align_stats_lbl.setObjectName('TimeCode')
        self._align_stats_lbl.setWordWrap(True)
        self._align_stats_lbl.setStyleSheet(f'color:{CYAN};')
        all_2.addWidget(self._align_stats_lbl, 1)
        al_vl.addLayout(all_2)

        # Dedicated second line for edge-pair statistics so the matched-points
        # count is always visible and not clipped by the right edge.
        self._align_edge_lbl = QLabel('')
        self._align_edge_lbl.setObjectName('TimeCode')
        self._align_edge_lbl.setWordWrap(True)
        self._align_edge_lbl.setStyleSheet(f'color:{FG_MUTED};')
        al_vl.addWidget(self._align_edge_lbl)

        align_vl.addWidget(al_grp)

        self._right_tabs.addTab(align_tab, 'Alignment')

        root.addWidget(self._right_tabs)
        root.setSizes([max(240, self._file_panel_width), 520, 980])
        self._set_file_panel_visible(self._file_panel_visible, persist=False)

        # playback timer
        self._play_timer = QTimer()
        self._play_timer.timeout.connect(self._play_step)

    # ── connections ───────────────────────────────────────────────────────────
    def _connect_signals(self):
        self._fslider.valueChanged.connect(self._on_slider)
        self._fspin.valueChanged.connect(self._on_spin)
        self._vid.roiChanged.connect(self._on_roi_changed)
        self._vid.zoomChanged.connect(self._on_zoom_changed)

        # threshold / view
        self._th_slider.valueChanged.connect(self._on_th_slider)
        self._thresh_line.sigPositionChanged.connect(self._on_thresh_drag)
        self._smooth_spin.valueChanged.connect(self._refresh_view)
        self._invert_cb.stateChanged.connect(self._refresh_view)
        self._out_mode_cb.currentIndexChanged.connect(self._refresh_view)
        self._thr_mode_grp.buttonClicked.connect(self._on_thr_mode_changed)
        self._thr_method_cb.currentIndexChanged.connect(self._on_thr_method_changed)
        self._ch_cb.currentIndexChanged.connect(self._on_extraction_config_changed)
        self._start_spin.valueChanged.connect(self._on_extraction_config_changed)
        self._end_spin.valueChanged.connect(self._on_extraction_config_changed)

        # comparison
        self._offset_spin.valueChanged.connect(self._refresh_cmp)
        self._offset_spin.valueChanged.connect(lambda *_: self._persist_current_cache_metadata())
        self._max_lag_spin.valueChanged.connect(self._redraw_alignment_diagnostic)
        self._max_lag_spin.valueChanged.connect(lambda *_: self._persist_current_cache_metadata())
        self._csv_t_cb.currentIndexChanged.connect(self._refresh_cmp)
        self._csv_v_cb.currentIndexChanged.connect(self._refresh_cmp)
        self._csv_align_cb.currentIndexChanged.connect(self._refresh_cmp)
        self._align_method_cb.currentIndexChanged.connect(self._on_alignment_method_changed)
        self._file_nav_cb.currentIndexChanged.connect(self._on_file_nav_changed)
        self._file_list.currentRowChanged.connect(self._on_file_list_row_changed)
        self._file_filter.textChanged.connect(self._apply_file_filter)

        # click on trace plots → seek video
        self._trace_pw.scene().sigMouseClicked.connect(self._on_trace_click)
        self._cmp_pw.scene().sigMouseClicked.connect(self._on_cmp_click)
        self._xcorr_pw.scene().sigMouseClicked.connect(self._on_xcorr_click)

        # viewport changes → re-slice data fed to plots
        self._trace_pw.getViewBox().sigXRangeChanged.connect(
            lambda *_: self._refresh_view())

    # ── signal cache helpers (thin shims over vbe.core.cache) ─────────────────
    def _signal_cache_key(self, roi, channel, start_frame, end_frame):
        return signal_cache_key(self._video_path, roi, channel, start_frame, end_frame)

    def _signal_cache_meta_path(self, key: str) -> Path:
        return signal_cache_meta_path(self._video_path, key)

    def _current_extraction_metadata(self, key: str | None = None):
        roi = self._vid.get_roi() if hasattr(self, '_vid') else None
        ch_map = {0: 'gray', 1: 'r', 2: 'g', 3: 'b'}
        video_path = self._video_path
        stat = None
        if video_path:
            try:
                p = Path(video_path)
                stat = {'size': p.stat().st_size, 'mtime': p.stat().st_mtime}
            except Exception:
                stat = None
        return {
            'version': 2,
            'created': datetime.now().isoformat(timespec='seconds'),
            'cache_key': key,
            'video_path': video_path,
            'video_name': Path(video_path).name if video_path else None,
            'video_stat': stat,
            'fps': self._fps,
            'n_frames': self._n_frames,
            'roi': roi,
            'channel_index': self._ch_cb.currentIndex(),
            'channel': ch_map.get(self._ch_cb.currentIndex(), 'gray'),
            'start_frame': self._start_spin.value(),
            'end_frame': min(self._end_spin.value(), self._n_frames),
            'csv_path': self._csv_path,
            'offset': self._offset_spin.value(),
            'time_col': self._csv_t_cb.currentText(),
            'display_col': self._csv_v_cb.currentText(),
            'align_col': self._csv_align_cb.currentText(),
            'align_method': self._align_method_cb.currentText(),
            'align_model': self._align_model,
            'max_lag_s': self._max_lag_spin.value(),
            'threshold': float(self._thresh_line.value()),
            'threshold_manual': self._rad_manual.isChecked(),
            'smooth_sigma': self._smooth_spin.value(),
            'invert': self._invert_cb.isChecked(),
            'output_mode': self._out_mode_cb.currentIndex(),
        }

    def _load_from_cache(self, key: str):
        return load_signal_cache(self._video_path, key)

    def _load_cache_metadata(self, key: str):
        return load_cache_metadata(self._video_path, key)

    def _save_to_cache(self, key: str, times: np.ndarray, values: np.ndarray,
                      metadata: dict | None = None):
        save_signal_cache(self._video_path, key, times, values,
                          metadata or self._current_extraction_metadata(key))

    def _apply_cache_metadata_to_item(self, key: str):
        item = self._current_project_item()
        if item is None:
            return
        meta = self._load_cache_metadata(key) or self._current_extraction_metadata(key)
        item.update({
            'roi': meta.get('roi'),
            'start_frame': meta.get('start_frame', self._start_spin.value()),
            'end_frame': meta.get('end_frame', self._end_spin.value()),
            'channel_index': meta.get('channel_index', self._ch_cb.currentIndex()),
            'csv': meta.get('csv_path') or item.get('csv'),
            'offset': meta.get('offset', self._offset_spin.value()),
            'time_col': meta.get('time_col', self._csv_t_cb.currentText()),
            'display_col': meta.get('display_col', self._csv_v_cb.currentText()),
            'align_col': meta.get('align_col', self._csv_align_cb.currentText()),
            'align_method': meta.get('align_method', self._align_method_cb.currentText()),
            'align_model': self._sanitize_align_model(meta.get('align_model', self._align_model)),
            'max_lag_s': meta.get('max_lag_s', self._max_lag_spin.value()),
            'threshold': meta.get('threshold', float(self._thresh_line.value())),
            'threshold_manual': meta.get('threshold_manual', self._rad_manual.isChecked()),
            'smooth_sigma': meta.get('smooth_sigma', self._smooth_spin.value()),
            'invert': meta.get('invert', self._invert_cb.isChecked()),
            'output_mode': meta.get('output_mode', self._out_mode_cb.currentIndex()),
            'cache_key': key,
        })
        method = meta.get('align_method')
        if method:
            i = self._align_method_cb.findText(method)
            if i >= 0:
                self._align_method_cb.setCurrentIndex(i)
        self._align_model = self._sanitize_align_model(meta.get('align_model', self._align_model))

    def _current_signal_cache_key(self):
        roi = self._vid.get_roi() if hasattr(self, '_vid') else None
        if roi is None or not self._video_path:
            return None
        ch_map = {0: 'gray', 1: 'r', 2: 'g', 3: 'b'}
        return self._signal_cache_key(
            roi, ch_map[self._ch_cb.currentIndex()],
            self._start_spin.value(), min(self._end_spin.value(), self._n_frames))

    def _try_load_current_signal_from_cache(self):
        key = self._current_signal_cache_key()
        if not key or key == self._last_loaded_signal_key:
            return False
        cached = self._load_from_cache(key)
        if cached is None:
            return False
        times, values = cached
        self._sig_cache_key = key
        self._loading_from_cache = True
        self._workers_info.setText('(cached)')
        self._on_extracted(times, values)
        self._loading_from_cache = False
        self._last_loaded_signal_key = key
        self._apply_cache_metadata_to_item(key)
        self._update_nav_state()
        self._set_status(f'Loaded cached extraction for {Path(self._video_path).name}  |  key {key}')
        return True

    def _clear_signal_cache(self):
        if not self._video_path:
            return
        n = clear_signal_cache(self._video_path)
        cache_dir = Path(self._video_path).parent / '.sig_cache'
        if n > 0:
            self._set_status(f'Cleared {n} cached signal file(s) from {cache_dir}')
        else:
            self._set_status('No cache folder found.')

    # ── project/list helpers ─────────────────────────────────────────────────
    def _current_project_item(self):
        if 0 <= self._current_item < len(self._items):
            return self._items[self._current_item]
        return None

    def _save_current_item_state(self):
        item = self._current_project_item()
        if not item:
            return
        roi = self._vid.get_roi()
        item.update({
            'video': self._video_path,
            'csv': self._csv_path,
            'roi': roi,
            'start_frame': self._start_spin.value(),
            'end_frame': self._end_spin.value(),
            'channel_index': self._ch_cb.currentIndex(),
            'offset': self._offset_spin.value(),
            'time_col': self._csv_t_cb.currentText(),
            'display_col': self._csv_v_cb.currentText(),
            'align_col': self._csv_align_cb.currentText(),
            'align_method': self._align_method_cb.currentText(),
            'align_model': self._align_model,
            'max_lag_s': self._max_lag_spin.value(),
            'threshold': float(self._thresh_line.value()),
            'threshold_manual': self._rad_manual.isChecked(),
            'smooth_sigma': self._smooth_spin.value(),
            'invert': self._invert_cb.isChecked(),
            'output_mode': self._out_mode_cb.currentIndex(),
        })
        if roi:
            self._default_roi = roi
        self._persist_current_cache_metadata()

    def _persist_current_cache_metadata(self):
        key = getattr(self, '_sig_cache_key', None)
        if not key or self._sig_t is None or self._sig_v is None or not self._video_path:
            return
        meta = self._current_extraction_metadata(key)
        item = self._current_project_item()
        if item is not None:
            meta['project_item'] = item
        write_cache_metadata(self._video_path, key, meta)

    def _file_item_text(self, item: dict, index: int):
        video_path = item.get('video') or ''
        csv_path = item.get('csv') or ''
        video_name = Path(video_path).name if video_path else '(no video)'
        csv_name = Path(csv_path).name if csv_path else 'No reference CSV'
        roi_state = 'ROI set' if item.get('roi') else 'ROI missing'
        signal_state = 'signal loaded' if index == self._current_item and self._sig_t is not None else 'not loaded'
        text = f'{index + 1:02d}  {video_name}\n     CSV: {csv_name}  |  {roi_state}  |  {signal_state}'
        search = ' '.join([video_name, video_path, csv_name, csv_path, roi_state, signal_state]).lower()
        tooltip = f'Video: {video_path or "None"}\nReference: {csv_path or "None"}\n{roi_state}'
        return text, search, tooltip

    def _update_current_file_details(self):
        item = self._current_project_item()
        if not hasattr(self, '_file_video_lbl'):
            return
        if not item:
            self._file_video_lbl.setText('No video selected')
            self._file_csv_lbl.setText('Reference: none')
            self._file_state_lbl.setText('Load a video, open a project, or drag files here.')
            return

        video_path = item.get('video') or self._video_path or ''
        csv_path = item.get('csv') or self._csv_path or ''
        video_name = Path(video_path).name if video_path else '(no video)'
        csv_name = Path(csv_path).name if csv_path else 'none'
        self._file_video_lbl.setText(video_name)
        self._file_video_lbl.setToolTip(video_path)
        self._file_csv_lbl.setText(f'Reference: {csv_name}')
        self._file_csv_lbl.setToolTip(csv_path)
        parts = [
            'ROI set' if item.get('roi') else 'ROI missing',
            'signal loaded' if self._sig_t is not None else 'signal not loaded',
        ]
        if self._csv_df is not None:
            parts.append(f'{len(self._csv_df)} ref rows')
        if self._n_frames:
            parts.append(f'{self._n_frames} frames')
        self._file_state_lbl.setText(' | '.join(parts))

    def _apply_file_filter(self):
        if not hasattr(self, '_file_list'):
            return
        needle = self._file_filter.text().strip().lower() if hasattr(self, '_file_filter') else ''
        visible = 0
        for row in range(self._file_list.count()):
            item = self._file_list.item(row)
            haystack = item.data(Qt.UserRole + 1) or item.text().lower()
            hidden = bool(needle and needle not in haystack)
            item.setHidden(hidden)
            if not hidden:
                visible += 1
        if hasattr(self, '_file_count_badge'):
            n = len(self._items)
            if needle:
                self._file_count_badge.setText(f'{visible}/{n}')
            else:
                self._file_count_badge.setText(f'{n} file' + ('' if n == 1 else 's'))

    def _set_file_panel_visible(self, visible: bool, persist: bool = True):
        visible = bool(visible)
        if hasattr(self, '_root_splitter'):
            sizes = self._root_splitter.sizes()
            if sizes and sizes[0] > 0:
                self._file_panel_width = sizes[0]
        self._file_panel_visible = visible

        if hasattr(self, '_file_panel'):
            self._file_panel.setVisible(visible)
        if visible and hasattr(self, '_root_splitter'):
            sizes = self._root_splitter.sizes()
            middle = sizes[1] if len(sizes) > 1 and sizes[1] > 0 else 520
            right = sizes[2] if len(sizes) > 2 and sizes[2] > 0 else 980
            self._root_splitter.setSizes([max(240, self._file_panel_width), middle, right])
        if hasattr(self, '_btn_files'):
            self._btn_files.blockSignals(True)
            self._btn_files.setChecked(visible)
            self._btn_files.setText('Files' if visible else 'Show files')
            self._btn_files.blockSignals(False)
        if persist:
            self._settings.setValue('ui/file_panel_visible', visible)
            self._settings.setValue('ui/file_panel_width', self._file_panel_width)
            self._settings.sync()
            if hasattr(self, '_sb_lbl'):
                self._set_status('File navigation panel shown.' if visible else 'File navigation panel hidden.')

    def _update_nav_state(self):
        n = len(self._items)
        self._file_index_lbl.setText(f'{self._current_item + 1}/{n}' if n else '0/0')
        self._btn_prev.setEnabled(n > 1 and self._current_item > 0)
        self._btn_next.setEnabled(n > 1 and self._current_item < n - 1)
        labels = []
        for i, item in enumerate(self._items):
            video = Path(item.get('video') or '').name or '(no video)'
            csv = Path(item.get('csv') or '').name
            labels.append(f'{i + 1}. {video}' + (f'  |  {csv}' if csv else ''))
        self._file_nav_cb.blockSignals(True)
        if self._file_nav_cb.count() != len(labels) or [
            self._file_nav_cb.itemText(i) for i in range(self._file_nav_cb.count())
        ] != labels:
            self._file_nav_cb.clear()
            self._file_nav_cb.addItems(labels)
        self._file_nav_cb.setCurrentIndex(self._current_item if n else -1)
        self._file_nav_cb.setEnabled(n > 1)
        self._file_nav_cb.blockSignals(False)
        if hasattr(self, '_side_prev_btn'):
            self._side_prev_btn.setEnabled(n > 1 and self._current_item > 0)
            self._side_next_btn.setEnabled(n > 1 and self._current_item < n - 1)
        if hasattr(self, '_file_list'):
            self._file_list.blockSignals(True)
            self._file_list.clear()
            for i, item in enumerate(self._items):
                text, search, tooltip = self._file_item_text(item, i)
                lw_item = QListWidgetItem(text)
                lw_item.setData(Qt.UserRole, i)
                lw_item.setData(Qt.UserRole + 1, search)
                lw_item.setToolTip(tooltip)
                lw_item.setSizeHint(QSize(230, 58))
                self._file_list.addItem(lw_item)
            self._file_list.setEnabled(n > 0)
            self._file_list.setCurrentRow(self._current_item if n else -1)
            self._file_list.blockSignals(False)
            self._apply_file_filter()
        self._update_current_file_details()

    def _on_file_nav_changed(self, index: int):
        if self._loading_item or index < 0 or index == self._current_item:
            return
        self._load_item(index)

    def _on_file_list_row_changed(self, row: int):
        if self._loading_item or row < 0 or row == self._current_item:
            return
        self._load_item(row)

    def _load_item(self, index: int):
        if not (0 <= index < len(self._items)):
            return
        if index == self._current_item and self._video_path == self._items[index].get('video'):
            self._try_load_current_signal_from_cache()
            self._update_nav_state()
            return
        self._save_current_item_state()
        self._loading_item = True
        try:
            self._current_item = index
            item = self._items[index]
            self._load_video_path(item['video'], apply_item_state=True)
            if item.get('csv'):
                self._load_csv_path(item['csv'])
            else:
                self._csv_df = None
                self._csv_path = None
                for cb in (self._csv_t_cb, self._csv_v_cb, self._csv_align_cb):
                    cb.clear()
            self._try_load_current_signal_from_cache()
        finally:
            self._loading_item = False
            self._update_nav_state()

    def _prev_item(self):
        self._load_item(self._current_item - 1)

    def _next_item(self):
        self._load_item(self._current_item + 1)

    def _open_video_list(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Open video list', self._last_video_dir,
            'Video (*.mp4 *.avi *.mkv *.mov *.m4v *.wmv);;All (*)')
        if not paths:
            return
        self._last_video_dir = str(Path(paths[0]).parent)
        self._last_dir = self._last_video_dir
        self._items = [{'video': p, 'csv': None, 'roi': self._default_roi} for p in paths]
        self._current_item = -1
        self._load_item(0)
        self._save_app_settings()

    def _open_csv_list(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Open reference CSV list', self._last_csv_dir,
            'CSV / TXT (*.csv *.txt *.tsv);;All (*)')
        if not paths:
            return
        self._last_csv_dir = str(Path(paths[0]).parent)
        self._last_dir = self._last_csv_dir
        if not self._items:
            self._items = [{'video': self._video_path, 'csv': None, 'roi': self._default_roi}]
            self._current_item = 0
        for i, p in enumerate(paths):
            if i < len(self._items):
                self._items[i]['csv'] = p
        item = self._current_project_item()
        if item and item.get('csv'):
            self._load_csv_path(item['csv'])
        self._update_nav_state()
        self._save_app_settings()

    def _save_project(self):
        self._save_current_item_state()
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save project', self._last_project_dir, 'Barcode project (*.vbep.json);;JSON (*.json)')
        if not path:
            return
        data = {
            'version': 1,
            'current_item': self._current_item,
            'last_dir': self._last_dir,
            'folders': {
                'video': self._last_video_dir,
                'csv': self._last_csv_dir,
                'project': self._last_project_dir,
                'export': self._last_export_dir,
            },
            'default_roi': self._default_roi,
            'items': self._items,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')
        self._last_project_dir = str(Path(path).parent)
        self._last_dir = self._last_project_dir
        self._save_app_settings()
        self._set_status(f'Project saved → {path}')

    def _load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load project', self._last_project_dir,
            'Barcode project (*.vbep.json *.json);;All (*)')
        if not path:
            return
        self._load_project_path(path)

    # ── video loading ─────────────────────────────────────────────────────────
    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Video', self._last_video_dir,
            'Video (*.mp4 *.avi *.mkv *.mov *.m4v *.wmv);;All (*)')
        if not path:
            return
        self._last_video_dir = str(Path(path).parent)
        self._last_dir = self._last_video_dir
        self._items = [{'video': path, 'csv': None, 'roi': self._default_roi}]
        self._current_item = 0
        self._loading_item = True
        try:
            self._load_video_path(path, apply_item_state=True)
        finally:
            self._loading_item = False
        self._csv_df = None
        self._csv_path = None
        for cb in (self._csv_t_cb, self._csv_v_cb, self._csv_align_cb):
            cb.clear()
        self._btn_xcorr.setEnabled(False)
        self._try_load_current_signal_from_cache()
        self._update_nav_state()
        self._save_app_settings()

    def _load_video_path(self, path: str, apply_item_state: bool = True):
        self._last_video_dir = str(Path(path).parent)
        self._last_dir = self._last_video_dir
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            QMessageBox.critical(self, 'Error', f'Cannot open:\n{path}')
            return
        self._video_path = path
        self._cache = FrameCache(120)
        self._sig_t = None
        self._sig_v = None
        self._invalidate_smooth_cache()
        self._sig_cache_key = None
        self._last_loaded_signal_key = None
        self._fps      = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._fslider.setMaximum(self._n_frames - 1)
        self._fspin.setMaximum(self._n_frames - 1)
        self._start_spin.setMaximum(self._n_frames - 1)
        self._end_spin.setMaximum(self._n_frames)
        self._end_spin.setValue(self._n_frames)
        self._ftotal_lbl.setText(
            f'{self._n_frames}  ({vw}×{vh}, {self._fps:.3f} fps)')

        self._btn_extract.setEnabled(True)
        self._seek(0)
        self._vid.fit_to_widget()
        if apply_item_state:
            item = self._current_project_item()
            roi = item.get('roi') if item else self._default_roi
            if roi:
                self._vid.set_roi(tuple(roi))
                self._on_roi_changed(tuple(roi))
            else:
                self._clear_roi()
            if item:
                self._start_spin.setValue(int(item.get('start_frame', 0)))
                self._end_spin.setValue(min(int(item.get('end_frame', self._n_frames)), self._n_frames))
                self._ch_cb.setCurrentIndex(int(item.get('channel_index', self._ch_cb.currentIndex())))
                self._offset_spin.setValue(float(item.get('offset', 0.0)))
                method = item.get('align_method')
                if method:
                    i = self._align_method_cb.findText(method)
                    if i >= 0:
                        self._align_method_cb.setCurrentIndex(i)
                self._align_model = self._sanitize_align_model(item.get('align_model', self._align_model))
                if 'max_lag_s' in item:
                    self._max_lag_spin.setValue(float(item.get('max_lag_s', self._max_lag_spin.value())))
                self._smooth_spin.setValue(int(item.get('smooth_sigma', self._smooth_spin.value())))
                self._invert_cb.setChecked(bool(item.get('invert', self._invert_cb.isChecked())))
                self._out_mode_cb.setCurrentIndex(int(item.get('output_mode', self._out_mode_cb.currentIndex())))
                manual = bool(item.get('threshold_manual', self._rad_manual.isChecked()))
                self._rad_manual.setChecked(manual)
                self._rad_auto.setChecked(not manual)
                self._on_thr_mode_changed()
                if 'threshold' in item:
                    self._set_threshold(float(item['threshold']))
        self.setWindowTitle(
            f'Video Barcode Signal Extractor — {Path(path).name}')
        self._set_status(
            f'{Path(path).name}  |  {self._n_frames} frames  '
            f'|  {vw}×{vh}  |  {self._fps:.3f} fps')

    @staticmethod
    def _format_timecode(seconds: float):
        total_ms = max(0, int(round(float(seconds) * 1000.0)))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        if h:
            return f'{h:02d}:{m:02d}:{s:02d}.{ms:03d}'
        return f'{m:02d}:{s:02d}.{ms:03d}'

    def _on_zoom_changed(self, scale: float):
        self._zoom_lbl.setText(f'{scale * 100.0:.0f}%')

    # ── frame seeking ─────────────────────────────────────────────────────────
    def _seek(self, idx: int):
        if not self._cap:
            return
        idx = max(0, min(idx, self._n_frames - 1))
        frame = self._cache.get(idx)
        if frame is None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self._cap.read()
            if not ret:
                return
            self._cache.put(idx, frame)
        self._vid.set_frame(frame)

        # live ROI value
        roi = self._vid.get_roi()
        if roi:
            x, y, w, h = roi
            crop = frame[max(0, y):y + h, max(0, x):x + w]
            if crop.size:
                ch_map = {0: None, 1: 2, 2: 1, 3: 0}
                ci = ch_map[self._ch_cb.currentIndex()]
                val = (np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
                       if ci is None else np.mean(crop[:, :, ci]))
                self._roi_val_lbl.setText(f'{val:.1f}')

        # block signals to avoid feedback loop
        for w, v in [(self._fslider, idx), (self._fspin, idx)]:
            w.blockSignals(True)
            w.setValue(idx)
            w.blockSignals(False)

        t = idx / self._fps
        self._cursor_v.setValue(t)
        self._cursor_v2.setValue(t)
        total_t = max(0.0, (self._n_frames - 1) / self._fps) if self._fps else 0.0
        self._time_lbl.setText(
            f'{self._format_timecode(t)} / {self._format_timecode(total_t)}')

    def _on_slider(self, v):
        self._seek(v)

    def _on_spin(self, v):
        self._seek(v)

    # ── playback ──────────────────────────────────────────────────────────────
    SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]

    def _toggle_play(self, checked):
        if checked:
            spd = self.SPEEDS[self._speed_cb.currentIndex()]
            interval = max(1, int(1000.0 / (self._fps * spd)))
            self._play_timer.start(interval)
            self._btn_play.setText('Pause')
        else:
            self._play_timer.stop()
            self._btn_play.setText('Play')

    def _play_step(self):
        cur = self._fslider.value()
        if cur >= self._n_frames - 1:
            self._btn_play.setChecked(False)
            self._toggle_play(False)
            return
        self._seek(cur + 1)

    def keyPressEvent(self, e):
        if e.modifiers() & Qt.ControlModifier and e.key() == Qt.Key_Left:
            self._prev_item()
        elif e.modifiers() & Qt.ControlModifier and e.key() == Qt.Key_Right:
            self._next_item()
        elif e.key() == Qt.Key_Space:
            self._btn_play.setChecked(not self._btn_play.isChecked())
            self._toggle_play(self._btn_play.isChecked())
        elif e.key() == Qt.Key_Left:
            self._seek(self._fslider.value() - 1)
        elif e.key() == Qt.Key_Right:
            self._seek(self._fslider.value() + 1)
        else:
            super().keyPressEvent(e)

    def _zoom100(self):
        self._vid.set_zoom_100()

    # ── ROI ───────────────────────────────────────────────────────────────────
    def _on_roi_changed(self, roi):
        x, y, w, h = roi
        self._roi_lbl.setText(f'x={x}  y={y}  w={w}  h={h}')
        self._btn_extract.setEnabled(True)
        self._default_roi = roi
        item = self._current_project_item()
        if item is not None:
            item['roi'] = roi
        self._update_nav_state()
        if not self._loading_item:
            for other in self._items:
                if other.get('roi') is None:
                    other['roi'] = roi
            self._save_app_settings()
            self._try_load_current_signal_from_cache()

    def _clear_roi(self):
        self._vid.clear_roi()
        self._roi_lbl.setText('Not set')
        self._roi_val_lbl.setText('—')
        item = self._current_project_item()
        if item is not None:
            item['roi'] = None
        self._update_nav_state()

    def _on_extraction_config_changed(self, *_):
        if self._loading_item:
            return
        self._save_current_item_state()
        self._try_load_current_signal_from_cache()

    # ── extraction ────────────────────────────────────────────────────────────
    def _extract_signal(self):
        roi = self._vid.get_roi()
        if roi is None:
            QMessageBox.warning(self, 'ROI needed',
                                'Draw a ROI on the video first.')
            return
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            if not self._worker.wait(3000):
                QMessageBox.warning(
                    self, 'Busy',
                    'A previous extraction is still running and did not stop '
                    'within 3 seconds. Try again in a moment.')
                return

        ch_map = {0: 'gray', 1: 'r', 2: 'g', 3: 'b'}
        channel     = ch_map[self._ch_cb.currentIndex()]
        start_frame = self._start_spin.value()
        end_frame   = min(self._end_spin.value(), self._n_frames)

        # ── cache lookup ──────────────────────────────────────────────────
        key = self._signal_cache_key(roi, channel, start_frame, end_frame)
        cached = self._load_from_cache(key)
        if cached is not None:
            times, values = cached
            self._sig_cache_key = key
            self._set_status(
                f'Loaded from cache  |  {len(values)} samples  '
                f'|  key {key}')
            self._workers_info.setText('(cached)')
            self._loading_from_cache = True
            self._on_extracted(times, values)
            self._loading_from_cache = False
            self._last_loaded_signal_key = key
            self._apply_cache_metadata_to_item(key)
            self._update_nav_state()
            return

        self._extract_start_frame = start_frame
        self._extract_end_frame = end_frame
        self._progress.setRange(start_frame, end_frame)
        self._progress.setValue(start_frame)
        self._progress.setFormat('%p%')
        self._progress_frame_lbl.setText(f'Frame {start_frame} / {end_frame}')
        self._progress_eta_lbl.setText('starting...')
        self._progress_widget.setVisible(True)
        self._abort_btn.setEnabled(True)
        self._btn_extract.setEnabled(False)
        nw = self._workers_spin.value()
        self._set_status(
            f'Extracting frames {start_frame}-{end_frame}  '
            f'({nw} worker{"s" if nw > 1 else ""})...')
        self._extract_t0 = __import__('time').perf_counter()
        self._sig_cache_key = key   # store so _on_extracted can save it

        decoder = 'pyav' if (self._decoder_cb.currentIndex() == 1
                              and self._pyav_available()) else 'opencv'
        self._worker = ExtractionWorker(
            self._video_path, roi, start_frame, end_frame,
            channel, self._fps, n_workers=self._workers_spin.value(),
            decoder=decoder)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_extracted)
        self._worker.error.connect(self._on_extract_error)
        self._worker.start()

    def _on_progress(self, frame: int):
        """Update the progress complex (bar, frame counter, rate, ETA)."""
        self._progress.setValue(int(frame))
        start = int(getattr(self, '_extract_start_frame', 0))
        end = int(getattr(self, '_extract_end_frame', max(1, frame)))
        total = max(1, end - start)
        done = max(0, int(frame) - start)
        pct = 100.0 * done / total
        self._progress_frame_lbl.setText(
            f'Frame {int(frame):>7d} / {end}  ({pct:5.1f}%)')
        elapsed = __import__('time').perf_counter() - getattr(self, '_extract_t0', 0.0)
        if elapsed > 0.5 and done > 0:
            rate = done / elapsed
            remaining = max(0, total - done)
            eta_s = remaining / max(rate, 1e-9)
            self._progress_eta_lbl.setText(
                f'{rate:6.0f} fr/s  ETA {self._format_eta(eta_s)}')

    @staticmethod
    def _format_eta(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        if seconds < 60:
            return f'{seconds:.0f}s'
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f'{m}m {s:02d}s'
        h, m = divmod(m, 60)
        return f'{h}h {m:02d}m'

    def _abort_extraction(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.abort()
            self._abort_btn.setEnabled(False)
            self._progress_eta_lbl.setText('aborting...')
            self._log('Extraction abort requested by user.', level='warn')
            self._set_status('Abort requested; waiting for workers to stop.')

    def _hide_progress_complex(self):
        self._progress_widget.setVisible(False)
        self._progress_frame_lbl.setText('')
        self._progress_eta_lbl.setText('')
        self._abort_btn.setEnabled(True)

    def _on_extract_error(self, msg: str):
        self._log(f'Extraction error: {msg}', level='error')
        self._set_status(f'Error: {msg}')
        self._btn_extract.setEnabled(True)
        self._hide_progress_complex()
        if getattr(self, '_batch_state', None):
            state = self._batch_state
            state['skipped'].append(f'(error) {msg}')
            QTimer.singleShot(50, self._batch_extract_next)

    # ── batch extract ───────────────────────────────────────────────────────
    def _batch_extract(self):
        if not self._items:
            QMessageBox.warning(self, 'No items',
                                'Load a video list or project first.')
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, 'Busy',
                                'An extraction is already running.')
            return
        if getattr(self, '_batch_state', None):
            return  # already running
        self._batch_state = {
            'pending': list(range(len(self._items))),
            'done': 0,
            'skipped': [],
            'total': len(self._items),
        }
        self._log(f'Batch extract: {len(self._items)} item(s) queued.')
        self._set_status('Batch extract started.')
        self._log_dock.setVisible(True)
        QTimer.singleShot(0, self._batch_extract_next)

    def _batch_extract_next(self):
        state = getattr(self, '_batch_state', None)
        if not state:
            return
        if not state['pending']:
            n_done = state['done']
            n_skip = len(state['skipped'])
            self._log(
                f'Batch extract complete: {n_done} extracted, {n_skip} skipped.')
            if state['skipped']:
                for reason in state['skipped'][:10]:
                    self._log(f'  skipped: {reason}', level='warn')
            self._set_status(
                f'Batch extract done: {n_done} ok, {n_skip} skipped.')
            self._batch_state = None
            return
        idx = state['pending'].pop(0)
        self._log(f'Batch extract: item {idx + 1}/{state["total"]}.')
        self._load_item(idx)
        roi = self._vid.get_roi()
        if roi is None:
            state['skipped'].append(
                f'{Path(self._items[idx]["video"]).name} (no ROI)')
            QTimer.singleShot(0, self._batch_extract_next)
            return
        # Skip if already cached for this exact config
        ch_map = {0: 'gray', 1: 'r', 2: 'g', 3: 'b'}
        channel = ch_map[self._ch_cb.currentIndex()]
        start_frame = self._start_spin.value()
        end_frame = min(self._end_spin.value(), self._n_frames)
        key = signal_cache_key(self._video_path, roi, channel,
                               start_frame, end_frame)
        if load_signal_cache(self._video_path, key) is not None:
            self._log(f'  item {idx + 1}: already cached, skipping extraction')
            state['done'] += 1
            QTimer.singleShot(0, self._batch_extract_next)
            return
        # Kick off a real extraction. _on_extracted will advance the batch.
        self._extract_signal()

    def _on_extracted(self, times: np.ndarray, values: np.ndarray):
        elapsed = __import__('time').perf_counter() - getattr(self, '_extract_t0', 0)
        fps_eff = len(values) / elapsed if elapsed > 0 else 0
        self._sig_t = times
        self._sig_v = values
        self._invalidate_smooth_cache()
        self._hide_progress_complex()
        self._btn_extract.setEnabled(True)
        key = getattr(self, '_sig_cache_key', None)
        if not self._loading_from_cache:
            self._workers_info.setText(f'{elapsed:.1f} s  |  {fps_eff:.0f} fr/s')
            # auto-save to cache
            if key:
                self._save_to_cache(key, times, values)
                self._last_loaded_signal_key = key
                self._apply_cache_metadata_to_item(key)
        elif key:
            self._last_loaded_signal_key = key

        # update region to span data
        if len(times) > 1:
            self._region.setRegion([times[0], times[-1]])

        self._raw_curve.setData(times, values)

        # Set initial view to first 30 s window so the plot isn't overloaded
        WIN = 30.0
        t0, t1 = float(times[0]), float(times[-1])
        view_end = min(t0 + WIN, t1)
        self._trace_pw.setXRange(t0, view_end, padding=0)

        # Update slider range to match actual signal range
        if len(values):
            lo, hi = float(values.min()), float(values.max())
            # Use 4000 integer steps across the real range for fine manual control
            self._th_slider_lo = lo
            self._th_slider_hi = hi
            self._th_slider.setRange(0, 4000)

            if self._rad_auto.isChecked():
                self._apply_auto_threshold()
            else:
                mid = (lo + hi) / 2.0
                self._set_threshold(mid)

        self._refresh_view()
        self._update_quality_readout()

        if self._csv_df is not None:
            self._btn_xcorr.setEnabled(True)

        self._update_nav_state()
        self._update_quality_readout()
        self._set_status(
            f'Extracted {len(values)} samples  |  '
            f'{times[-1]:.2f} s  |  '
            f'min={values.min():.1f}  max={values.max():.1f}')

        # If we are in a batch extract run, advance to the next item.
        if getattr(self, '_batch_state', None):
            QTimer.singleShot(50, self._batch_extract_next)

    # ── viewport helpers ──────────────────────────────────────────────────────────
    # Max points sent to the plot at once.  pyqtgraph peak-downsampling
    # handles the rest; 20k is smooth and responsive.
    _MAX_PTS = 20_000

    def _view_range(self):
        """Current x-axis range of the signal plot (seconds)."""
        vb = self._trace_pw.getViewBox()
        (x0, x1), _ = vb.viewRange()
        return float(x0), float(x1)

    def _slice_to_view(self, t: np.ndarray, *arrays,
                       pad: float = 0.0):
        """
        Return index slice [i0:i1] covering [x0-pad, x1+pad] of *t*,
        plus optionally decimated arrays if the slice is still too large.
        """
        if t is None or len(t) == 0:
            return slice(0, 0)
        x0, x1 = self._view_range()
        i0 = max(0,       int(np.searchsorted(t, x0 - pad, side='left')  - 1))
        i1 = min(len(t),  int(np.searchsorted(t, x1 + pad, side='right') + 1))
        span = i1 - i0
        if span > self._MAX_PTS:
            step = max(1, span // self._MAX_PTS)
            # always include the last point of the slice
            idx = np.arange(i0, i1, step)
            if idx[-1] != i1 - 1:
                idx = np.append(idx, i1 - 1)
            return idx
        return slice(i0, i1)

    # ── threshold / view ──────────────────────────────────────────────────────
    def _thr_real_to_slider(self, t: float) -> int:
        lo = getattr(self, '_th_slider_lo', 0.0)
        hi = getattr(self, '_th_slider_hi', 255.0)
        span = hi - lo if hi > lo else 1.0
        return int((t - lo) / span * 4000)

    def _thr_slider_to_real(self, v: int) -> float:
        lo = getattr(self, '_th_slider_lo', 0.0)
        hi = getattr(self, '_th_slider_hi', 255.0)
        return lo + (v / 4000.0) * (hi - lo)

    def _set_threshold(self, t: float):
        """Update threshold line, slider, and label without feedback loops."""
        self._thresh_line.blockSignals(True)
        self._thresh_line.setValue(t)
        self._thresh_line.blockSignals(False)
        self._th_slider.blockSignals(True)
        self._th_slider.setValue(self._thr_real_to_slider(t))
        self._th_slider.blockSignals(False)
        self._th_val_lbl.setText(f'{t:.2f}')

    def _on_thr_mode_changed(self):
        manual = self._rad_manual.isChecked()
        self._th_slider.setEnabled(manual)
        self._thr_method_cb.setEnabled(not manual)
        self._thresh_line.setMovable(manual)
        if not manual:
            self._apply_auto_threshold()

    def _on_thr_method_changed(self):
        if self._rad_auto.isChecked():
            self._apply_auto_threshold()

    def _apply_auto_threshold(self):
        if self._sig_v is None:
            return
        sm = self._smooth_spin.value()
        v = gaussian_filter1d(self._sig_v, sm) if sm > 0 else self._sig_v
        method = self._thr_method_cb.currentText() if hasattr(self, '_thr_method_cb') else 'Auto (recommended)'
        if method.startswith('Auto'):
            t, picked = auto_threshold(v)
            self._set_status(f'Auto threshold: {picked} → {t:.2f}')
        else:
            fn = THRESHOLD_METHODS.get(method)
            if fn is None:
                return
            t = float(fn(v))
        self._set_threshold(t)
        self._refresh_view()
        self._update_quality_readout()

    # Backward-compatibility shim: a few internal callers still say _apply_otsu.
    def _apply_otsu(self):
        self._apply_auto_threshold()

    def _update_quality_readout(self):
        """Recompute and display signal-quality stats."""
        if not hasattr(self, '_q_snr_lbl'):
            return
        if self._sig_v is None or self._sig_t is None:
            for lbl in (self._q_snr_lbl, self._q_edges_lbl, self._q_cv_lbl,
                        self._q_sat_lbl):
                lbl.setText('—')
            return
        sig, output = self._get_processed_signal()
        if sig is None:
            return
        thr = float(self._thresh_line.value())
        snr = signal_snr_db(sig, thr)
        ei = edge_interval_stats(self._sig_t, output)
        sat = saturated_fraction(self._sig_v)
        self._q_snr_lbl.setText('inf' if np.isinf(snr) else
                                ('nan' if np.isnan(snr) else f'{snr:.1f}'))
        self._q_edges_lbl.setText(f"{ei['n_edges']}")
        if np.isnan(ei['cv']):
            self._q_cv_lbl.setText('—')
        else:
            self._q_cv_lbl.setText(
                f"{ei['cv']:.3f}  (mean={ei['mean_s']:.3f}s)")
        self._q_sat_lbl.setText(f'{sat * 100.0:.2f}%')

    def _on_th_slider(self, val):
        if not self._rad_manual.isChecked():
            return
        t = self._thr_slider_to_real(val)
        self._thresh_line.blockSignals(True)
        self._thresh_line.setValue(t)
        self._thresh_line.blockSignals(False)
        self._th_val_lbl.setText(f'{t:.2f}')
        self._refresh_view()

    def _on_thresh_drag(self):
        if not self._rad_manual.isChecked():
            return
        t = self._thresh_line.value()
        self._th_slider.blockSignals(True)
        self._th_slider.setValue(self._thr_real_to_slider(t))
        self._th_slider.blockSignals(False)
        self._th_val_lbl.setText(f'{t:.2f}')
        self._refresh_view()

    def _invalidate_smooth_cache(self):
        self._smooth_cache_key = None
        self._smooth_cache_val = None

    def _get_processed_signal(self):
        """Return (smoothed_signal, output_signal) where output depends on mode.

        The smoothed signal is memoized keyed on (id(sig_v), len, sigma).
        Use _invalidate_smooth_cache() whenever self._sig_v is reassigned.
        """
        if self._sig_v is None:
            return None, None
        sm = self._smooth_spin.value()
        if sm <= 0:
            sig = self._sig_v
        else:
            key = (id(self._sig_v), len(self._sig_v), sm)
            if getattr(self, '_smooth_cache_key', None) == key:
                sig = self._smooth_cache_val
            else:
                sig = gaussian_filter1d(self._sig_v, sm).astype(np.float32)
                self._smooth_cache_key = key
                self._smooth_cache_val = sig
        thr  = float(self._thresh_line.value())
        mask = (sig >= thr)
        if self._invert_cb.isChecked():
            mask = ~mask
        gated_mode = self._out_mode_cb.currentIndex() == 1
        if gated_mode:
            output = np.where(mask, sig, 0.0).astype(np.float32)
        else:
            output = mask.astype(np.float32)
        return sig, output

    @staticmethod
    def _norm01(x):
        return norm01(x)

    def _alignment_method_key(self):
        text = self._align_method_cb.currentText().lower()
        if 'dtw' in text:
            return 'dtw'
        if 'interpolation' in text:
            return 'edge_interpolation'
        if 'regression' in text:
            return 'linear_regression'
        return 'cross_correlation'

    def _on_alignment_method_changed(self, *_):
        self._align_model = self._sanitize_align_model(self._align_model)
        self._align_model['method'] = self._alignment_method_key()
        self._redraw_alignment_diagnostic()
        self._refresh_cmp()
        self._persist_current_cache_metadata()

    def _offset_direction_text(self, offset: float) -> str:
        if abs(offset) < 0.5 / max(self._fps, 1e-9):
            return 'in phase'
        if offset > 0:
            return f'video behind reference by {offset:.4f} s'
        return f'video ahead of reference by {abs(offset):.4f} s'

    def _edge_times(self, t, v):
        return edge_times(t, v)

    def _pair_edges(self, video_edges, video_dirs, ref_edges, ref_dirs, offset):
        return pair_edges(video_edges, video_dirs, ref_edges, ref_dirs,
                          offset, self._fps)

    def _estimate_xcorr_offset(self, t_csv, v_csv, output):
        max_lag = float(self._max_lag_spin.value()) if hasattr(self, '_max_lag_spin') else 2.5
        return estimate_xcorr_offset(
            self._sig_t, output, t_csv, v_csv,
            max_lag_s=max_lag, fps=self._fps,
            view_range=self._view_range())

    def _reset_diagnostic_plot(self):
        self._xcorr_curve.setData([], [])
        self._align_fit_curve.setData([], [])
        self._xcorr_lag_line.setVisible(False)
        self._xcorr_curve.setSymbol(None)
        self._xcorr_curve.setPen(pg.mkPen(MAUVE, width=1.5))

    def _show_correlation_diagnostic(self, xcorr):
        self._reset_diagnostic_plot()
        self._xcorr_pw.setLabel('bottom', 'Lag', units='s')
        self._xcorr_pw.setLabel('left', 'Pearson r')
        self._xcorr_curve.setData(xcorr['lags'].astype(np.float32),
                                  xcorr['corr_norm'].astype(np.float32))
        self._xcorr_lag_line.setVisible(True)
        self._xcorr_lag_line.setValue(float(xcorr['lag']))
        self._xcorr_pw.setYRange(-1.05, 1.05, padding=0)
        if len(xcorr.get('lags', [])):
            self._xcorr_pw.setXRange(float(np.nanmin(xcorr['lags'])),
                                     float(np.nanmax(xcorr['lags'])),
                                     padding=0.02)

    def _show_regression_diagnostic(self, pairs_v, pairs_r, slope, intercept):
        self._reset_diagnostic_plot()
        self._xcorr_pw.setLabel('bottom', 'Video edge time', units='s')
        self._xcorr_pw.setLabel('left', 'Reference edge time', units='s')
        self._xcorr_curve.setPen(None)
        self._xcorr_curve.setSymbol('o')
        self._xcorr_curve.setSymbolSize(5)
        self._xcorr_curve.setSymbolBrush(pg.mkBrush(MAUVE))
        self._xcorr_curve.setData(pairs_v.astype(np.float32), pairs_r.astype(np.float32))
        if len(pairs_v):
            xline = np.array([pairs_v.min(), pairs_v.max()], dtype=np.float64)
            yline = slope * xline + intercept
            self._align_fit_curve.setData(xline.astype(np.float32), yline.astype(np.float32))

    def _show_interpolation_diagnostic(self, pairs_v, pairs_r):
        self._reset_diagnostic_plot()
        self._xcorr_pw.setLabel('bottom', 'Video edge time', units='s')
        self._xcorr_pw.setLabel('left', 'Reference-video shift', units='s')
        shift = pairs_r - pairs_v
        self._xcorr_curve.setPen(None)
        self._xcorr_curve.setSymbol('o')
        self._xcorr_curve.setSymbolSize(5)
        self._xcorr_curve.setSymbolBrush(pg.mkBrush(MAUVE))
        self._xcorr_curve.setData(pairs_v.astype(np.float32), shift.astype(np.float32))
        self._align_fit_curve.setData(pairs_v.astype(np.float32), shift.astype(np.float32))

    def _redraw_alignment_diagnostic(self):
        model = self._sanitize_align_model(self._align_model)
        self._align_model = model
        method = self._alignment_method_key()
        pairs_v = np.asarray(model.get('video_edges', []), dtype=np.float64).reshape(-1)
        pairs_r = np.asarray(model.get('ref_edges', []), dtype=np.float64).reshape(-1)
        if method == 'linear_regression' and pairs_v.size >= 2 and pairs_r.size == pairs_v.size:
            self._show_regression_diagnostic(
                pairs_v, pairs_r,
                float(model.get('slope', 1.0)),
                float(model.get('intercept', 0.0)))
        elif method == 'edge_interpolation' and pairs_v.size >= 2 and pairs_r.size == pairs_v.size:
            self._show_interpolation_diagnostic(pairs_v, pairs_r)
        elif self._sig_t is not None and self._csv_df is not None:
            try:
                tc = self._csv_t_cb.currentText()
                ac = self._csv_align_cb.currentText()
                _, output = self._get_processed_signal()
                xcorr = self._estimate_xcorr_offset(
                    self._csv_df[tc].values.astype(np.float64),
                    self._csv_df[ac].values.astype(np.float64),
                    output)
                if xcorr:
                    self._show_correlation_diagnostic(xcorr)
            except Exception:
                self._reset_diagnostic_plot()
        else:
            self._reset_diagnostic_plot()

    def _aligned_time_for_video_times(self, times):
        model = self._sanitize_align_model(self._align_model)
        model['method'] = self._alignment_method_key()
        return aligned_time_for_video_times(
            times, model, offset_fallback=self._offset_spin.value())

    def _video_time_from_aligned_time(self, aligned_time):
        model = self._sanitize_align_model(self._align_model)
        model['method'] = self._alignment_method_key()
        return video_time_from_aligned_time(
            aligned_time, model, offset_fallback=self._offset_spin.value())

    def _alignment_metrics(self, output_norm, aligned_t, t_csv, v_csv, x_range=None):
        return alignment_metrics(output_norm, aligned_t, t_csv, v_csv,
                                 fps=self._fps, x_range=x_range)

    def _refresh_view(self):
        sig, output = self._get_processed_signal()
        if sig is None:
            return
        lo, hi = float(sig.min()), float(sig.max())
        span   = hi - lo if hi > lo else 1.0
        gated_mode = self._out_mode_cb.currentIndex() == 1

        # Slice to visible window only
        idx = self._slice_to_view(self._sig_t)
        t_v   = self._sig_t[idx]
        sig_v = sig[idx]
        out_v = output[idx]

        self._raw_curve.setData(t_v, sig_v)
        if gated_mode:
            self._bin_curve.setData(t_v, out_v)
            self._bin_curve.opts['name'] = 'Gated peaks'
        else:
            bin_lo = lo
            bin_hi = lo + span * 0.30
            self._bin_curve.setData(t_v, bin_lo + out_v * (bin_hi - bin_lo))
            self._bin_curve.opts['name'] = 'Binary (scaled)'
        self._refresh_cmp()

    def _refresh_cmp(self):
        """Redraw alignment overlay plot (viewport-sliced)."""
        _, output = self._get_processed_signal()
        if output is None:
            return

        # Normalise over full signal, then slice for display only
        out_norm_full = self._norm01(output)
        aligned_t_full = self._aligned_time_for_video_times(self._sig_t)
        ext_y_offset = 1.25
        idx = self._slice_to_view(self._sig_t)
        self._ext_curve.setData(aligned_t_full[idx],
                                (out_norm_full[idx] + ext_y_offset).astype(np.float32))

        if self._csv_df is not None:
            tc = self._csv_t_cb.currentText()
            vc = self._csv_v_cb.currentText()
            ac = self._csv_align_cb.currentText()
            if tc and vc:
                try:
                    t_csv = self._csv_df[tc].values.astype(np.float64)
                    v_csv = self._csv_df[vc].values.astype(np.float64)
                    v_align = self._csv_df[ac].values.astype(np.float64) if ac else v_csv
                    csv_norm  = self._norm01(v_csv).astype(np.float32)

                    # Slice CSV to visible window
                    x0, x1 = self._view_range()
                    ax = self._aligned_time_for_video_times(np.array([x0, x1], dtype=np.float64))
                    rx0, rx1 = float(np.nanmin(ax)), float(np.nanmax(ax))
                    ci0 = max(0,          int(np.searchsorted(t_csv, rx0, 'left')  - 1))
                    ci1 = min(len(t_csv), int(np.searchsorted(t_csv, rx1, 'right') + 1))
                    cspan = ci1 - ci0
                    if cspan > self._MAX_PTS:
                        step = max(1, cspan // self._MAX_PTS)
                        cidx: np.ndarray | slice = np.arange(ci0, ci1, step)
                    else:
                        cidx = slice(ci0, ci1)
                    self._csv_curve.setData(t_csv[cidx], csv_norm[cidx])

                    metrics = self._alignment_metrics(out_norm_full, aligned_t_full, t_csv, v_csv)
                    metrics_view = self._alignment_metrics(
                        out_norm_full, aligned_t_full, t_csv, v_csv, x_range=(rx0, rx1))
                    ext_edges, ext_dirs = self._edge_times(self._sig_t, output)
                    ref_edges, ref_dirs = self._edge_times(t_csv, v_align)
                    model = self._sanitize_align_model(self._align_model)
                    self._align_model = model
                    matched = int(model.get('edge_pairs', 0))
                    method = self._alignment_method_key()

                    # ── line 1: timing + quality ─────────────────────────────
                    method_text = ''
                    if method == 'cross_correlation':
                        peak = model.get('xcorr_peak')
                        method_text = f"   peakR={peak:.4f}" if peak is not None else ''
                    elif method == 'linear_regression' and matched >= 2:
                        method_text = (
                            f"   slope={float(model.get('slope', 1.0)):.8f}"
                            f"   intercept={float(model.get('intercept', 0.0)):.4f}s"
                            f"   drift={float(model.get('drift_s', 0.0)):+.4f}s")
                        if 'ransac_inliers' in model:
                            method_text += (
                                f"   RANSAC inliers={model['ransac_inliers']}"
                                f"/{model['ransac_total']}")
                    elif method == 'edge_interpolation' and matched >= 2:
                        method_text = (f"   piecewise warp"
                                       f"   segments={max(0, matched - 1)}")
                    elif method == 'dtw' and matched >= 2:
                        cost = model.get('dtw_cost')
                        win = model.get('dtw_window')
                        n_ds = model.get('downsampled_n')
                        bits = [f'anchors={matched}']
                        if cost is not None: bits.append(f'cost={cost:.4f}')
                        if win is not None: bits.append(f'window={win}')
                        if n_ds is not None: bits.append(f'N={n_ds}')
                        method_text = '   DTW  ' + '  '.join(bits)

                    if metrics:
                        primary = metrics_view or metrics
                        all_text = f"   r_all={metrics['r']:.4f}" if metrics_view else ""
                        self._align_stats_lbl.setText(
                            f"{self._offset_direction_text(self._offset_spin.value())}"
                            f"   r_view={primary['r']:.4f}{all_text}"
                            f"   RMSE={primary['rmse']:.3f}"
                            f"   MAE={primary['mae']:.3f}"
                            f"   bin-agree={primary['agree']:.1%}"
                            f"   overlap={primary['overlap_s']:.2f}s"
                            f"{method_text}")
                    else:
                        self._align_stats_lbl.setText(
                            f"(no overlap){method_text}")

                    # ── line 2: edge / anchor counts (always shown) ──────────
                    if method == 'dtw':
                        anchors_label = 'anchors'
                    else:
                        anchors_label = 'matched edges'
                    edge_bits = [
                        f"video edges: {len(ext_edges)}",
                        f"reference edges: {len(ref_edges)}",
                        f"{anchors_label}: {matched}",
                    ]
                    if (method in ('linear_regression', 'edge_interpolation', 'dtw')
                            and len(aligned_t_full) > 1):
                        s0 = float(aligned_t_full[0] - self._sig_t[0])
                        s1 = float(aligned_t_full[-1] - self._sig_t[-1])
                        edge_bits.append(f"endpoint shift {s0:+.4f} -> {s1:+.4f} s")
                    self._align_edge_lbl.setText('   '.join(edge_bits))
                except Exception:
                    pass

    # ── CSV ───────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_csv_robust(path: str) -> pd.DataFrame:
        return load_csv_robust(path)

    def _open_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open CSV trace', self._last_csv_dir,
            'CSV / TXT (*.csv *.txt *.tsv);;All (*)')
        if not path:
            return
        self._last_csv_dir = str(Path(path).parent)
        self._last_dir = self._last_csv_dir
        if self._current_item >= 0 and self._current_item < len(self._items):
            self._items[self._current_item]['csv'] = path
        self._load_csv_path(path)
        self._save_app_settings()

    def _load_csv_path(self, path: str):
        self._last_csv_dir = str(Path(path).parent)
        self._last_dir = self._last_csv_dir
        try:
            df = self._load_csv_robust(path)
        except Exception as e:
            QMessageBox.critical(self, 'CSV Error', str(e))
            return
        self._csv_df = df
        self._csv_path = path
        cols = list(df.columns)
        for cb in (self._csv_t_cb, self._csv_v_cb, self._csv_align_cb):
            cb.blockSignals(True)
            cb.clear()
            for c in cols:
                cb.addItem(c)
            cb.blockSignals(False)
        # heuristic: pick likely time column
        time_idx = 0
        for i, c in enumerate(cols):
            if any(k in c.lower() for k in ['time', 'sec', 'ms', 't_']):
                time_idx = i
                break
        self._csv_t_cb.setCurrentIndex(time_idx)
        # default display & align: prefer digital/reference columns over dFF/AIN traces
        val_idx = next(
            (i for i, c in enumerate(cols)
             if i != time_idx and any(k in str(c).lower()
                                      for k in ['dio', 'ttl', 'sync', 'barcode', 'trigger'])),
            next((i for i in range(len(cols)) if i != time_idx), time_idx))
        self._csv_v_cb.setCurrentIndex(val_idx)
        self._csv_align_cb.setCurrentIndex(val_idx)
        item = self._current_project_item()
        if item:
            for key, cb in [('time_col', self._csv_t_cb), ('display_col', self._csv_v_cb), ('align_col', self._csv_align_cb)]:
                col = item.get(key)
                if col in cols:
                    cb.setCurrentText(col)
        if self._sig_t is not None:
            self._btn_xcorr.setEnabled(True)
        self._set_status(
            f'CSV: {Path(path).name}  |  {len(df)} rows  |  cols: {cols}')
        self._update_nav_state()
        self._refresh_cmp()

    # ── auto-align (cross-correlation) ───────────────────────────────────────
    def _auto_align(self):
        if self._sig_t is None or self._csv_df is None:
            return
        tc = self._csv_t_cb.currentText()
        ac = self._csv_align_cb.currentText()   # column used for xcorr
        if not tc or not ac:
            return
        try:
            t_csv = self._csv_df[tc].values.astype(np.float64)
            v_csv = self._csv_df[ac].values.astype(np.float64)
        except Exception:
            return

        _, output = self._get_processed_signal()
        if output is None:
            return

        xcorr = self._estimate_xcorr_offset(t_csv, v_csv, output)
        if xcorr is None:
            self._set_status('Too little overlap for alignment.')
            return

        method = self._alignment_method_key()
        offset = float(xcorr['offset'])
        peak_r = float(xcorr['peak'])
        low_r_warning = ''
        if peak_r < DEFAULT_MIN_PEAK_R:
            low_r_warning = (
                f'  WARNING: peak r={peak_r:.3f} < {DEFAULT_MIN_PEAK_R:.2f}; '
                'alignment may be unreliable. Verify visually.')
        current_aligned = self._aligned_time_for_video_times(np.array(self._view_range(), dtype=np.float64))
        proposed_aligned = np.array(self._view_range(), dtype=np.float64) - offset
        ref_lo, ref_hi = float(np.nanmin(t_csv)), float(np.nanmax(t_csv))
        has_current_overlap = not (np.nanmax(current_aligned) < ref_lo or np.nanmin(current_aligned) > ref_hi)
        has_proposed_overlap = not (np.nanmax(proposed_aligned) < ref_lo or np.nanmin(proposed_aligned) > ref_hi)
        if has_current_overlap and not has_proposed_overlap:
            self._show_correlation_diagnostic(xcorr)
            self._set_status(
                f'Auto-align rejected: proposed offset {offset:.4f}s would remove visible overlap. '
                f'Use a larger visible window or increase Max lag only if the true offset is large.')
            return
        self._offset_spin.setValue(offset)
        self._show_correlation_diagnostic(xcorr)

        ext_edges, ext_dirs = self._edge_times(self._sig_t, output)
        ref_edges, ref_dirs = self._edge_times(t_csv, v_csv)
        pairs_v, pairs_r = self._pair_edges(ext_edges, ext_dirs, ref_edges, ref_dirs, offset)

        model = {
            'method': method,
            'offset': offset,
            'slope': 1.0,
            'intercept': -offset,
            'video_edges': pairs_v.tolist(),
            'ref_edges': pairs_r.tolist(),
            'edge_pairs': int(len(pairs_v)),
            'n_ext_edges': int(len(ext_edges)),
            'n_ref_edges': int(len(ref_edges)),
            'xcorr_peak': float(xcorr['peak']),
        }

        # DTW path: replaces the model with full-waveform anchors. We still
        # ran xcorr to set the residual offset spinbox to a sensible value.
        if method == 'dtw':
            dtw = dtw_alignment(self._sig_t, output, t_csv, v_csv,
                                fps=self._fps,
                                max_warp_s=float(self._max_lag_spin.value())
                                            if hasattr(self, '_max_lag_spin') else 2.0)
            if dtw is None:
                self._set_status('DTW failed: signals too short, flat, '
                                 'or with no overlapping time range.')
                self._align_model = sanitize_align_model(model)
                self._refresh_cmp()
                return
            model.update(dtw)
            self._align_model = model
            self._show_interpolation_diagnostic(
                np.asarray(model['video_edges'], dtype=np.float64),
                np.asarray(model['ref_edges'], dtype=np.float64))
            self._right_tabs.setCurrentIndex(1)
            self._refresh_cmp()
            self._persist_current_cache_metadata()
            self._set_status(
                f"DTW alignment: {model['edge_pairs']} anchors, "
                f"cost={model.get('dtw_cost', float('nan')):.4f}, "
                f"N={model.get('downsampled_n', '?')} samples after downsampling"
                f"{low_r_warning}")
            return

        if method in ('linear_regression', 'edge_interpolation'):
            if len(pairs_v) < 2:
                self._align_model = model
                self._right_tabs.setCurrentIndex(1)
                self._refresh_cmp()
                self._set_status(
                    f'Edge alignment needs at least 2 matched edges; found '
                    f'{len(pairs_v)} match(es), ext={len(ext_edges)}, ref={len(ref_edges)}.')
                return
            if method == 'linear_regression':
                ransac = ransac_regression(pairs_v, pairs_r, residual_threshold_s=0.05)
                if ransac is not None and ransac['inliers'] >= 3:
                    slope = ransac['slope']
                    intercept = ransac['intercept']
                    model['ransac_inliers'] = ransac['inliers']
                    model['ransac_total'] = ransac['total']
                else:
                    slope, intercept = np.polyfit(pairs_v, pairs_r, 1)
                model['slope'] = float(slope)
                model['intercept'] = float(intercept)
                model['drift_s'] = float(
                    (pairs_r[-1] - pairs_v[-1]) - (pairs_r[0] - pairs_v[0]))
                self._show_regression_diagnostic(pairs_v, pairs_r, slope, intercept)
            else:
                slope, intercept = np.polyfit(pairs_v, pairs_r, 1)
                model['slope'] = float(slope)
                model['intercept'] = float(intercept)
                model['drift_s'] = float(
                    (pairs_r[-1] - pairs_v[-1]) - (pairs_r[0] - pairs_v[0]))
                self._show_interpolation_diagnostic(pairs_v, pairs_r)

        self._align_model = model
        self._right_tabs.setCurrentIndex(1)
        self._refresh_cmp()
        self._persist_current_cache_metadata()
        ransac_text = ''
        if 'ransac_inliers' in model:
            ransac_text = f'  RANSAC inliers={model["ransac_inliers"]}/{model["ransac_total"]}'
        self._set_status(
            f'Auto-align ({self._align_method_cb.currentText()}): '
            f'offset={offset:.4f}s  {self._offset_direction_text(offset)}  '
            f'edges ext={len(ext_edges)} ref={len(ref_edges)} matched={len(pairs_v)}  '
            f'peak R={xcorr["peak"]:.3f}{ransac_text}{low_r_warning}')

    # ── export ────────────────────────────────────────────────────────────────
    def _default_export_path(self, video_path: str | None = None, out_dir: str | None = None) -> str:
        base = Path(video_path or self._video_path or 'signal').stem
        folder = Path(out_dir or self._last_export_dir)
        return str(folder / f'{base}_aligned_time.csv')

    def _export_dataframe(self, times=None, raw=None):
        if times is None:
            times = self._sig_t
        if raw is None:
            raw = self._sig_v
        sig, output = self._get_processed_signal()
        if times is not self._sig_t or raw is not self._sig_v:
            old_t, old_v = self._sig_t, self._sig_v
            self._sig_t, self._sig_v = times, raw
            sig, output = self._get_processed_signal()
            self._sig_t, self._sig_v = old_t, old_v
        offset = float(self._offset_spin.value())
        aligned_time = self._aligned_time_for_video_times(times)
        binary = (output > 0).astype(np.int8)
        df = pd.DataFrame({
            'aligned_time': aligned_time,
            'time_s':  times,
            'frame':   np.rint(times * self._fps).astype(np.int32),
            'raw':     raw,
            'smoothed': sig,
            'binary':  binary,
            'offset_s': offset,
            'alignment_method': self._align_method_cb.currentText(),
        })
        if self._csv_df is not None:
            try:
                tc = self._csv_t_cb.currentText()
                vc = self._csv_v_cb.currentText()
                ac = self._csv_align_cb.currentText()
                t_ref = self._csv_df[tc].to_numpy(dtype=np.float64)
                v_ref = self._csv_df[vc].to_numpy(dtype=np.float64)
                df['reference_time'] = aligned_time
                df['reference_signal'] = np.interp(
                    aligned_time, t_ref, v_ref, left=np.nan, right=np.nan)
                if ac and ac in self._csv_df.columns:
                    v_align = self._csv_df[ac].to_numpy(dtype=np.float64)
                    df['reference_align_signal'] = np.interp(
                        aligned_time, t_ref, v_align, left=np.nan, right=np.nan)
                df['reference_display_col'] = vc
                df['reference_align_col'] = ac
                df['reference_source_csv'] = self._csv_path or ''
            except Exception:
                pass
        return df

    def _write_export_metadata(self, csv_path: str):
        meta = self._current_extraction_metadata(getattr(self, '_sig_cache_key', None))
        meta['export_csv'] = csv_path
        meta['aligned_time_definition'] = (
            'Cross-correlation: aligned_time = video time_s - offset_s. '
            'Linear regression/interpolation: aligned_time is mapped from video time '
            'to reference time using matched signal edges.'
        )
        try:
            Path(csv_path).with_suffix('.metadata.json').write_text(
                json.dumps(meta, indent=2), encoding='utf-8')
        except Exception:
            pass

    def _export_signal(self):
        if self._sig_t is None:
            QMessageBox.warning(self, 'No data',
                                'Extract a signal first.')
            return
        self._save_current_item_state()
        default_export = self._default_export_path()
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export signal', default_export, 'CSV (*.csv)')
        if not path:
            return
        self._last_export_dir = str(Path(path).parent)
        self._last_dir = self._last_export_dir
        df = self._export_dataframe()
        df.to_csv(path, index=False)
        self._write_export_metadata(path)
        self._persist_current_cache_metadata()
        self._save_app_settings()
        self._set_status(f'Exported → {path}')

    def _open_aligned_signal(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load extracted/aligned-time CSV', self._last_export_dir,
            'CSV / TXT (*.csv *.txt *.tsv);;All (*)')
        if not path:
            return
        self._last_export_dir = str(Path(path).parent)
        self._last_dir = self._last_export_dir
        try:
            df = self._load_csv_robust(path)
        except Exception as e:
            QMessageBox.critical(self, 'CSV Error', str(e))
            return
        has_embedded_ref = 'aligned_time' in df.columns and 'reference_signal' in df.columns
        t_col = 'aligned_time' if has_embedded_ref else (
            'time_s' if 'time_s' in df.columns else ('aligned_time' if 'aligned_time' in df.columns else df.columns[0]))
        v_col = 'raw' if 'raw' in df.columns else ('binary' if 'binary' in df.columns else df.columns[min(1, len(df.columns) - 1)])
        self._sig_t = df[t_col].to_numpy(dtype=np.float64)
        self._sig_v = df[v_col].to_numpy(dtype=np.float32)
        self._invalidate_smooth_cache()
        if has_embedded_ref:
            self._offset_spin.setValue(0.0)
            self._align_model = {
                'method': 'cross_correlation',
                'offset': 0.0,
                'slope': 1.0,
                'intercept': 0.0,
                'video_edges': [],
                'ref_edges': [],
                'edge_pairs': 0,
            }
        elif 'offset_s' in df.columns and len(df):
            self._offset_spin.setValue(float(df['offset_s'].iloc[0]))
        if 'alignment_method' in df.columns and len(df):
            method = str(df['alignment_method'].iloc[0])
            i = self._align_method_cb.findText(method)
            if i >= 0:
                self._align_method_cb.setCurrentIndex(i)
        if has_embedded_ref:
            ref_df = pd.DataFrame({
                'aligned_time': df['aligned_time'].to_numpy(dtype=np.float64),
                'reference_signal': df['reference_signal'].to_numpy(dtype=np.float64),
            })
            if 'reference_align_signal' in df.columns:
                ref_df['reference_align_signal'] = df['reference_align_signal'].to_numpy(dtype=np.float64)
            self._csv_df = ref_df
            self._csv_path = path
            for cb in (self._csv_t_cb, self._csv_v_cb, self._csv_align_cb):
                cb.blockSignals(True)
                cb.clear()
                cb.addItems(list(ref_df.columns))
                cb.blockSignals(False)
            self._csv_t_cb.setCurrentText('aligned_time')
            self._csv_v_cb.setCurrentText('reference_signal')
            self._csv_align_cb.setCurrentText(
                'reference_align_signal' if 'reference_align_signal' in ref_df.columns else 'reference_signal')
            self._btn_xcorr.setEnabled(True)
        self._raw_curve.setData(self._sig_t, self._sig_v)
        if len(self._sig_t) > 1:
            self._region.setRegion([self._sig_t[0], self._sig_t[-1]])
            self._trace_pw.setXRange(float(self._sig_t[0]), float(min(self._sig_t[0] + 30.0, self._sig_t[-1])), padding=0)
        self._refresh_view()
        self._set_status(f'Loaded extracted/aligned signal → {path}')

    def _batch_export(self):
        self._save_current_item_state()
        if not self._items:
            QMessageBox.warning(self, 'No file list', 'Load a video list first.')
            return
        out_dir = QFileDialog.getExistingDirectory(self, 'Batch export folder', self._last_export_dir)
        if not out_dir:
            return
        self._last_export_dir = out_dir
        self._last_dir = out_dir
        exported, skipped = 0, []
        cur = self._current_item
        for i, item in enumerate(self._items):
            self._load_item(i)
            roi = self._vid.get_roi()
            if roi is None:
                skipped.append((Path(item['video']).name, 'missing ROI'))
                continue
            ch_map = {0: 'gray', 1: 'r', 2: 'g', 3: 'b'}
            key = self._signal_cache_key(
                roi, ch_map[self._ch_cb.currentIndex()],
                self._start_spin.value(), min(self._end_spin.value(), self._n_frames))
            cached = self._load_from_cache(key)
            if cached is None and item.get('video') == self._video_path and self._sig_t is not None:
                cached = (self._sig_t, self._sig_v)
            if cached is None:
                skipped.append((Path(item['video']).name, 'not extracted/cached'))
                continue
            self._sig_t, self._sig_v = cached
            self._invalidate_smooth_cache()
            if item.get('csv'):
                self._load_csv_path(item['csv'])
            out_path = self._default_export_path(item['video'], out_dir)
            self._export_dataframe().to_csv(out_path, index=False)
            self._write_export_metadata(out_path)
            exported += 1
        if 0 <= cur < len(self._items):
            self._load_item(cur)
        msg = f'Batch exported {exported} file(s) to {out_dir}'
        if skipped:
            msg += f' | skipped {len(skipped)}: ' + ', '.join(f'{n} ({why})' for n, why in skipped[:4])
        self._save_app_settings()
        self._set_status(msg)

    # ── plot click → seek video ───────────────────────────────────────────────
    def _on_trace_click(self, event):
        if self._sig_t is None:
            return
        pos = self._trace_pw.plotItem.vb.mapSceneToView(event.scenePos())
        self._seek(int(pos.x() * self._fps))

    def _on_cmp_click(self, event):
        if self._sig_t is None:
            return
        pos = self._cmp_pw.plotItem.vb.mapSceneToView(event.scenePos())
        video_t = self._video_time_from_aligned_time(float(pos.x()))
        self._seek(int(video_t * self._fps))

    def _on_xcorr_click(self, event):
        """Click on xcorr plot → manually set offset to that lag."""
        if self._alignment_method_key() != 'cross_correlation':
            self._set_status('Manual lag click applies only to cross-correlation mode.')
            return
        pos = self._xcorr_pw.plotItem.vb.mapSceneToView(event.scenePos())
        lag_s = pos.x()
        self._xcorr_lag_line.setValue(lag_s)
        self._offset_spin.setValue(-lag_s)
        self._align_model = {
            'method': 'cross_correlation',
            'offset': float(-lag_s),
            'slope': 1.0,
            'intercept': float(lag_s),
            'video_edges': [],
            'ref_edges': [],
            'edge_pairs': 0,
            'xcorr_peak': None,
        }

    # ── helpers ───────────────────────────────────────────────────────────────
    def _set_status(self, msg: str):
        self._sb_lbl.setText(msg)
        self._log(msg)

    def _log(self, msg: str, level: str = 'info'):
        """Append a timestamped line to the log dock and tee to stderr."""
        if not hasattr(self, '_log_view'):
            return
        ts = datetime.now().strftime('%H:%M:%S')
        prefix = {'info': '', 'warn': '[WARN] ', 'error': '[ERR ] '}.get(level, '')
        self._log_view.appendPlainText(f'{ts}  {prefix}{msg}')

    # ── drag and drop ─────────────────────────────────────────────────────────
    VIDEO_EXTS = {'.mp4', '.avi', '.mkv', '.mov', '.m4v', '.wmv'}
    CSV_EXTS = {'.csv', '.txt', '.tsv'}
    PROJECT_EXTS = {'.vbep.json', '.json'}

    @classmethod
    def _is_video_path(cls, p: str) -> bool:
        return Path(p).suffix.lower() in cls.VIDEO_EXTS

    @classmethod
    def _is_csv_path(cls, p: str) -> bool:
        return Path(p).suffix.lower() in cls.CSV_EXTS

    @classmethod
    def _is_project_path(cls, p: str) -> bool:
        name = Path(p).name.lower()
        return name.endswith('.vbep.json') or name.endswith('.json')

    @classmethod
    def _dropped_paths(cls, event) -> list:
        md = event.mimeData()
        if not md.hasUrls():
            return []
        return [u.toLocalFile() for u in md.urls() if u.isLocalFile()]

    def _has_recognized_drop(self, event) -> bool:
        for p in self._dropped_paths(event):
            if (self._is_video_path(p) or self._is_csv_path(p)
                    or self._is_project_path(p)):
                return True
        return False

    def dragEnterEvent(self, e):
        if self._has_recognized_drop(e):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if self._has_recognized_drop(e):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        paths = self._dropped_paths(e)
        if not paths:
            e.ignore()
            return
        self._handle_dropped_paths(paths)
        e.acceptProposedAction()

    def _handle_dropped_paths(self, paths):
        projects = [p for p in paths if self._is_project_path(p)
                    and not self._is_csv_path(p)]
        videos   = [p for p in paths if self._is_video_path(p)]
        csvs     = [p for p in paths if self._is_csv_path(p)
                    and not self._is_project_path(p)]

        # A single dropped project file wins outright.
        if projects and not videos and not csvs:
            try:
                self._load_project_path(projects[0])
            except Exception as exc:
                self._log(f'Failed to load dropped project: {exc}',
                          level='error')
                QMessageBox.critical(self, 'Project error', str(exc))
            return

        if not videos and not csvs:
            self._set_status(
                'Drop ignored: no recognized video or CSV/TXT in the selection.')
            return

        self._log(f'Drop: {len(videos)} video(s), {len(csvs)} CSV/TXT.')

        # Videos: replace the current item list (matches Open video / Videos...).
        if videos:
            self._last_video_dir = str(Path(videos[0]).parent)
            self._last_dir = self._last_video_dir
            self._items = [{'video': v, 'csv': None, 'roi': self._default_roi}
                           for v in videos]
            self._current_item = -1
            self._load_item(0)

        # CSVs: attach by row order if multiple; attach to current if single.
        if csvs:
            self._last_csv_dir = str(Path(csvs[0]).parent)
            self._last_dir = self._last_csv_dir
            if not self._items:
                self._items = [{'video': None, 'csv': None,
                                'roi': self._default_roi}]
                self._current_item = 0
            if len(csvs) == 1:
                cur = self._current_project_item()
                if cur is not None:
                    cur['csv'] = csvs[0]
                    self._load_csv_path(csvs[0])
            else:
                for i, p in enumerate(csvs):
                    if i < len(self._items):
                        self._items[i]['csv'] = p
                cur = self._current_project_item()
                if cur and cur.get('csv'):
                    self._load_csv_path(cur['csv'])

        self._update_nav_state()
        self._save_app_settings()
        self._set_status(
            f'Dropped: {len(videos)} video(s), {len(csvs)} CSV/TXT file(s).')

    def _load_project_path(self, path: str):
        """Internal: load a project file by path. Mirrors `_load_project`
        without the file dialog so drag-and-drop can reuse it."""
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        self._items = data.get('items', [])
        self._default_roi = data.get('default_roi')
        self._last_project_dir = str(Path(path).parent)
        self._last_dir = data.get('last_dir', self._last_project_dir)
        folders = data.get('folders', {})
        self._last_video_dir = folders.get('video', self._last_video_dir)
        self._last_csv_dir = folders.get('csv', self._last_csv_dir)
        self._last_export_dir = folders.get('export', self._last_export_dir)
        idx = int(data.get('current_item', 0))
        self._current_item = -1
        if self._items:
            self._load_item(max(0, min(idx, len(self._items) - 1)))
        else:
            self._update_nav_state()
        self._save_app_settings()
        self._set_status(f'Project loaded -> {path}')

    def closeEvent(self, e):
        self._save_app_settings()
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait(1000)
        if self._cap:
            self._cap.release()
        super().closeEvent(e)


def run(argv=None) -> int:
    """Construct the QApplication, show the main window, and enter the event loop."""
    import sys as _sys
    import multiprocessing
    multiprocessing.freeze_support()
    argv = list(argv) if argv is not None else _sys.argv

    app = QApplication(argv)
    app.setApplicationName('Video Barcode Signal Extractor')
    pg.setConfigOptions(background=BG2, foreground=FG, antialias=True)
    win = MainWindow()
    win.show()
    if len(argv) > 1 and Path(argv[1]).is_file():
        cli_path = str(Path(argv[1]).resolve())
        win._items = [{'video': cli_path, 'csv': None, 'roi': win._default_roi}]
        win._current_item = 0
        win._loading_item = True
        try:
            win._load_video_path(cli_path, apply_item_state=True)
        finally:
            win._loading_item = False
        win._try_load_current_signal_from_cache()
        win._update_nav_state()
    return app.exec_()
