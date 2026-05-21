"""Nord-inspired dark palette and Qt stylesheet.

Nord is a calm, low-saturation palette designed for long sessions. The
mapping below preserves the variable names the rest of the codebase uses
(BG, BG2, BG3, OVERLAY, FG, BLUE, GREEN, ...) so app.py and pyqtgraph
pen colors do not need to change.
"""

# Polar Night (base layers)
BG       = '#242933'   # window background (one shade darker than nord0)
BG2      = '#2E3440'   # nord0  — panels, status/menu bar
BG3      = '#3B4252'   # nord1  — inputs, cards, raised surfaces
OVERLAY  = '#4C566A'   # nord3  — borders, separators
BG_HOVER = '#434C5E'   # nord2  — hover background

# Snow Storm (text)
FG       = '#ECEFF4'   # nord6  — primary text
FG_MUTED = '#D8DEE9'   # nord4  — secondary text
FG_DIM   = '#A9B2C0'   # custom — tertiary / disabled-ish

# Frost (cool accents)
BLUE     = '#88C0D0'   # nord8  — primary accent
CYAN     = '#8FBCBB'   # nord7  — secondary accent, badges
SAPPHIRE = '#81A1C1'   # nord9  — hover blue
DEEP     = '#5E81AC'   # nord10 — pressed / strong

# Aurora (warm accents)
RED      = '#BF616A'   # nord11 — threshold line, errors
PEACH    = '#D08770'   # nord12 — cursor, time markers
YELLOW   = '#EBCB8B'   # nord13 — warnings
GREEN    = '#A3BE8C'   # nord14 — binary trace
MAUVE    = '#B48EAD'   # nord15 — alignment, cross-correlation
PINK     = '#B48EAD'   # alias

FIT_MARGIN = 0.97


def stylesheet() -> str:
    return f"""
        QMainWindow, QWidget {{
            background: {BG};
            color: {FG};
            font-family: 'Segoe UI Variable Text', 'Segoe UI', sans-serif;
            font-size: 13px;
        }}

        /* ── containers ── */
        QFrame#SidePanel {{
            background: {BG2};
            border: none;
            border-right: 1px solid {OVERLAY};
        }}
        QFrame#InfoPanel {{
            background: {BG3};
            border: 1px solid {OVERLAY};
            border-radius: 10px;
        }}

        /* ── labels ── */
        QLabel {{ color: {FG}; }}
        QLabel#PanelTitle {{
            color: {FG};
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 0.2px;
        }}
        QLabel#MutedLabel {{ color: {FG_DIM}; font-size: 12px; }}
        QLabel#BadgeLabel {{
            color: {CYAN};
            background: rgba(143, 188, 187, 0.12);
            border: 1px solid rgba(143, 188, 187, 0.30);
            border-radius: 10px;
            padding: 2px 10px;
            font-weight: 600;
            font-size: 12px;
        }}
        QLabel#TimeCode {{
            color: {CYAN};
            font-family: 'JetBrains Mono', 'Cascadia Mono', Consolas, monospace;
            font-size: 13px;
        }}

        /* ── toolbar ── */
        QToolBar {{
            background: {BG2};
            border: none;
            border-bottom: 1px solid {OVERLAY};
            spacing: 8px;
            padding: 8px 10px;
        }}
        QToolBar::separator {{
            background: {OVERLAY};
            width: 1px;
            margin: 4px 6px;
        }}

        /* ── menu bar ── */
        QMenuBar {{
            background: {BG2};
            color: {FG};
            border-bottom: 1px solid {OVERLAY};
            padding: 2px 6px;
            spacing: 4px;
        }}
        QMenuBar::item {{
            background: transparent;
            padding: 6px 12px;
            border-radius: 6px;
        }}
        QMenuBar::item:selected {{ background: {BG_HOVER}; }}
        QMenu {{
            background: {BG3};
            color: {FG};
            border: 1px solid {OVERLAY};
            border-radius: 8px;
            padding: 6px;
        }}
        QMenu::item {{ padding: 7px 22px; border-radius: 4px; }}
        QMenu::item:selected {{
            background: {DEEP};
            color: {FG};
        }}
        QMenu::separator {{
            height: 1px;
            background: {OVERLAY};
            margin: 5px 8px;
        }}

        /* ── buttons ── */
        QPushButton {{
            background: {BG3};
            color: {FG};
            border: 1px solid transparent;
            border-radius: 7px;
            padding: 6px 14px;
            min-height: 24px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background: {BG_HOVER};
            border-color: {SAPPHIRE};
        }}
        QPushButton:pressed {{
            background: {DEEP};
            color: {FG};
        }}
        QPushButton:disabled {{
            color: {FG_DIM};
            background: {BG2};
            border-color: transparent;
        }}
        QPushButton:checked {{
            background: {DEEP};
            border-color: {BLUE};
            color: {FG};
        }}
        QPushButton#PrimaryButton {{
            background: {BLUE};
            color: {BG};
            border-color: {BLUE};
            font-weight: 700;
        }}
        QPushButton#PrimaryButton:hover {{
            background: {SAPPHIRE};
            border-color: {SAPPHIRE};
        }}
        QPushButton#PrimaryButton:pressed {{
            background: {DEEP};
            color: {FG};
        }}

        /* ── inputs ── */
        QLineEdit, QComboBox {{
            background: {BG3};
            border: 1px solid {OVERLAY};
            border-radius: 7px;
            padding: 5px 10px;
            selection-background-color: {DEEP};
            color: {FG};
        }}
        QLineEdit:focus, QComboBox:focus {{ border-color: {BLUE}; }}
        QComboBox::drop-down {{
            border: none;
            width: 22px;
        }}
        QComboBox QAbstractItemView {{
            background: {BG3};
            color: {FG};
            border: 1px solid {OVERLAY};
            border-radius: 6px;
            selection-background-color: {DEEP};
            padding: 4px;
            outline: 0;
        }}
        QSpinBox, QDoubleSpinBox {{
            background: {BG3};
            border: 1px solid {OVERLAY};
            border-radius: 7px;
            padding: 4px 6px;
            color: {FG};
        }}
        QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {BLUE}; }}

        /* ── file list ── */
        QListWidget#FileList {{
            background: {BG};
            border: 1px solid {OVERLAY};
            border-radius: 10px;
            padding: 6px;
            outline: 0;
        }}
        QListWidget#FileList::item {{
            border: 1px solid transparent;
            border-radius: 7px;
            padding: 9px 12px;
            margin: 3px 0;
        }}
        QListWidget#FileList::item:hover {{ background: {BG_HOVER}; }}
        QListWidget#FileList::item:selected {{
            background: {DEEP};
            border-color: {BLUE};
            color: {FG};
        }}

        /* ── sliders ── */
        QSlider::groove:horizontal {{
            background: {BG3};
            height: 6px;
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {BLUE};
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }}
        QSlider::handle:horizontal:hover {{ background: {SAPPHIRE}; }}
        QSlider::sub-page:horizontal {{
            background: {BLUE};
            border-radius: 3px;
        }}

        /* ── group boxes ── */
        QGroupBox {{
            border: 1px solid {OVERLAY};
            border-radius: 10px;
            margin-top: 14px;
            padding: 18px 12px 12px 12px;
            background: rgba(59, 66, 82, 0.30);
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 14px;
            padding: 0 6px;
            background: {BG};
            color: {CYAN};
            font-weight: 700;
            font-size: 12px;
            letter-spacing: 0.3px;
            text-transform: uppercase;
        }}

        /* ── tabs ── */
        QTabWidget::pane {{
            border: none;
            background: {BG};
            top: -1px;
        }}
        QTabBar::tab {{
            background: transparent;
            color: {FG_DIM};
            padding: 9px 22px;
            border: none;
            border-bottom: 2px solid transparent;
            font-weight: 600;
        }}
        QTabBar::tab:selected {{
            color: {BLUE};
            border-bottom-color: {BLUE};
        }}
        QTabBar::tab:hover:!selected {{
            color: {FG};
        }}

        /* ── progress ── */
        QProgressBar {{
            background: {BG3};
            border: 1px solid {OVERLAY};
            border-radius: 7px;
            text-align: center;
            color: {FG};
            font-weight: 600;
        }}
        QProgressBar::chunk {{
            background: {BLUE};
            border-radius: 6px;
        }}

        /* ── radio / checkbox ── */
        QRadioButton, QCheckBox {{
            spacing: 8px;
            color: {FG};
        }}
        QRadioButton::indicator, QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid {OVERLAY};
            background: {BG3};
        }}
        QRadioButton::indicator {{ border-radius: 8px; }}
        QCheckBox::indicator:checked {{
            background: {BLUE};
            border-color: {BLUE};
        }}
        QRadioButton::indicator:checked {{
            background: {BLUE};
            border-color: {BLUE};
        }}

        /* ── status bar ── */
        QStatusBar {{
            background: {BG2};
            border-top: 1px solid {OVERLAY};
            padding: 4px 10px;
            color: {FG_MUTED};
            font-size: 12px;
        }}
        QStatusBar::item {{ border: none; }}

        /* ── splitter ── */
        QSplitter::handle {{ background: transparent; }}
        QSplitter::handle:horizontal {{ width: 6px; }}
        QSplitter::handle:vertical   {{ height: 6px; }}
        QSplitter::handle:hover {{ background: {OVERLAY}; }}

        /* ── scrollbars ── */
        QScrollBar:vertical {{
            background: transparent;
            width: 10px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {OVERLAY};
            border-radius: 5px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {SAPPHIRE}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

        /* ── tooltip ── */
        QToolTip {{
            background: {BG3};
            color: {FG};
            border: 1px solid {OVERLAY};
            border-radius: 6px;
            padding: 6px 8px;
        }}

        /* ── dock / log ── */
        QDockWidget {{ color: {FG}; }}
        QDockWidget::title {{
            background: {BG2};
            padding: 7px 10px;
            border-bottom: 1px solid {OVERLAY};
            font-weight: 600;
        }}
        QPlainTextEdit#LogPanel {{
            background: {BG2};
            color: {FG_MUTED};
            border: 1px solid {OVERLAY};
            border-radius: 8px;
            font-family: 'JetBrains Mono', 'Cascadia Mono', Consolas, monospace;
            font-size: 12px;
            padding: 6px;
        }}
    """
