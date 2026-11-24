"""Dark and light Qt style sheets."""

_BASE = """
* {{ font-family: 'Segoe UI', 'Noto Sans', sans-serif; font-size: 10pt; }}
QMainWindow, QWidget {{ background: {bg}; color: {fg}; }}
QTabWidget::pane {{ border: 1px solid {line}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{ background: {panel}; color: {muted}; padding: 9px 18px; margin-right: 2px;
               border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:selected {{ background: {bg}; color: {fg}; border: 1px solid {line}; border-bottom: none; }}
QGroupBox {{ border: 1px solid {line}; border-radius: 8px; margin-top: 14px; padding: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {accent}; font-weight: 600; }}
QPushButton {{ background: {panel}; border: 1px solid {line}; border-radius: 6px; padding: 7px 14px; }}
QPushButton:hover {{ border-color: {accent}; }}
QPushButton:disabled {{ color: {muted}; }}
QPushButton#primary {{ background: {accent}; color: white; border: none; font-weight: 600; }}
QPushButton#primary:hover {{ background: {accent2}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
    background: {field}; border: 1px solid {line}; border-radius: 6px; padding: 5px; color: {fg}; }}
QComboBox QAbstractItemView {{ background: {field}; color: {fg}; selection-background-color: {accent}; }}
QTableWidget, QListWidget, QTreeWidget {{ background: {field}; alternate-background-color: {panel};
    border: 1px solid {line}; border-radius: 6px; gridline-color: {line}; }}
QHeaderView::section {{ background: {panel}; color: {muted}; border: none; padding: 6px; }}
QTableWidget::item:selected, QListWidget::item:selected {{ background: {accent}; color: white; }}
QProgressBar {{ background: {field}; border: 1px solid {line}; border-radius: 6px; text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 5px; }}
QToolBar {{ background: {panel}; border: none; spacing: 6px; padding: 4px; }}
QToolButton {{ padding: 6px 10px; border-radius: 6px; }}
QToolButton:hover {{ background: {line}; }}
QStatusBar {{ background: {panel}; color: {muted}; }}
QMenuBar {{ background: {panel}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {accent}; color: white; }}
QMenu {{ background: {panel}; border: 1px solid {line}; }}
QSlider::groove:horizontal {{ height: 6px; background: {line}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {accent}; width: 16px; margin: -6px 0; border-radius: 8px; }}
QLabel#title {{ font-size: 20pt; font-weight: 700; }}
QLabel#muted {{ color: {muted}; }}
QFrame#drop {{ border: 2px dashed {accent}; border-radius: 14px; background: {panel}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
"""

DARK = dict(bg="#15171c", fg="#e8eaf0", panel="#1d2027", field="#22262e", line="#323743",
            muted="#9aa1b1", accent="#6c5ce7", accent2="#8577f0")
LIGHT = dict(bg="#f6f7fb", fg="#1b1e25", panel="#ffffff", field="#ffffff", line="#d9dce5",
             muted="#5d6475", accent="#5b4bdb", accent2="#7466e6")


def stylesheet(theme: str) -> str:
    return _BASE.format(**(LIGHT if theme == "light" else DARK))
