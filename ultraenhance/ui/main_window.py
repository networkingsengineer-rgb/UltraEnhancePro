"""Main window: menus, toolbar, tabs, status bar."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSlider, QSpinBox, QSplitter, QStatusBar, QTableWidget, QTableWidgetItem, QTabWidget,
    QTextBrowser, QToolBar, QVBoxLayout, QWidget,
)

from .. import __app_name__, __version__, engines, face_restore, formats, gpu as gpu_mod
from ..config import APP_DIR, LOG_FILE, Settings, log
from ..estimates import human_bytes, human_time
from ..ffmpeg_utils import CODECS
from ..job_queue import Job, JobQueue, JobState
from ..photo import EngineContext, PhotoOptions, enhance_photo, preview_crop
from ..presets import FIT_MODES, MODELS, QUALITY_PRESETS, RESOLUTION_PRESETS, SCALE_PRESETS
from ..video import VideoOptions, discard_job, enhance_video, make_plan, preview_clip, unfinished_jobs
from .i18n import LANGUAGES, Translator
from .theme import stylesheet
from .widgets import CompareView, DropArea, pil_to_qpixmap

PHOTO_MODELS = ["Photo (realistic)", "Anime / Art", "Old photo restore"]
VIDEO_MODELS = ["Photo (realistic)", "Anime / Art", "Video (anime/cartoon, fast)"]
FIT_LABELS = {"fit": "Fit (keep everything)", "fill": "Fill (crop edges)", "pad": "Pad (add black bars)"}
ENCODER_FAMILIES = {"auto": "Auto (hardware if available)", "nvenc": "NVIDIA NVENC",
                    "amf": "AMD AMF", "qsv": "Intel Quick Sync", "software": "Software (x264/x265/SVT-AV1)"}


class Bridge(QObject):
    """Carries queue events from the worker thread to the GUI thread."""
    job_event = Signal(str, object)
    preview_ready = Signal(object, object, str)
    gpus_ready = Signal(list)
    gpu_load = Signal(object)
    probe_done = Signal(str, str, str)


def open_path(path: str | Path):
    path = str(path)
    if os.name == "nt":
        os.startfile(path)  # noqa
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def _combo(items, current=None) -> QComboBox:
    c = QComboBox()
    for it in items:
        if isinstance(it, tuple):
            c.addItem(it[1], it[0])
        else:
            c.addItem(it, it)
    if current is not None:
        i = c.findData(current)
        if i >= 0:
            c.setCurrentIndex(i)
    return c


def _set_combo(c: QComboBox, value):
    i = c.findData(value)
    if i >= 0:
        c.setCurrentIndex(i)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings.load()
        self.tr_ = Translator(self.settings.language)
        self.queue = JobQueue()
        self.bridge = Bridge()
        self.gpus: List[gpu_mod.GPUInfo] = []
        self.current_gpu: Optional[gpu_mod.GPUInfo] = None
        self.photo_opts: Dict[str, PhotoOptions] = {}
        self.video_opts: Dict[str, VideoOptions] = {}
        self.video_info: Dict[str, object] = {}
        self._last_output: Optional[str] = None
        self._gpu_load_value = None

        self.setWindowTitle(f"{__app_name__} {__version__}")
        icon = Path(__file__).resolve().parent.parent.parent / "assets" / "icon.ico"
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))
        self.resize(1360, 860)
        self.setAcceptDrops(True)

        self._build_actions()
        self._build_menu()
        self._build_toolbar()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._home_tab(), self.tr_("Home"))
        self.tabs.addTab(self._photo_tab(), self.tr_("Photo Enhance"))
        self.tabs.addTab(self._video_tab(), self.tr_("Video Enhance"))
        self.tabs.addTab(self._queue_tab(), self.tr_("Batch Queue"))
        self.tabs.addTab(self._settings_tab(), self.tr_("Settings"))
        self.tabs.addTab(self._help_tab(), self.tr_("Help"))
        self._build_status()

        self.bridge.job_event.connect(self._on_job_event)
        self.bridge.preview_ready.connect(self._on_preview_ready)
        self.bridge.gpus_ready.connect(self._on_gpus)
        self.bridge.probe_done.connect(self._set_video_cells)
        self.bridge.gpu_load.connect(lambda v: setattr(self, "_gpu_load_value", v))
        for ev in ("added", "started", "progress", "finished", "changed"):
            self.queue.on(ev, lambda job, ev=ev: self.bridge.job_event.emit(ev, job))

        self.apply_theme(self.settings.theme)
        threading.Thread(target=lambda: self.bridge.gpus_ready.emit(gpu_mod.detect_gpus()), daemon=True).start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1500)
        self._load_thread_running = False
        QTimer.singleShot(400, self._check_engines)

    # ================================================================ actions/menus
    def _act(self, text, slot, shortcut=None, tip=None) -> QAction:
        a = QAction(self.tr_(text), self)
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setStatusTip(tip)
        return a

    def _build_actions(self):
        self.a_add = self._act("Add Files", self.add_files_dialog, "Ctrl+O", "Add photos or videos")
        self.a_add_folder = self._act("Add Folder...", self.add_folder_dialog, "Ctrl+Shift+O")
        self.a_start = self._act("Start", self.start, "F5", "Start processing")
        self.a_pause = self._act("Pause", self.pause_or_resume, "F6", "Pause or resume")
        self.a_stop = self._act("Stop", self.stop, "F8", "Stop all jobs")
        self.a_open_out = self._act("Open Output Folder", self.open_output, "Ctrl+E")
        self.a_exit = self._act("Exit", self.close, "Alt+F4")
        self.a_select_all = self._act("Select all", self.select_all, "Ctrl+A")
        self.a_remove = self._act("Remove selected", self.remove_selected, "Delete")
        self.a_clear = self._act("Clear list", self.clear_list)
        self.a_apply_sel = self._act("Apply settings to selected", self.apply_to_selected, "Ctrl+Shift+A")
        self.a_apply_all = self._act("Apply settings to all", self.apply_to_all)
        self.a_preview = self._act("Preview", self.preview_current, "Ctrl+P")
        self.a_resume = self._act("Resume", lambda: self.queue.resume())
        self.a_detect = self._act("Detect GPU again", self.redetect_gpu)
        self.a_log = self._act("Open log file", lambda: open_path(LOG_FILE) if LOG_FILE.exists() else None)
        self.a_clean = self._act("Clean temporary files", self.clean_temp)
        self.a_engines = self._act("Check AI engines", lambda: self._check_engines(show=True))
        self.a_dark = self._act("Dark theme", lambda: self.apply_theme("dark", save=True))
        self.a_light = self._act("Light theme", lambda: self.apply_theme("light", save=True))
        self.a_help = self._act("Help", lambda: self.tabs.setCurrentIndex(5), "F1")
        self.a_about = self._act("About", self.about)
        self.a_licences = self._act("Open-source licences", self.show_licences)

    def _build_menu(self):
        mb = self.menuBar()
        m = mb.addMenu(self.tr_("File"))
        for a in (self.a_add, self.a_add_folder, self.a_open_out):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.a_exit)
        m = mb.addMenu(self.tr_("Edit"))
        for a in (self.a_select_all, self.a_remove, self.a_clear):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.a_apply_sel)
        m.addAction(self.a_apply_all)
        m = mb.addMenu(self.tr_("Process"))
        for a in (self.a_start, self.a_pause, self.a_resume, self.a_stop, self.a_preview):
            m.addAction(a)
        m = mb.addMenu(self.tr_("Tools"))
        for a in (self.a_detect, self.a_engines, self.a_clean, self.a_log):
            m.addAction(a)
        m = mb.addMenu(self.tr_("View"))
        m.addAction(self.a_dark)
        m.addAction(self.a_light)
        m.addSeparator()
        for i, name in enumerate(("Home", "Photo Enhance", "Video Enhance", "Batch Queue", "Settings", "Help")):
            m.addAction(self._act(name, lambda _=False, i=i: self.tabs.setCurrentIndex(i), f"Ctrl+{i + 1}"))
        m = mb.addMenu(self.tr_("Help"))
        for a in (self.a_help, self.a_licences, self.a_about):
            m.addAction(a)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)
        for a in (self.a_add, self.a_start, self.a_pause, self.a_stop, self.a_open_out):
            tb.addAction(a)

    def _build_status(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.s_gpu = QLabel("GPU: detecting...")
        self.s_load = QLabel("CPU --% | GPU --%")
        self.s_job = QLabel("Idle")
        self.s_eta = QLabel("")
        for w in (self.s_gpu, self.s_load, self.s_job):
            sb.addWidget(w)
            sb.addWidget(QLabel("|"))
        sb.addPermanentWidget(self.s_eta)

    def apply_theme(self, theme: str, save: bool = False):
        QApplication.instance().setStyleSheet(stylesheet(theme))
        if save:
            self.settings.theme = theme
            self.settings.save()
            if hasattr(self, "set_theme"):
                _set_combo(self.set_theme, theme)

    # ================================================================ Home tab
    def _home_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 20, 24, 20)
        t = QLabel(__app_name__)
        t.setObjectName("title")
        lay.addWidget(t)
        sub = QLabel("Free AI upscaling for photos and videos: 720p, 1080p, 2K, 4K and 8K. "
                     "Drop files below or use Add Files.")
        sub.setObjectName("muted")
        sub.setWordWrap(True)
        lay.addWidget(sub)
        drop = DropArea(self.tr_("Drop photos or videos here") + "\n\nor click to choose files\n\n"
                        "Photos: JPG, PNG, WEBP, BMP, TIFF, HEIC and more\n"
                        "Videos: MP4, MKV, MOV, AVI, WEBM, WMV, FLV and more")
        drop.setMinimumHeight(220)
        drop.dropped.connect(self.add_paths)
        drop.clicked.connect(self.add_files_dialog)
        lay.addWidget(drop)
        row = QHBoxLayout()
        for text, idx in (("Enhance photos", 1), ("Enhance videos", 2), ("Open batch queue", 3)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, i=idx: self.tabs.setCurrentIndex(i))
            row.addWidget(b)
        lay.addLayout(row)
        self.resume_box = QGroupBox("Unfinished video jobs (resume from the last finished chunk)")
        rl = QVBoxLayout(self.resume_box)
        self.resume_table = QTableWidget(0, 3)
        self.resume_table.setHorizontalHeaderLabels(["Video", "Stage", "Chunks done"])
        self.resume_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.resume_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.resume_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        rl.addWidget(self.resume_table)
        br = QHBoxLayout()
        b1, b2 = QPushButton("Resume selected"), QPushButton("Discard selected")
        b1.setObjectName("primary")
        b1.clicked.connect(self.resume_unfinished)
        b2.clicked.connect(self.discard_unfinished)
        br.addWidget(b1)
        br.addWidget(b2)
        br.addStretch()
        rl.addLayout(br)
        lay.addWidget(self.resume_box)
        self.engine_label = QLabel("")
        self.engine_label.setObjectName("muted")
        self.engine_label.setWordWrap(True)
        lay.addWidget(self.engine_label)
        lay.addStretch()
        self._refresh_unfinished()
        return w

    def _refresh_unfinished(self):
        jobs = unfinished_jobs()
        self._unfinished = jobs
        self.resume_table.setRowCount(len(jobs))
        for r, j in enumerate(jobs):
            self.resume_table.setItem(r, 0, QTableWidgetItem(j.get("source", "")))
            self.resume_table.setItem(r, 1, QTableWidgetItem(j.get("stage", "")))
            self.resume_table.setItem(r, 2, QTableWidgetItem(str(len(j.get("done_chunks", [])))))
        self.resume_box.setVisible(bool(jobs))

    def resume_unfinished(self):
        rows = sorted({i.row() for i in self.resume_table.selectedIndexes()}) or list(range(len(self._unfinished)))
        for r in rows:
            j = self._unfinished[r]
            opts = VideoOptions(**{k: v for k, v in j.get("options", {}).items()
                                   if k in VideoOptions.__dataclass_fields__})
            if not opts.output_dir:
                opts.output_dir = str(self.settings.resolved_output_dir())
            self._enqueue_video(j["source"], opts)
        self.tabs.setCurrentIndex(3)

    def discard_unfinished(self):
        rows = sorted({i.row() for i in self.resume_table.selectedIndexes()})
        for r in rows:
            discard_job(self._unfinished[r]["id"])
        self._refresh_unfinished()

    # ================================================================ Photo tab
    def _photo_tab(self) -> QWidget:
        w = QWidget()
        outer = QHBoxLayout(w)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)

        left = QWidget()
        ll = QVBoxLayout(left)
        drop = DropArea("Drop photos or folders here, or click to add")
        drop.setMinimumHeight(80)
        drop.dropped.connect(lambda p: self.add_paths(p, "photo"))
        drop.clicked.connect(lambda: self.add_files_dialog("photo"))
        ll.addWidget(drop)
        self.photo_table = QTableWidget(0, 3)
        self.photo_table.setHorizontalHeaderLabels(["Photo", "Size", "Settings"])
        self.photo_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.photo_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.photo_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.photo_table.itemSelectionChanged.connect(self._photo_selected)
        ll.addWidget(self.photo_table)
        self.compare = CompareView()
        ll.addWidget(self.compare, 2)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        g = QGroupBox("Output size")
        f = QFormLayout(g)
        self.p_preset = _combo([(k, k) for k in RESOLUTION_PRESETS] + [(f"scale:{k}", f"Scale {k}") for k in SCALE_PRESETS],
                               self.settings.default_preset)
        self.p_fit = _combo([(k, FIT_LABELS[k]) for k in FIT_MODES])
        f.addRow("Preset", self.p_preset)
        f.addRow("Aspect ratio", self.p_fit)
        rl.addWidget(g)
        g = QGroupBox("AI and clean-up")
        f = QFormLayout(g)
        self.p_model = _combo(PHOTO_MODELS)
        f.addRow("AI model", self.p_model)
        self.p_face = QCheckBox("Face restore (GFPGAN)")
        self.p_face_strength = QSlider(Qt.Horizontal)
        self.p_face_strength.setRange(10, 100)
        self.p_face_strength.setValue(80)
        self.p_denoise = QCheckBox("Denoise")
        self.p_sharpen = QCheckBox("Sharpen")
        self.p_autofix = QCheckBox("Auto-fix colour, brightness and contrast")
        self.p_jpeg = QCheckBox("Remove JPEG artefacts")
        for cb in (self.p_face,):
            f.addRow(cb)
        f.addRow("Face strength", self.p_face_strength)
        for cb in (self.p_denoise, self.p_sharpen, self.p_autofix, self.p_jpeg):
            f.addRow(cb)
        rl.addWidget(g)
        g = QGroupBox("Save as")
        f = QFormLayout(g)
        self.p_format = _combo(list(formats.PHOTO_OUTPUT_FORMATS))
        self.p_quality = QSlider(Qt.Horizontal)
        self.p_quality.setRange(50, 100)
        self.p_quality.setValue(self.settings.jpg_quality)
        self.p_quality_lbl = QLabel(str(self.settings.jpg_quality))
        self.p_quality.valueChanged.connect(lambda v: self.p_quality_lbl.setText(str(v)))
        qrow = QHBoxLayout()
        qrow.addWidget(self.p_quality)
        qrow.addWidget(self.p_quality_lbl)
        self.p_exif = QCheckBox("Keep EXIF data (camera, date, GPS)")
        self.p_exif.setChecked(self.settings.keep_exif)
        f.addRow("Format", self.p_format)
        f.addRow("Quality (JPG/WEBP)", qrow)
        f.addRow(self.p_exif)
        rl.addWidget(g)
        self.p_info = QLabel("")
        self.p_info.setObjectName("muted")
        self.p_info.setWordWrap(True)
        rl.addWidget(self.p_info)
        b_prev = QPushButton(self.tr_("Preview"))
        b_prev.clicked.connect(self.preview_photo)
        b_apply = QPushButton(self.tr_("Apply settings to selected"))
        b_apply.clicked.connect(self.apply_to_selected)
        b_all = QPushButton(self.tr_("Apply settings to all"))
        b_all.clicked.connect(self.apply_to_all)
        b_go = QPushButton("Enhance photos (add to queue)")
        b_go.setObjectName("primary")
        b_go.clicked.connect(self.enqueue_photos)
        for b in (b_prev, b_apply, b_all, b_go):
            rl.addWidget(b)
        rl.addStretch()
        right.setMaximumWidth(420)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        return w

    def photo_form(self) -> PhotoOptions:
        preset = self.p_preset.currentData()
        scale = None
        if preset.startswith("scale:"):
            scale, preset = SCALE_PRESETS[preset.split(":", 1)[1]], None
        return PhotoOptions(
            preset=preset, scale=scale, fit_mode=self.p_fit.currentData(), model=self.p_model.currentData(),
            face_restore=self.p_face.isChecked(), face_strength=self.p_face_strength.value() / 100,
            denoise=self.p_denoise.isChecked(), sharpen=self.p_sharpen.isChecked(),
            auto_fix=self.p_autofix.isChecked(), remove_jpeg_artifacts=self.p_jpeg.isChecked(),
            out_format=self.p_format.currentData(), jpg_quality=self.p_quality.value(),
            keep_exif=self.p_exif.isChecked(), output_dir=str(self.settings.resolved_output_dir()))

    def load_photo_form(self, o: PhotoOptions):
        _set_combo(self.p_preset, o.preset if o.scale is None else f"scale:{o.scale}x")
        _set_combo(self.p_fit, o.fit_mode)
        _set_combo(self.p_model, o.model)
        self.p_face.setChecked(o.face_restore)
        self.p_face_strength.setValue(int(o.face_strength * 100))
        self.p_denoise.setChecked(o.denoise)
        self.p_sharpen.setChecked(o.sharpen)
        self.p_autofix.setChecked(o.auto_fix)
        self.p_jpeg.setChecked(o.remove_jpeg_artifacts)
        _set_combo(self.p_format, o.out_format)
        self.p_quality.setValue(o.jpg_quality)
        self.p_exif.setChecked(o.keep_exif)

    @staticmethod
    def _photo_summary(o: PhotoOptions) -> str:
        size = o.preset.split(" ")[0] if o.scale is None else f"{o.scale}x"
        extras = [n for n, on in (("faces", o.face_restore), ("denoise", o.denoise), ("sharpen", o.sharpen),
                                  ("auto-fix", o.auto_fix), ("de-JPEG", o.remove_jpeg_artifacts)) if on]
        return f"{size}, {o.fit_mode}, {o.model.split(' ')[0]}, {o.out_format}" + (f" + {', '.join(extras)}" if extras else "")

    def _photo_selected(self):
        rows = self._selected_rows(self.photo_table)
        if len(rows) == 1:
            path = self.photo_table.item(rows[0], 0).data(Qt.UserRole)
            self.load_photo_form(self.photo_opts[path])
            self.p_info.setText(path)

    # ================================================================ Video tab
    def _video_tab(self) -> QWidget:
        w = QWidget()
        outer = QHBoxLayout(w)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)
        left = QWidget()
        ll = QVBoxLayout(left)
        drop = DropArea("Drop videos or folders here, or click to add")
        drop.setMinimumHeight(80)
        drop.dropped.connect(lambda p: self.add_paths(p, "video"))
        drop.clicked.connect(lambda: self.add_files_dialog("video"))
        ll.addWidget(drop)
        self.video_table = QTableWidget(0, 4)
        self.video_table.setHorizontalHeaderLabels(["Video", "Source", "Duration", "Export settings"])
        self.video_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.video_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.video_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.video_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.video_table.itemSelectionChanged.connect(self._video_selected)
        ll.addWidget(self.video_table, 2)
        self.v_plan = QTextBrowser()
        self.v_plan.setPlaceholderText("Select a video and press Estimate to see the output size, disk space and time.")
        ll.addWidget(self.v_plan, 1)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        g = QGroupBox("Export settings")
        f = QFormLayout(g)
        self.v_preset = _combo(list(RESOLUTION_PRESETS), self.settings.default_preset
                               if self.settings.default_preset in RESOLUTION_PRESETS else "1080p Full HD")
        self.v_fit = _combo([(k, FIT_LABELS[k]) for k in FIT_MODES])
        self.v_model = _combo(VIDEO_MODELS)
        self.v_interp = QCheckBox("Smooth motion (frame interpolation)")
        self.v_fps = _combo([(24, "24 fps"), (30, "30 fps"), (48, "48 fps"), (50, "50 fps"), (60, "60 fps"),
                             (120, "120 fps")], 60)
        self.v_deint = _combo([("auto", "Auto"), ("on", "On"), ("off", "Off")])
        self.v_denoise = QCheckBox("Denoise")
        self.v_stab = QCheckBox("Stabilise shaky video (vidstab)")
        self.v_codec = _combo(list(CODECS), self.settings.video_codec)
        self.v_encoder = _combo(list(ENCODER_FAMILIES.items()), self.settings.video_encoder)
        self.v_quality = _combo(list(QUALITY_PRESETS), self.settings.video_quality)
        self.v_rate_mode = _combo([("crf", "Constant quality (CRF)"), ("bitrate", "Target bitrate")])
        self.v_crf = QSpinBox()
        self.v_crf.setRange(0, 51)
        self.v_crf.setSpecialValueText("Auto")
        self.v_crf.setValue(0)
        self.v_bitrate = QSpinBox()
        self.v_bitrate.setRange(500, 400000)
        self.v_bitrate.setSuffix(" kbps")
        self.v_bitrate.setValue(20000)
        self.v_bitrate.setEnabled(False)
        self.v_rate_mode.currentIndexChanged.connect(
            lambda: (self.v_crf.setEnabled(self.v_rate_mode.currentData() == "crf"),
                     self.v_bitrate.setEnabled(self.v_rate_mode.currentData() == "bitrate")))
        self.v_container = _combo(list(formats.VIDEO_CONTAINERS), self.settings.video_container)
        f.addRow("Resolution", self.v_preset)
        f.addRow("Aspect ratio", self.v_fit)
        f.addRow("AI model", self.v_model)
        f.addRow(self.v_interp)
        f.addRow("Target frame rate", self.v_fps)
        f.addRow("Deinterlace", self.v_deint)
        f.addRow(self.v_denoise)
        f.addRow(self.v_stab)
        f.addRow("Codec", self.v_codec)
        f.addRow("Encoder", self.v_encoder)
        f.addRow("Quality preset", self.v_quality)
        f.addRow("Rate control", self.v_rate_mode)
        f.addRow("CRF (0 = auto)", self.v_crf)
        f.addRow("Bitrate", self.v_bitrate)
        f.addRow("Container", self.v_container)
        rl.addWidget(g)
        hint = QLabel("Select one or more videos, change the settings, then press "
                      "'Apply settings to selected'. Each video keeps its own export settings.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        rl.addWidget(hint)
        grid = QGridLayout()
        buttons = [
            (self.tr_("Apply settings to selected"), self.apply_to_selected),
            (self.tr_("Apply settings to all"), self.apply_to_all),
            (self.tr_("Estimate"), self.estimate_video),
            (self.tr_("Preview 5 seconds"), self.preview_video),
        ]
        for i, (t, fn) in enumerate(buttons):
            b = QPushButton(t)
            b.clicked.connect(fn)
            grid.addWidget(b, i // 2, i % 2)
        rl.addLayout(grid)
        b_go = QPushButton("Enhance videos (add to queue)")
        b_go.setObjectName("primary")
        b_go.clicked.connect(self.enqueue_videos)
        rl.addWidget(b_go)
        rl.addStretch()
        right.setMaximumWidth(440)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        return w

    def video_form(self) -> VideoOptions:
        mode = self.v_rate_mode.currentData()
        return VideoOptions(
            preset=self.v_preset.currentData(), fit_mode=self.v_fit.currentData(),
            model=self.v_model.currentData(), interpolate=self.v_interp.isChecked(),
            target_fps=int(self.v_fps.currentData()), deinterlace=self.v_deint.currentData(),
            denoise=self.v_denoise.isChecked(), stabilize=self.v_stab.isChecked(),
            codec=self.v_codec.currentData(), encoder_family=self.v_encoder.currentData(),
            quality=self.v_quality.currentData(),
            crf=(self.v_crf.value() or None) if mode == "crf" else None,
            bitrate_kbps=self.v_bitrate.value() if mode == "bitrate" else None,
            container=self.v_container.currentData(), output_dir=str(self.settings.resolved_output_dir()))

    def load_video_form(self, o: VideoOptions):
        _set_combo(self.v_preset, o.preset)
        _set_combo(self.v_fit, o.fit_mode)
        _set_combo(self.v_model, o.model)
        self.v_interp.setChecked(o.interpolate)
        _set_combo(self.v_fps, o.target_fps)
        _set_combo(self.v_deint, o.deinterlace)
        self.v_denoise.setChecked(o.denoise)
        self.v_stab.setChecked(o.stabilize)
        _set_combo(self.v_codec, o.codec)
        _set_combo(self.v_encoder, o.encoder_family)
        _set_combo(self.v_quality, o.quality)
        _set_combo(self.v_rate_mode, "bitrate" if o.bitrate_kbps else "crf")
        self.v_crf.setValue(o.crf or 0)
        if o.bitrate_kbps:
            self.v_bitrate.setValue(o.bitrate_kbps)
        _set_combo(self.v_container, o.container)

    @staticmethod
    def _video_summary(o: VideoOptions) -> str:
        parts = [o.preset.split(" ")[0], o.codec, o.container, o.quality]
        if o.interpolate:
            parts.append(f"{o.target_fps} fps")
        if o.stabilize:
            parts.append("stabilised")
        if o.denoise:
            parts.append("denoise")
        if o.bitrate_kbps:
            parts.append(f"{o.bitrate_kbps} kbps")
        elif o.crf:
            parts.append(f"CRF {o.crf}")
        return ", ".join(parts)

    def _video_selected(self):
        rows = self._selected_rows(self.video_table)
        if len(rows) == 1:
            path = self.video_table.item(rows[0], 0).data(Qt.UserRole)
            self.load_video_form(self.video_opts[path])

    # ================================================================ Queue tab
    def _queue_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.q_table = QTableWidget(0, 6)
        self.q_table.setHorizontalHeaderLabels(["#", "File", "Type", "Status", "Progress", "Details / ETA"])
        hh = self.q_table.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(5, QHeaderView.Stretch)
        self.q_table.setColumnWidth(4, 180)
        self.q_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.q_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.q_table.doubleClicked.connect(self._open_job_result)
        lay.addWidget(self.q_table)
        row = QHBoxLayout()
        specs = [("Pause selected", lambda: self._for_selected_jobs(self.queue.pause)),
                 ("Resume selected", lambda: self._for_selected_jobs(self.queue.resume)),
                 ("Cancel selected", lambda: self._for_selected_jobs(self.queue.cancel)),
                 ("Retry / resume failed", lambda: self._for_selected_jobs(self.queue.retry)),
                 ("Move up", lambda: self._for_selected_jobs(lambda j: self.queue.move(j, -1))),
                 ("Move down", lambda: self._for_selected_jobs(lambda j: self.queue.move(j, 1))),
                 ("Remove finished", self.queue.remove_finished),
                 ("Pause all", lambda: self.queue.pause()), ("Resume all", lambda: self.queue.resume()),
                 ("Stop all", self.stop)]
        for t, fn in specs:
            b = QPushButton(t)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.q_total = QProgressBar()
        lay.addWidget(self.q_total)
        return w

    def _job_rows(self) -> List[Job]:
        return list(self.queue.jobs)

    def _refresh_queue(self):
        jobs = self._job_rows()
        self.q_table.setRowCount(len(jobs))
        total = 0.0
        for r, j in enumerate(jobs):
            vals = [str(j.id), j.title, j.kind.capitalize(), j.state.value]
            for c, v in enumerate(vals):
                it = self.q_table.item(r, c)
                if it is None:
                    it = QTableWidgetItem()
                    self.q_table.setItem(r, c, it)
                it.setText(v)
                it.setData(Qt.UserRole, j.id)
            bar = self.q_table.cellWidget(r, 4)
            if not isinstance(bar, QProgressBar):
                bar = QProgressBar()
                bar.setRange(0, 1000)
                self.q_table.setCellWidget(r, 4, bar)
            bar.setValue(int(j.progress * 1000))
            bar.setFormat(f"{j.progress * 100:.1f}%")
            detail = j.error or j.message
            if j.eta and j.state == JobState.RUNNING:
                detail += f"  (about {human_time(j.eta)} left)"
            it = self.q_table.item(r, 5)
            if it is None:
                it = QTableWidgetItem()
                self.q_table.setItem(r, 5, it)
            it.setText(detail)
            it.setToolTip(detail)
            total += j.progress
        self.q_total.setValue(int(100 * total / len(jobs)) if jobs else 0)

    def _for_selected_jobs(self, fn):
        ids = {self.q_table.item(r, 0).data(Qt.UserRole) for r in self._selected_rows(self.q_table)}
        for j in list(self.queue.jobs):
            if j.id in ids:
                fn(j)

    def _open_job_result(self, index):
        job_id = self.q_table.item(index.row(), 0).data(Qt.UserRole)
        j = self.queue.get(job_id)
        if j and j.state == JobState.DONE and j.result:
            open_path(j.result)

    # ================================================================ Settings tab
    def _settings_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("Processing")
        f = QFormLayout(g)
        self.set_gpu = QComboBox()
        self.set_gpu.addItem("Auto (best GPU)", -2)
        self.set_gpu.addItem("CPU only (slow)", -1)
        self.set_tile = QSpinBox()
        self.set_tile.setRange(0, 1024)
        self.set_tile.setSingleStep(32)
        self.set_tile.setSpecialValueText("Auto (from VRAM)")
        self.set_tile.setValue(self.settings.tile_size)
        self.tile_hint = QLabel("")
        self.tile_hint.setObjectName("muted")
        self.set_threads = QLineEdit(str(self.settings.threads))
        self.set_threads.setToolTip("Real-ESRGAN load:process:save threads, e.g. 1:2:2. Raise the middle "
                                    "number on fast GPUs, lower it if you see errors.")
        f.addRow("GPU", self.set_gpu)
        f.addRow("Tile size", self.set_tile)
        f.addRow("", self.tile_hint)
        f.addRow("Threads", self.set_threads)
        lay.addWidget(g)
        g = QGroupBox("Folders")
        f = QFormLayout(g)
        self.set_temp = QLineEdit(str(self.settings.resolved_temp_dir()))
        self.set_out = QLineEdit(str(self.settings.resolved_output_dir()))
        for label, edit in (("Temp folder", self.set_temp), ("Output folder", self.set_out)):
            row = QHBoxLayout()
            row.addWidget(edit)
            b = QPushButton("Browse...")
            b.clicked.connect(lambda _=False, e=edit: self._browse_dir(e))
            row.addWidget(b)
            f.addRow(label, row)
        lay.addWidget(g)
        g = QGroupBox("Defaults and appearance")
        f = QFormLayout(g)
        self.set_preset = _combo(list(RESOLUTION_PRESETS), self.settings.default_preset)
        self.set_lang = _combo(list(LANGUAGES), self.settings.language)
        self.set_theme = _combo([("dark", "Dark"), ("light", "Light")], self.settings.theme)
        self.set_theme.currentIndexChanged.connect(lambda: self.apply_theme(self.set_theme.currentData()))
        f.addRow("Default preset", self.set_preset)
        f.addRow("Language (restart to apply)", self.set_lang)
        f.addRow("Theme", self.set_theme)
        lay.addWidget(g)
        b = QPushButton(self.tr_("Save settings"))
        b.setObjectName("primary")
        b.clicked.connect(self.save_settings)
        lay.addWidget(b)
        lay.addStretch()
        return w

    def _browse_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Choose folder", edit.text())
        if d:
            edit.setText(d)

    def save_settings(self):
        s = self.settings
        s.gpu_id = int(self.set_gpu.currentData())
        s.tile_size = self.set_tile.value()
        threads = self.set_threads.text().strip()
        if not all(p.isdigit() for p in threads.split(":")) or len(threads.split(":")) != 3:
            QMessageBox.warning(self, "Settings", "Threads must look like 1:2:2 (three numbers).")
            return
        s.threads = threads
        for edit, attr in ((self.set_temp, "temp_dir"), (self.set_out, "output_dir")):
            try:
                Path(edit.text()).mkdir(parents=True, exist_ok=True)
                setattr(s, attr, edit.text())
            except OSError as e:
                QMessageBox.warning(self, "Settings", f"That folder can't be used: {e}")
                return
        s.default_preset = self.set_preset.currentData()
        s.language = self.set_lang.currentData()
        s.theme = self.set_theme.currentData()
        s.jpg_quality = self.p_quality.value()
        s.keep_exif = self.p_exif.isChecked()
        s.video_codec = self.v_codec.currentData()
        s.video_encoder = self.v_encoder.currentData()
        s.video_quality = self.v_quality.currentData()
        s.video_container = self.v_container.currentData()
        s.save()
        self._select_gpu()
        self.statusBar().showMessage("Settings saved", 4000)

    # ================================================================ Help tab
    def _help_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        tb = QTextBrowser()
        tb.setOpenExternalLinks(True)
        tb.setHtml(HELP_HTML.format(log=LOG_FILE, app=__app_name__, ver=__version__))
        lay.addWidget(tb)
        return w

    # ================================================================ file adding
    def add_files_dialog(self, kind: str = "all"):
        files, _ = QFileDialog.getOpenFileNames(self, "Add photos or videos", "", formats.dialog_filter(kind))
        if files:
            self.add_paths(files, kind)

    def add_folder_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Add a folder of photos/videos")
        if d:
            self.add_paths([d])

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    def add_paths(self, paths, kind: str = "all"):
        files = formats.expand_paths(paths, kind)
        if not files:
            QMessageBox.information(self, __app_name__, "No supported photos or videos were found there.")
            return
        n_p = n_v = 0
        for f in files:
            if formats.is_photo(f) and f not in self.photo_opts:
                self.photo_opts[f] = self.photo_form()
                self._add_photo_row(f)
                n_p += 1
            elif formats.is_video(f) and f not in self.video_opts:
                self.video_opts[f] = self.video_form()
                self._add_video_row(f)
                n_v += 1
        if n_v and not n_p:
            self.tabs.setCurrentIndex(2)
        elif n_p:
            self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage(f"Added {n_p} photo(s) and {n_v} video(s)", 5000)

    def _add_photo_row(self, path: str):
        r = self.photo_table.rowCount()
        self.photo_table.insertRow(r)
        it = QTableWidgetItem(Path(path).name)
        it.setData(Qt.UserRole, path)
        it.setToolTip(path)
        self.photo_table.setItem(r, 0, it)
        size = "?"
        try:
            from PIL import Image
            with Image.open(path) as im:
                size = f"{im.width}x{im.height}"
        except Exception:
            pass
        self.photo_table.setItem(r, 1, QTableWidgetItem(size))
        self.photo_table.setItem(r, 2, QTableWidgetItem(self._photo_summary(self.photo_opts[path])))

    def _add_video_row(self, path: str):
        r = self.video_table.rowCount()
        self.video_table.insertRow(r)
        it = QTableWidgetItem(Path(path).name)
        it.setData(Qt.UserRole, path)
        it.setToolTip(path)
        self.video_table.setItem(r, 0, it)
        self.video_table.setItem(r, 1, QTableWidgetItem("reading..."))
        self.video_table.setItem(r, 2, QTableWidgetItem(""))
        self.video_table.setItem(r, 3, QTableWidgetItem(self._video_summary(self.video_opts[path])))

        def probe():
            try:
                from ..ffmpeg_utils import probe as fprobe
                info = fprobe(str(engines.ffprobe_path()), path)
                self.video_info[path] = info
                text = f"{info.width}x{info.height} @ {info.fps_float:.2f} fps"
                dur = human_time(info.duration)
            except Exception as e:
                text, dur = "unreadable", str(e)[:60]
            self.bridge.probe_done.emit(path, text, dur)
        threading.Thread(target=probe, daemon=True).start()

    def _set_video_cells(self, path, text, dur):
        for r in range(self.video_table.rowCount()):
            if self.video_table.item(r, 0).data(Qt.UserRole) == path:
                self.video_table.item(r, 1).setText(text)
                self.video_table.item(r, 2).setText(dur)

    @staticmethod
    def _selected_rows(table: QTableWidget) -> List[int]:
        return sorted({i.row() for i in table.selectedIndexes()})

    def _active_table(self):
        i = self.tabs.currentIndex()
        if i == 1:
            return self.photo_table, "photo"
        if i == 2:
            return self.video_table, "video"
        return None, None

    def select_all(self):
        t, _ = self._active_table()
        if t:
            t.selectAll()
        elif self.tabs.currentIndex() == 3:
            self.q_table.selectAll()

    def remove_selected(self):
        t, kind = self._active_table()
        if not t:
            return
        for r in reversed(self._selected_rows(t)):
            path = t.item(r, 0).data(Qt.UserRole)
            (self.photo_opts if kind == "photo" else self.video_opts).pop(path, None)
            t.removeRow(r)

    def clear_list(self):
        t, kind = self._active_table()
        if not t:
            return
        t.setRowCount(0)
        (self.photo_opts if kind == "photo" else self.video_opts).clear()

    # ================================================================ apply settings
    def apply_to_selected(self, all_rows: bool = False):
        t, kind = self._active_table()
        if not t:
            QMessageBox.information(self, __app_name__, "Open the Photo or Video tab first.")
            return
        rows = list(range(t.rowCount())) if all_rows else self._selected_rows(t)
        if not rows:
            QMessageBox.information(self, __app_name__, "Select one or more files in the list first "
                                    "(Ctrl+click or Shift+click to select several).")
            return
        for r in rows:
            path = t.item(r, 0).data(Qt.UserRole)
            if kind == "photo":
                self.photo_opts[path] = self.photo_form()
                t.item(r, 2).setText(self._photo_summary(self.photo_opts[path]))
            else:
                self.video_opts[path] = self.video_form()
                t.item(r, 3).setText(self._video_summary(self.video_opts[path]))
        self.statusBar().showMessage(f"Export settings applied to {len(rows)} file(s)", 5000)

    def apply_to_all(self):
        self.apply_to_selected(all_rows=True)

    # ================================================================ engine context
    def engine_ctx(self) -> EngineContext:
        g = self.current_gpu
        return EngineContext(
            gpu_id=g.index if g else -2, vram_mb=g.vram_mb if g else 0,
            cpu=self.settings.gpu_id == -1, tile_override=self.settings.tile_size,
            threads=self.settings.threads, temp_dir=self.settings.resolved_temp_dir())

    def _on_gpus(self, gpus):
        self.gpus = gpus
        cur = self.settings.gpu_id
        while self.set_gpu.count() > 2:
            self.set_gpu.removeItem(2)
        for g in gpus:
            self.set_gpu.addItem(g.label, g.index)
        _set_combo(self.set_gpu, cur)
        self._select_gpu()

    def _select_gpu(self):
        self.current_gpu = gpu_mod.pick_gpu(self.gpus, self.settings.gpu_id)
        if self.settings.gpu_id == -1:
            self.s_gpu.setText("CPU mode (no GPU)")
        elif self.current_gpu:
            g = self.current_gpu
            self.s_gpu.setText(f"GPU: {g.name} | VRAM {g.vram_mb / 1024:.1f} GB" if g.vram_mb else f"GPU: {g.name}")
        else:
            self.s_gpu.setText("GPU: none found (CPU fallback)")
        ctx = self.engine_ctx()
        self.tile_hint.setText(f"Auto tile for your card: {ctx.tile(4)} px (lower it if you get memory errors)")

    def redetect_gpu(self):
        self.s_gpu.setText("GPU: detecting...")
        threading.Thread(target=lambda: self.bridge.gpus_ready.emit(gpu_mod.detect_gpus()), daemon=True).start()

    def _check_engines(self, show: bool = False):
        st = engines.engines_status()
        missing = [k for k, v in st.items() if not v]
        text = "Engines: " + ", ".join(f"{k} {'OK' if v else 'MISSING'}" for k, v in st.items())
        self.engine_label.setText(text)
        if missing and (show or "Real-ESRGAN" in missing or "FFmpeg" in missing):
            QMessageBox.warning(self, "Missing components",
                                "These parts are missing: " + ", ".join(missing) +
                                ".\n\nPlease reinstall UltraEnhance Pro. If you are running from source, "
                                "run tools\\fetch_deps.py once.")
        elif show:
            QMessageBox.information(self, "AI engines", text.replace(", ", "\n"))

    # ================================================================ processing
    def start(self):
        i = self.tabs.currentIndex()
        if i == 1:
            self.enqueue_photos()
        elif i == 2:
            self.enqueue_videos()
        else:
            if self.queue.paused:
                self.queue.resume()
            elif self.photo_opts or self.video_opts:
                if self.photo_opts:
                    self.enqueue_photos(switch=False)
                if self.video_opts:
                    self.enqueue_videos(switch=False)
                self.tabs.setCurrentIndex(3)

    def pause_or_resume(self):
        if self.queue.paused:
            self.queue.resume()
            self.a_pause.setText(self.tr_("Pause"))
        else:
            self.queue.pause()
            self.a_pause.setText(self.tr_("Resume"))

    def stop(self):
        if self.queue.pending():
            if QMessageBox.question(self, "Stop", "Stop all running and waiting jobs? "
                                    "Video jobs can be resumed later from the Home tab.") != QMessageBox.Yes:
                return
        self.queue.cancel()
        self.queue.resume()
        self.a_pause.setText(self.tr_("Pause"))

    def _rows_or_all(self, table) -> List[int]:
        return self._selected_rows(table) or list(range(table.rowCount()))

    def enqueue_photos(self, switch: bool = True):
        rows = self._rows_or_all(self.photo_table)
        if not rows:
            QMessageBox.information(self, __app_name__, "Add some photos first.")
            return
        ctx = self.engine_ctx()
        for r in rows:
            path = self.photo_table.item(r, 0).data(Qt.UserRole)
            opts = self.photo_opts[path]
            if opts.face_restore and not face_restore.available():
                QMessageBox.warning(self, "Face restore", "Face restore models are missing, so faces will be skipped.")
            self.queue.add("photo", path, lambda ctl, p=path, o=opts: enhance_photo(p, o, ctx, ctl),
                           title=Path(path).name)
        if switch:
            self.tabs.setCurrentIndex(3)

    def _enqueue_video(self, path: str, opts: VideoOptions):
        ctx = self.engine_ctx()
        self.queue.add("video", path, lambda ctl: enhance_video(path, opts, ctx, ctl), title=Path(path).name)

    def enqueue_videos(self, switch: bool = True):
        rows = self._rows_or_all(self.video_table)
        if not rows:
            QMessageBox.information(self, __app_name__, "Add some videos first.")
            return
        big = []
        for r in rows:
            path = self.video_table.item(r, 0).data(Qt.UserRole)
            if self.video_opts[path].preset.startswith("8K"):
                big.append(Path(path).name)
        if big and QMessageBox.question(
                self, "8K video", "8K video is very slow (often many hours per minute of video) and the "
                "files are very large.\n\n" + "\n".join(big) + "\n\nContinue?") != QMessageBox.Yes:
            return
        for r in rows:
            path = self.video_table.item(r, 0).data(Qt.UserRole)
            self._enqueue_video(path, self.video_opts[path])
        if switch:
            self.tabs.setCurrentIndex(3)

    def estimate_video(self):
        rows = self._rows_or_all(self.video_table)
        if not rows:
            return
        ctx = self.engine_ctx()
        out = []
        for r in rows:
            path = self.video_table.item(r, 0).data(Qt.UserRole)
            try:
                plan = make_plan(path, self.video_opts[path], ctx, info=self.video_info.get(path))
                out.append(f"<b>{Path(path).name}</b><br>" + plan.summary().replace("\n", "<br>"))
            except Exception as e:
                out.append(f"<b>{Path(path).name}</b><br>Could not estimate: {e}")
        self.v_plan.setHtml("<br><br>".join(out))

    def preview_video(self):
        rows = self._selected_rows(self.video_table) or ([0] if self.video_table.rowCount() else [])
        if not rows:
            QMessageBox.information(self, __app_name__, "Add a video first.")
            return
        path = self.video_table.item(rows[0], 0).data(Qt.UserRole)
        opts = self.video_form()
        ctx = self.engine_ctx()
        self.queue.add("preview", path, lambda ctl: preview_clip(path, opts, ctx, ctl),
                       title=f"5-second preview: {Path(path).name}", open_when_done=True)
        self.tabs.setCurrentIndex(3)

    def preview_photo(self):
        rows = self._selected_rows(self.photo_table) or ([0] if self.photo_table.rowCount() else [])
        if not rows:
            QMessageBox.information(self, __app_name__, "Add a photo first.")
            return
        path = self.photo_table.item(rows[0], 0).data(Qt.UserRole)
        opts = self.photo_form()
        ctx = self.engine_ctx()
        self.compare.placeholder = "Working on the preview..."
        self.compare.set_images(None, None)

        def work():
            try:
                before, after = preview_crop(path, opts, ctx)
                self.bridge.preview_ready.emit(before, after, "")
            except Exception as e:
                log.exception("preview failed")
                from ..job_queue import friendly_error
                self.bridge.preview_ready.emit(None, None, friendly_error(e))
        threading.Thread(target=work, daemon=True).start()

    def preview_current(self):
        if self.tabs.currentIndex() == 2:
            self.preview_video()
        else:
            self.tabs.setCurrentIndex(1)
            self.preview_photo()

    def _on_preview_ready(self, before, after, err):
        if err:
            self.compare.placeholder = "Preview failed"
            self.compare.set_images(None, None)
            QMessageBox.warning(self, "Preview", err)
            return
        self.compare.set_images(pil_to_qpixmap(before), pil_to_qpixmap(after))

    def _on_job_event(self, ev, job):
        self._refresh_queue()
        if ev == "finished" and job is not None:
            if job.state == JobState.DONE and job.result:
                self._last_output = job.result
                if job.options.get("open_when_done"):
                    open_path(job.result)
            elif job.state == JobState.FAILED:
                self.statusBar().showMessage(f"{job.title}: {job.error}", 10000)
            if job.kind == "video":
                self._refresh_unfinished()
            if not self.queue.pending():
                QApplication.alert(self)

    def open_output(self):
        if self._last_output and Path(self._last_output).exists():
            folder = Path(self._last_output).parent
        else:
            folder = self.settings.resolved_output_dir()
        open_path(folder)

    def clean_temp(self):
        if self.queue.pending():
            QMessageBox.information(self, __app_name__, "Wait until all jobs finish before cleaning.")
            return
        temp = self.settings.resolved_temp_dir()
        size = sum(f.stat().st_size for f in temp.rglob("*") if f.is_file())
        if QMessageBox.question(self, "Clean temporary files",
                                f"Delete {human_bytes(size)} of temporary files in {temp}?\n"
                                "Unfinished video jobs will then start from the beginning.") == QMessageBox.Yes:
            shutil.rmtree(temp, ignore_errors=True)
            temp.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(APP_DIR / "jobs", ignore_errors=True)
            self._refresh_unfinished()

    # ================================================================ status / misc
    def _tick(self):
        cpu = gpu_mod.cpu_load_percent()
        if not self._load_thread_running:
            self._load_thread_running = True

            def read():
                try:
                    self.bridge.gpu_load.emit(gpu_mod.gpu_load_percent())
                finally:
                    self._load_thread_running = False
            threading.Thread(target=read, daemon=True).start()
        g = self._gpu_load_value
        self.s_load.setText(f"CPU {cpu}% | GPU {g if g is not None else '--'}%")
        running = [j for j in self.queue.jobs if j.state in (JobState.RUNNING, JobState.PAUSED)]
        if running:
            j = running[0]
            self.s_job.setText(f"{j.state.value}: {j.title[:40]} {j.progress * 100:.0f}%")
            self.s_eta.setText(f"ETA {human_time(j.eta)}" if j.eta else "")
        else:
            waiting = len([j for j in self.queue.jobs if j.state == JobState.QUEUED])
            self.s_job.setText(f"Idle ({waiting} waiting)" if waiting else "Idle")
            self.s_eta.setText("")

    def about(self):
        QMessageBox.about(self, f"About {__app_name__}",
                          f"<h3>{__app_name__} {__version__}</h3>"
                          "<p>Free AI photo and video upscaler for Windows. No watermarks, no subscriptions.</p>"
                          "<p>Built with Real-ESRGAN, RIFE, GFPGAN, FFmpeg, Qt (PySide6), OpenCV, Pillow and ONNX Runtime. "
                          "See Help &gt; Open-source licences.</p>")

    def show_licences(self):
        from ..config import resource_root
        for cand in (resource_root() / "licenses" / "THIRD_PARTY_NOTICES.txt",
                     resource_root() / "THIRD_PARTY_NOTICES.txt"):
            if cand.exists():
                open_path(cand)
                return
        QMessageBox.information(self, "Licences", "The licence file is in the 'licenses' folder of the installation.")

    def closeEvent(self, e):
        if self.queue.pending():
            if QMessageBox.question(self, "Quit", "Jobs are still running. Quit anyway? "
                                    "Video jobs can be resumed next time.") != QMessageBox.Yes:
                e.ignore()
                return
        self.queue.shutdown()
        self.save_settings_quiet()
        e.accept()

    def save_settings_quiet(self):
        try:
            self.settings.save()
        except Exception:
            pass


HELP_HTML = """
<h2>{app} {ver} - quick guide</h2>
<h3>Photos</h3>
<ol>
<li>Open <b>Photo Enhance</b> and drop photos or whole folders (JPG, PNG, WEBP, BMP, TIFF, HEIC and more).</li>
<li>Choose a preset (720p to 8K) or a scale (2x, 3x, 4x, 8x) and how to keep the aspect ratio:
<i>Fit</i> keeps the whole picture, <i>Fill</i> crops the edges to fill the frame exactly, <i>Pad</i> adds black bars.</li>
<li>Pick a model: <i>Photo</i> for real photos, <i>Anime / Art</i> for drawings, <i>Old photo restore</i> for scanned prints
(turns on face restore, denoise, colour fix and JPEG clean-up automatically).</li>
<li>Press <b>Preview</b> to see a before/after of the centre of the photo. Drag the white line, use the mouse wheel to zoom.</li>
<li>Press <b>Enhance photos</b>. Results go to your output folder; originals are never changed.</li>
</ol>
<h3>Videos</h3>
<ol>
<li>Open <b>Video Enhance</b> and drop videos (MP4, MKV, MOV, AVI, WEBM, WMV, FLV and more).</li>
<li>Choose the export settings, select one or more videos in the list, then press
<b>Apply settings to selected</b>. Every video keeps its own settings, shown in the list.</li>
<li>Press <b>Estimate</b> for an honest size, disk-space and time estimate, and <b>Preview 5 seconds</b> to test first.</li>
<li>Press <b>Enhance videos</b>. If the PC shuts down, open the app again: the Home tab offers to resume.</li>
</ol>
<h3>Tips</h3>
<ul>
<li>"Ran out of memory": Settings &gt; Tile size, try 128 or 64, then Retry in the Batch Queue.</li>
<li>Video 4K/8K is slow. 8K can take many hours per minute of video and needs hundreds of GB of free space.</li>
<li>Pause keeps the job in memory; Stop cancels it (video jobs can still be resumed later).</li>
<li>Log file: {log}</li>
</ul>
"""
