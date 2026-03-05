import time

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QScrollArea
)
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSlider, QLabel, QGridLayout, QFrame, QSizePolicy, QCheckBox
from PySide6.QtCore import Qt
from config import SystemConfig
import numpy as np
from PySide6.QtGui import QFontMetrics
import math


class TopBar(QWidget):
    def __init__(self, config, on_config_change):
        super().__init__()
        self.config = config
        self.on_config_change = on_config_change

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # --- Zoom slider ---
        self.zoom_label = QLabel("Zoom:")
        self.zoom_label.setMinimumHeight(50)
        layout.addWidget(self.zoom_label)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(1)
        self.zoom_slider.setMaximum(100)
        self.zoom_slider.setValue(int(config.display_minutes/SystemConfig.DISPLAY_MINUTES_BOUNDS[1])*100)
        #self.zoom_slider.setMinimumHeight(50)
        self.zoom_slider.valueChanged.connect(self._zoom_changed)
        layout.addWidget(self.zoom_slider)
        layout.setAlignment(self.zoom_label, Qt.AlignVCenter)
        layout.setAlignment(self.zoom_slider, Qt.AlignVCenter)

        self.sync_slider(config)

        # --- Live indicator ---
        self.live_indicator = QLabel("DISCONNECTED")
        self.live_indicator.setStyleSheet("""
                        QLabel {
                            color: palette(window-text);
                            background-color: red;
                            padding: 5px 10px;
                            border-radius: 5px;
                            font-weight: bold;
                            font-size: 11px;
                        }
                    """)
        self.live_indicator.setMinimumHeight(50)
        self.live_indicator.setMinimumWidth(110)
        self.live_indicator.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.live_indicator)
        layout.setAlignment(self.live_indicator, Qt.AlignVCenter)

        self.live_btn = QPushButton("▶ Live")
        self.live_btn.setStyleSheet("""
            QPushButton {
                background-color: palette(button);
                color: palette(button-text);
                border: 1px solid palette(mid);
                border-radius: 5px;
                padding: 5px 10px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: palette(midlight);
            }
            QPushButton:pressed {
                background-color: palette(mid);
            }
        """)
        self.live_btn.setMinimumHeight(50)
        self.live_btn.setMinimumWidth(80)
        
        policy = self.live_btn.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.live_btn.setSizePolicy(policy)
        
        self.live_btn.hide()
        layout.addWidget(self.live_btn)
        layout.setAlignment(self.live_btn, Qt.AlignVCenter)

        self._last_data_receive_time = 0

        # --- PSD Normalization Checkbox ---
        self.norm_checkbox = QCheckBox("Relative PSD")
        self.norm_checkbox.setChecked(self.config.normalize_psd)
        self.norm_checkbox.setMinimumHeight(60)

        self.norm_checkbox.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                font-size: 11px;
                padding: 5px;
            }
        """)

        self.norm_checkbox.toggled.connect(self._normalize_toggled)

        layout.addWidget(self.norm_checkbox)
        layout.setAlignment(self.norm_checkbox, Qt.AlignVCenter)


    def _normalize_toggled(self, checked):
        new_config = self.config.update(normalize_psd=bool(checked))
        self.on_config_change(new_config)

    def _zoom_changed(self, value):
        min_minutes = SystemConfig.DISPLAY_MINUTES_BOUNDS[0]
        max_minutes = SystemConfig.DISPLAY_MINUTES_BOUNDS[1]

        # Non-linear: square the normalized value so low end (zoomed out)
        # is coarse and high end (zoomed in) has finer control
        t = 1.0 - ((1.0 - (value - 1) / 99.0) ** 2)
        new_display_minutes = max_minutes - t * (max_minutes - min_minutes)

        new_config = self.config.update(display_minutes=new_display_minutes)
        self.on_config_change(new_config)

    def sync_slider(self, config):
        """Update sliders based on current config without triggering feedback."""
        self.config = config

        # Sync Zoom Slider
        min_min = SystemConfig.DISPLAY_MINUTES_BOUNDS[0]
        max_min = SystemConfig.DISPLAY_MINUTES_BOUNDS[1]
        curr_min = self.config.display_minutes

        t = (max_min - curr_min) / (max_min - min_min) if max_min != min_min else 0
        val_zoom = 1 + 99.0 * (1.0 - np.sqrt(max(0, 1.0 - t)))
        
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(np.round(val_zoom)))
        self.zoom_slider.blockSignals(False)

    def reset_last_data_timer(self):
        self._last_data_receive_time = time.time()

    def update_indicator(self):
        if self._last_data_receive_time + 2.0 < time.time():
            self.live_indicator.setText("CONNECTED")
            self.live_indicator.setStyleSheet("""
                            QLabel {
                                color: palette(window-text);
                                background-color: green;
                                padding: 5px 10px;
                                border-radius: 5px;
                                font-weight: bold;
                                font-size: 11px;
                            }
                        """)
        else:
            self.live_indicator.setText("DISCONNECTED")
            self.live_indicator.setStyleSheet("""
                            QLabel {
                                color: palette(window-text);
                                background-color: red;
                                padding: 5px 10px;
                                border-radius: 5px;
                                font-weight: bold;
                                font-size: 11px;
                            }
                        """)

    def update_jump_live_btn(self, dsa_view):
        has_data = False
        try:
            if hasattr(dsa_view, "_buffer") and getattr(dsa_view, "_buffer") is not None:
                last_ts = dsa_view._buffer.get_last_timestamp()
                if last_ts is not None and np.isfinite(float(last_ts)):
                    has_data = True
        except Exception:
            has_data = False

        is_live = False
        if has_data:
            if hasattr(dsa_view, "is_last_dsa_visible"):
                is_live = dsa_view.is_last_dsa_visible()
            elif hasattr(dsa_view, "_live_mode"):
                is_live = bool(dsa_view._live_mode)

        if has_data and (not is_live):
            self.live_btn.show()
        else:
            self.live_btn.hide()


class SettingsDialog(QDialog):
    def __init__(self, config, on_config_change, parent=None):
        super().__init__(parent)

        self.setWindowTitle("System Settings")
        self.setMinimumSize(520, 240)

        self.config = config
        self.on_config_change = on_config_change

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        # Column policy: [0]=labels (fixed), [1]=sliders (expanding), [2]=values (fixed)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)

        self.sliders = {}
        row_idx = 0

        def add_slider(name, bounds, value, scale=1, unit: str = "", display_factor: float = 1.0, decimals_override: int | None = None):
            nonlocal row_idx
            row_h = 26  # uniform row height for visual centering across DPI

            name_label = QLabel(name)
            name_label.setStyleSheet("font-size: 16px;")
            name_label.setContentsMargins(0, 0, 0, 0)
            name_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
            name_label.setFixedHeight(row_h)
            name_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(bounds[0] * scale))
            slider.setMaximum(int(bounds[1] * scale))
            slider.setValue(int(value * scale))
            slider.setFixedHeight(row_h)
            slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            # Scoped style to normalize handle/groove vertical centering on Windows themes
            slider.setStyleSheet(
                """
                QSlider::groove:horizontal {
                    height: 5px;
                    background: palette(mid);
                    border: 1px solid palette(shadow);
                    margin: 0px 0px;
                    border-radius: 2px;
                }
                QSlider::sub-page:horizontal {
                    background: palette(highlight);
                    border-radius: 2px;
                }
                QSlider::add-page:horizontal {
                    background: palette(mid);
                    border-radius: 2px;
                }
                QSlider::handle:horizontal {
                    width: 14px;
                    height: 14px;
                    margin: -10px 0px;
                    border-radius: 7px;
                    background: palette(window-text);
                    border: 1px solid palette(base);
                }
                """
            )

            # Determine fixed decimal formatting
            if decimals_override is not None:
                decimals = max(0, int(decimals_override))
            else:
                decimals = 0
                if scale and scale > 1:
                    try:
                        decimals = max(0, int(round(math.log10(scale))))
                    except Exception:
                        decimals = 0

            def fmt_number(x: float) -> str:
                return f"{x:.{decimals}f}"

            def fmt_with_unit(x: float) -> str:
                return f"{fmt_number(x)}{unit}"

            # Initial label with unit
            value_label = QLabel(fmt_with_unit(value * display_factor))
            value_label.setStyleSheet("font-size: 16px;")
            value_label.setAlignment(Qt.AlignLeft)
            value_label.setFixedHeight(row_h)
            value_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

            # Fixed width based on transformed bounds text width (including unit)
            fm = QFontMetrics(value_label.font())
            b0_txt = fmt_with_unit(bounds[0] * display_factor)
            b1_txt = fmt_with_unit(bounds[1] * display_factor)
            max_w = max(fm.horizontalAdvance(b0_txt), fm.horizontalAdvance(b1_txt)) + 12
            value_label.setFixedWidth(max_w)

            def update(val):
                # val is in scaled integer space; convert back, then apply display factor
                shown = (val / scale) * display_factor if scale != 0 else 0.0
                value_label.setText(fmt_with_unit(shown))

            slider.valueChanged.connect(update)

            grid.addWidget(name_label, row_idx, 0, alignment=Qt.AlignVCenter | Qt.AlignRight)
            grid.addWidget(slider, row_idx, 1, alignment=Qt.AlignVCenter)
            grid.addWidget(value_label, row_idx, 2, alignment=Qt.AlignLeft)

            self.sliders[name] = (slider, scale)
            row_idx += 1

        add_slider("Window (s)", SystemConfig.WINDOW_SEC_BOUNDS, config.window_sec, 1, unit=" s")
        add_slider("Segment (s)", SystemConfig.SEGMENT_SEC_BOUNDS, config.segment_sec, 10, unit=" s")
        add_slider("Window Overlap", SystemConfig.WINDOW_OVERLAP_BOUNDS, config.window_overlap, 100, unit=" %", display_factor=100.0, decimals_override=0)
        add_slider("Segment Overlap", SystemConfig.SEGMENT_OVERLAP_BOUNDS, config.segment_overlap, 100, unit=" %", display_factor=100.0, decimals_override=0)
        add_slider("Max Frequency (Hz)", SystemConfig.MAX_FREQ_HZ_BOUNDS, config.max_freq_hz, 1, unit=" Hz")

        # --- Buttons row: Reset on the left, Apply on the right ---
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setMinimumHeight(40)
        reset_btn.setStyleSheet("font-size: 16px;")
        reset_btn.clicked.connect(self._reset_to_defaults)

        apply_btn = QPushButton("Apply and Close")
        apply_btn.setMinimumHeight(40)
        apply_btn.setStyleSheet("font-size: 16px;")
        apply_btn.clicked.connect(self._apply)

        button_bar = QWidget()
        bar_layout = QHBoxLayout(button_bar)
        bar_layout.setContentsMargins(0, 4, 0, 0)
        bar_layout.setSpacing(8)
        bar_layout.addWidget(reset_btn, alignment=Qt.AlignLeft)
        bar_layout.addStretch(1)
        bar_layout.addWidget(apply_btn, alignment=Qt.AlignRight)

        # Button bar spans all columns
        grid.addWidget(button_bar, row_idx, 0, 1, 3)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _reset_to_defaults(self):
        """Reset sliders to factory defaults displayed in this dialog without applying.
        No persistent change until the user confirms with Apply.
        """
        try:
            from PySide6.QtWidgets import QMessageBox
            resp = QMessageBox.question(
                self,
                "Reset Settings",
                "Reset all settings to default values?\n(Changes are not saved until you click 'Apply and Close'.)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if resp != QMessageBox.Yes:
                return
        except Exception:
            # If QMessageBox not available for some reason, continue without confirmation
            pass

        # Build defaults from SystemConfig (single source of truth) and map to sliders using each slider's scale
        mapping = {
            "Window (s)": SystemConfig.window_sec,
            "Segment (s)": SystemConfig.segment_sec,
            "Window Overlap": SystemConfig.window_overlap,
            "Segment Overlap": SystemConfig.segment_overlap,
            "Max Frequency (Hz)": SystemConfig.max_freq_hz,
        }
        for name, (slider, scale) in self.sliders.items():
            if name in mapping:
                try:
                    val = mapping[name]
                    slider.blockSignals(True)
                    slider.setValue(int(round(val * scale)))
                    slider.blockSignals(False)
                    # Manually emit to refresh the value label text
                    slider.valueChanged.emit(slider.value())
                except Exception:
                    try:
                        slider.blockSignals(False)
                    except Exception:
                        pass
                    continue

    def _apply(self):
        try:
            # Read proposed values from sliders (without applying yet)
            proposed_window_sec = self.sliders["Window (s)"][0].value()
            proposed_segment_sec = self.sliders["Segment (s)"][0].value() / 10
            proposed_window_overlap = self.sliders["Window Overlap"][0].value() / 100
            proposed_segment_overlap = self.sliders["Segment Overlap"][0].value() / 100
            proposed_max_freq_hz = self.sliders["Max Frequency (Hz)"][0].value()

            # If segment length changes, warn that DSA history will be cleared
            if abs(proposed_segment_sec - float(self.config.segment_sec)) > 1e-9:
                from PySide6.QtWidgets import QMessageBox
                resp = QMessageBox.question(
                    self,
                    "Change Segment Length",
                    (
                        "Changing the segment length will clear the current DSA view/history.\n\n"
                        "Do you want to proceed?"
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if resp != QMessageBox.Yes:
                    return  # Abort apply; let the user adjust or cancel

            new_config = self.config.update(
                window_sec=proposed_window_sec,
                segment_sec=proposed_segment_sec,
                window_overlap=proposed_window_overlap,
                segment_overlap=proposed_segment_overlap,
                max_freq_hz=proposed_max_freq_hz,
            )
            self.on_config_change(new_config)
            self.accept()
        except ValueError as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Invalid Configuration", str(e))
