from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QScrollArea
)
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSlider, QLabel
from PySide6.QtCore import Qt
from config import SystemConfig
import numpy as np


class TopBar(QWidget):
    def __init__(self, config, on_config_change):
        super().__init__()
        self.config = config
        self.on_config_change = on_config_change

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # --- Settings Button ---
        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.setMinimumHeight(50)
        layout.addWidget(self.settings_btn)
        layout.setAlignment(self.settings_btn, Qt.AlignVCenter)

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
        # Center the zoom controls vertically within the top bar
        layout.setAlignment(self.zoom_label, Qt.AlignVCenter)
        layout.setAlignment(self.zoom_slider, Qt.AlignVCenter)

        # Sync initial values
        self.sync_sliders(config)

        # --- Live indicator ---
        self.live_indicator = QLabel("DISCONNECTED")
        self.live_indicator.setStyleSheet("""
                        QLabel {
                            color: white;
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
                background-color: #444;
                color: white;
                border: 2px solid #666;
                border-radius: 5px;
                padding: 5px 10px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
        """)
        self.live_btn.setMinimumHeight(50)
        self.live_btn.setMinimumWidth(80) # Reserve enough space
        
        # Ensure the button occupies space even when hidden, so the layout doesn't jump
        policy = self.live_btn.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.live_btn.setSizePolicy(policy)
        
        self.live_btn.hide() # Hidden initially
        layout.addWidget(self.live_btn)
        layout.setAlignment(self.live_btn, Qt.AlignVCenter)

        self._last_data_receive_time = 0

    def _zoom_changed(self, value):
        min_minutes = SystemConfig.DISPLAY_MINUTES_BOUNDS[0]
        max_minutes = SystemConfig.DISPLAY_MINUTES_BOUNDS[1]

        # Non-linear: square the normalized value so low end (zoomed out)
        # is coarse and high end (zoomed in) has finer control
        t = 1.0 - ((1.0 - (value - 1) / 99.0) ** 2)
        new_display_minutes = max_minutes - t * (max_minutes - min_minutes)

        new_config = self.config.update(display_minutes=new_display_minutes)
        self.on_config_change(new_config)

    def sync_sliders(self, config, is_new_data=False):
        """Update sliders based on current config without triggering feedback."""
        import time
        if is_new_data:
            self._last_data_receive_time = time.time()

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

    def update_indicator(self, dsa_view):
        """Update live/review/disconnected indicator and Live button visibility.
        Rule: The 'Jump to Live' button must NOT be shown when there is no DSA data.
        It should only be visible when there is historical data (review mode) to jump from.
        """
        import time
        dsa = dsa_view

        # Determine if we have any DSA data yet
        has_data = False
        try:
            if hasattr(dsa, "_buffer") and getattr(dsa, "_buffer") is not None:
                last_ts = dsa._buffer.get_last_timestamp()
                if last_ts is not None and np.isfinite(float(last_ts)):
                    has_data = True
        except Exception:
            has_data = False

        # Determine LIVE vs REVIEW status based on view position (only meaningful if we have data)
        is_live = False
        if has_data:
            if hasattr(dsa, "is_last_dsa_visible"):
                is_live = dsa.is_last_dsa_visible()
            elif hasattr(dsa, "_live_mode"):
                is_live = bool(dsa._live_mode)

        # --- Connection/flow status and indicator text ---
        now = time.time()
        disconnected = (self._last_data_receive_time == 0 or (now - self._last_data_receive_time) > 2.0)
        if disconnected:
            self.live_indicator.setText("DISCONNECTED")
            self.live_indicator.setStyleSheet("""
                QLabel {
                    color: white;
                    background-color: red;
                    padding: 5px 10px;
                    border-radius: 5px;
                    font-weight: bold;
                    font-size: 11px;
                }
            """)
        else:
            if is_live:
                self.live_indicator.setText("LIVE")
                self.live_indicator.setStyleSheet("""
                    QLabel {
                        color: white;
                        background-color: green;
                        padding: 5px 10px;
                        border-radius: 5px;
                        font-weight: bold;
                        font-size: 11px;
                    }
                """)
            else:
                self.live_indicator.setText("REVIEW")
                self.live_indicator.setStyleSheet("""
                    QLabel {
                        color: white;
                        background-color: gray;
                        padding: 5px 10px;
                        border-radius: 5px;
                        font-weight: bold;
                        font-size: 11px;
                    }
                """)

        # --- Live button visibility rule ---
        # Show only if: we have data AND we are not at live (i.e., in review)
        if has_data and (not is_live):
            self.live_btn.show()
        else:
            self.live_btn.hide()


class SettingsDialog(QDialog):
    def __init__(self, config, on_config_change, parent=None):
        super().__init__(parent)

        self.setWindowTitle("System Settings")
        self.setMinimumSize(600, 800)

        self.config = config
        self.on_config_change = on_config_change

        main_layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(30)

        self.sliders = {}

        def add_slider(name, bounds, value, scale=1):
            label = QLabel(f"{name}: {value}")
            label.setStyleSheet("font-size: 16px;")

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(bounds[0] * scale))
            slider.setMaximum(int(bounds[1] * scale))
            slider.setValue(int(value * scale))
            slider.setMinimumHeight(50)

            def update(val):
                label.setText(f"{name}: {val / scale}")

            slider.valueChanged.connect(update)

            layout.addWidget(label)
            layout.addWidget(slider)

            self.sliders[name] = (slider, scale)

        add_slider("Window (s)", SystemConfig.WINDOW_SEC_BOUNDS, config.window_sec, 10)
        add_slider("Segment (s)", SystemConfig.SEGMENT_SEC_BOUNDS, config.segment_sec, 10)
        add_slider("Window Overlap", SystemConfig.WINDOW_OVERLAP_BOUNDS, config.window_overlap, 100)
        add_slider("Segment Overlap", SystemConfig.SEGMENT_OVERLAP_BOUNDS, config.segment_overlap, 100)
        add_slider("Max Frequency (Hz)", SystemConfig.MAX_FREQ_HZ_BOUNDS, config.max_freq_hz, 1)

        apply_btn = QPushButton("Apply & Close")
        apply_btn.setMinimumHeight(60)
        apply_btn.setStyleSheet("font-size: 18px;")
        apply_btn.clicked.connect(self._apply)

        layout.addWidget(apply_btn)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _apply(self):
        try:
            new_config = self.config.update(
                window_sec=self.sliders["Window (s)"][0].value() / 10,
                segment_sec=self.sliders["Segment (s)"][0].value() / 10,
                window_overlap=self.sliders["Window Overlap"][0].value() / 100,
                segment_overlap=self.sliders["Segment Overlap"][0].value() / 100,
                max_freq_hz=self.sliders["Max Frequency (Hz)"][0].value()
            )
            self.on_config_change(new_config)
            self.accept()
        except ValueError as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Invalid Configuration", str(e))
