from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSlider,
    QLabel
)
from PySide6.QtCore import QByteArray, QPoint, QSize, Qt, QRectF
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QPolygon
from PySide6.QtSvg import QSvgRenderer
from .. import config


RESET_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
    <path fill="#FFFFFF" d="M18,28A12,12,0,1,0,6,16v6.2L2.4,18.6,1,20l6,6,6-6-1.4-1.4L8,22.2V16H8A10,10,0,1,1,18,26Z"/>
</svg>"""


def create_reset_icon(size=18):
    renderer = QSvgRenderer(QByteArray(RESET_SVG.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()

    return QIcon(pixmap)


# ------------------ TopBar ------------------ #
class TopBar(QWidget):
    BUTTON_HEIGHT = 35

    def __init__(self, user_config, on_config_change, on_zoom_change, on_pan, on_calibrate):
        super().__init__()
        self.user_config = user_config
        self.on_config_change = on_config_change
        self.on_zoom_change = on_zoom_change
        self.on_pan = on_pan
        self.on_calibrate = on_calibrate

        self.GAMMA = 2.5 # Zoom shape parameter

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.left_btn = self._create_arrow_button("left")
        self.left_btn.clicked.connect(lambda: self._pan(-1))
        layout.addWidget(self.left_btn)

        self.right_btn = self._create_arrow_button("right")
        self.right_btn.clicked.connect(lambda: self._pan(1))
        layout.addWidget(self.right_btn)

        self.zoom_label = QLabel("Zoom:")
        self.zoom_label.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
        layout.addWidget(self.zoom_label)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(1)
        self.zoom_slider.setMaximum(1000)
        self.zoom_slider.valueChanged.connect(self._zoom_changed)
        layout.addWidget(self.zoom_slider)

        self.live_btn = self._create_button("\u25b6 Live")
        self.live_btn.setMinimumWidth(70)
        policy = self.live_btn.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.live_btn.setSizePolicy(policy)
        self.live_btn.hide()
        layout.addWidget(self.live_btn)

        self.calibrate_btn = self._create_button("Calibrate")
        self.calibrate_btn.clicked.connect(self.on_calibrate)
        layout.addWidget(self.calibrate_btn)

        self.reset_btn = self._create_button()
        self.reset_btn.setIcon(create_reset_icon())
        self.reset_btn.setIconSize(QSize(18, 18))
        self.reset_btn.clicked.connect(self._reset_calibration)
        layout.addWidget(self.reset_btn)

    def _button_style(self, font_size=None, padding="8px 12px"):
        resolved_font_size = config.FONT_SIZE if font_size is None else font_size
        return f"""
            QPushButton {{
                background-color: rgb(58, 58, 58);
                color: white;
                border: 1px solid rgb(78, 78, 78);
                border-radius: 6px;
                padding: {padding};
                font-size: {resolved_font_size}px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: rgb(72, 72, 72);
            }}
            QPushButton:pressed {{
                background-color: rgb(92, 92, 92);
            }}
            QPushButton:disabled {{
                background-color: rgb(48, 48, 48);
                color: rgb(150, 150, 150);
            }}
        """

    def _create_button(self, text="", font_size=None, padding="8px 12px", min_width=None):
        button = QPushButton(text)
        button.setMinimumHeight(self.BUTTON_HEIGHT)
        if min_width is not None:
            button.setMinimumWidth(min_width)
        button.setStyleSheet(self._button_style(font_size=font_size, padding=padding))
        return button

    def _create_arrow_button(self, direction):
        button = self._create_button("", padding="0px", min_width=self.BUTTON_HEIGHT)
        button.setIcon(self._create_arrow_icon(direction))
        button.setIconSize(QSize(16, 16))
        return button

    def _create_arrow_icon(self, direction):
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("white"))

        if direction == "left":
            points = [QPoint(10, 3), QPoint(5, 8), QPoint(10, 13)]
        else:
            points = [QPoint(6, 3), QPoint(11, 8), QPoint(6, 13)]

        painter.drawPolygon(QPolygon(points))
        painter.end()
        return QIcon(pixmap)

    # ------------------ New actions ------------------ #
    def _reset_calibration(self):
        new_config = self.user_config.update(psd_db_min=config.PSD_DB_MIN, psd_db_max=config.PSD_DB_MAX)
        self.on_config_change(new_config)

    def _pan(self, direction: int):
        """direction: -1 = left, +1 = right"""
        step = 0.27 # percent of display width
        self.on_pan(direction * step)

    def _zoom_changed(self, value):
        min_minutes, max_minutes = config.DISPLAY_MINUTES_BOUNDS

        t = (value - 1) / 999.0  # 0..1 linear

        # invert gamma effect for desired behavior
        t = 1.0 - (1.0 - t) ** self.GAMMA

        display_minutes = max_minutes - t * (max_minutes - min_minutes)
        self.on_zoom_change(display_minutes)

    def sync_slider(self, display_minutes):
        min_minutes, max_minutes = config.DISPLAY_MINUTES_BOUNDS

        t = (max_minutes - display_minutes) / (max_minutes - min_minutes)
        t = max(0.0, min(1.0, t))

        # inverse of forward curve
        t = 1.0 - (1.0 - t) ** (1.0 / self.GAMMA)

        val_zoom = 1 + 999.0 * t

        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(round(val_zoom)))
        self.zoom_slider.blockSignals(False)

    def update_jump_live_btn(self, dsa_view):
        if dsa_view.dsa_buffer.t0 is None or dsa_view.live_mode:
            self.live_btn.hide()
        else:
            self.live_btn.show()

    def apply_config(self, user_config):
        self.user_config = user_config