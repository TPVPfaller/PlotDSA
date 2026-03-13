import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer
from main import DSAApplication, DSAView, PSDView, EEGView, TopBar, SettingsDialog

import sys
import time
import numpy as np

@pytest.fixture(scope="session")
def app():
    """Create the QApplication once for all tests."""
    test_app = QApplication.instance()
    if test_app is None:
        test_app = QApplication(sys.argv)
    return test_app


@pytest.fixture
def dsa_app(qtbot, app):
    """Create a DSAApplication instance for testing."""
    window = DSAApplication()
    qtbot.addWidget(window)
    window.show()
    return window


def test_initial_ui_state(dsa_app):
    """Verify that main views and widgets are initialized and visible as expected."""
    assert isinstance(dsa_app.dsa_view, DSAView)
    assert isinstance(dsa_app.psd_view, PSDView)
    assert isinstance(dsa_app.eeg_view, EEGView)
    # Initial visibility matches toggle defaults
    assert dsa_app.dsa_view.isVisible() is True
    assert dsa_app.psd_view.isVisible() is False
    assert dsa_app.eeg_view.isVisible() is True
    # TopBar widgets exist
    assert hasattr(dsa_app.topbar, "live_btn")
    assert hasattr(dsa_app.topbar, "zoom_slider")
    assert hasattr(dsa_app.topbar, "norm_checkbox")


def test_menu_toggle_views(qtbot, dsa_app):
    """Toggle view visibility via menu actions."""
    dsa_app.action_show_psd.trigger()
    assert dsa_app.psd_view.isVisible() is True

    dsa_app.action_show_dsa.trigger()
    assert dsa_app.dsa_view.isVisible() is False

    dsa_app.action_show_eeg.trigger()
    assert dsa_app.eeg_view.isVisible() is False

    # Toggle back
    dsa_app.action_show_dsa.trigger()
    assert dsa_app.dsa_view.isVisible() is True


def test_jump_to_live_button_shows(qtbot, dsa_app, monkeypatch):
    """Ensure 'Jump to Live' button visibility logic works."""
    # Fake a DSA buffer with last timestamp
    class FakeBuffer:
        t0 = 0
        def get_last_timestamp(self):
            return 100.0
        def get_view_at(self, width, height, pan_offset_sec):
            return 0, np.ones((height, width))

    dsa_app.dsa_view._buffer = FakeBuffer()
    dsa_app.dsa_view._live_mode = False

    # Initially hidden
    dsa_app.topbar.update_jump_live_btn(dsa_app.dsa_view)
    assert dsa_app.topbar.live_btn.isVisible() is True

    # If live mode active, button hides
    dsa_app.dsa_view._live_mode = True
    dsa_app.topbar.update_jump_live_btn(dsa_app.dsa_view)
    assert dsa_app.topbar.live_btn.isVisible() is False


def test_dsa_view_update_and_live_mode(qtbot, dsa_app):
    """Test updating DSAView with a buffer triggers live mode and updates image."""
    class FakeBuffer:
        t0 = 0
        def get_last_timestamp(self):
            return 10.0
        def get_view_at(self, width, height, pan_offset_sec):
            return 0, np.ones((height, width))

    buf = FakeBuffer()
    dsa_app.dsa_view._buffer = buf
    dsa_app.dsa_view._live_mode = False
    dsa_app.dsa_view.update(buf)
    assert dsa_app.dsa_view._live_mode is True
    # Image should have proper shape
    assert dsa_app.dsa_view.dsa_rect.shape[0] == dsa_app.dsa_view.n_freq_bins


def test_topbar_zoom_slider_changes_config(qtbot, dsa_app):
    """Check that moving the zoom slider updates the config via callback."""
    old_display_minutes = dsa_app.user_config.display_minutes
    slider = dsa_app.topbar.zoom_slider
    # Simulate moving slider
    slider.setValue(min(slider.maximum(), slider.value() + 10))
    # Config updated
    new_display_minutes = dsa_app.user_config.display_minutes
    assert new_display_minutes != old_display_minutes


def test_topbar_norm_checkbox_updates_config(qtbot, dsa_app):
    """Toggling the PSD normalization checkbox should update config."""
    checkbox = dsa_app.topbar.norm_checkbox
    initial = dsa_app.user_config.normalize_psd
    checkbox.setChecked(not initial)
    time.sleep(0.1)  # Allow signal to propagate
    assert dsa_app.user_config.normalize_psd != initial


def test_settings_dialog_sliders_apply(qtbot, dsa_app):
    """Verify SettingsDialog can apply changes and trigger config callback."""
    dialog = SettingsDialog(dsa_app.user_config, dsa_app._on_config_change)
    qtbot.addWidget(dialog)
    dialog.show()

    # Modify a slider
    window_slider, scale = dialog.sliders["Window (s)"]
    old_value = window_slider.value()
    new_value = old_value + int(scale)
    window_slider.setValue(new_value)

    # Simulate Apply click
    dialog._apply()
    # Config updated
    assert dsa_app.user_config.window_sec == new_value


def test_eeg_view_append_sample(qtbot):
    """Appending a sample adds it to pending queue and eventually renders."""
    view = EEGView(window_sec=1.0)
    qtbot.addWidget(view)
    val = 42.0
    ts = time.perf_counter()
    view.append_sample(ts, val)
    assert len(view._pending) > 0
    # Force render
    view._render_frame()
    # Display should contain the value
    assert val in view.display