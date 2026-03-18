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
    view = EEGView()
    qtbot.addWidget(view)
    val = 42.0
    view.append_sample(val)
    assert len(view._pending) > 0
    # Force render
    time.sleep(0.1)
    view._render_frame()
    # Display should contain the value
    assert val in view.display