import pytest
from PySide6.QtWidgets import QApplication
from main import DSAApplication, DSAView, PSDView, EEGView, SettingsDialog
from config import UserConfig

import sys
import time

@pytest.fixture(scope="session")
def app():
    """Create the QApplication once for all tests."""
    test_app = QApplication.instance()
    if test_app is None:
        test_app = QApplication(sys.argv)
    return test_app


@pytest.fixture
def dsa_app(qtbot, app, monkeypatch):
    """Create a DSAApplication instance for testing."""
    class DummyWorker:
        def __init__(self, user_config):
            self.user_config = user_config

        def apply_config(self, user_config):
            self.user_config = user_config

        def stop(self):
            pass

    class DummyThread:
        def quit(self):
            pass

        def wait(self):
            pass

    def fake_init_worker(self):
        self.thread = DummyThread()
        self.worker = DummyWorker(self.user_config)

    monkeypatch.setattr(DSAApplication, "_init_worker", fake_init_worker)

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
    assert hasattr(dsa_app.topbar, "calibrate_btn")


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
    conf = UserConfig()
    view = EEGView(conf)
    qtbot.addWidget(view)
    val = 42.0
    view.append_sample(val)
    assert len(view._pending) > 0
    view._pending[0] = (time.perf_counter() - 1.0, val)
    view._render_frame()
    assert view.history[-1] == val


def test_multitaper_menu_action_updates_user_config(dsa_app):
    dsa_app.action_multitaper.trigger()

    assert dsa_app.user_config.use_multitaper is False
    assert dsa_app.worker.user_config.use_multitaper is False


def test_on_new_samples_is_ignored_when_eeg_view_is_hidden(dsa_app):
    dsa_app.eeg_view.hide()
    before = dsa_app._last_data_receive_time

    dsa_app._on_new_samples([1.0, 2.0, 3.0])

    assert len(dsa_app.eeg_view._pending) == 0
    assert dsa_app._last_data_receive_time == before
