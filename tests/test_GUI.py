from settings_ui import TopBar
from config import UserConfig


def test_live_button_visibility(qtbot):
    config = UserConfig()
    topbar = TopBar(config, lambda x: None)

    qtbot.addWidget(topbar)

    topbar.set_stream_connected(True)
    topbar.update_indicator(dsa_view=type("dummy", (), {"is_last_dsa_visible": lambda: False})())

    assert topbar.live_btn.isVisible()