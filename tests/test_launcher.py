import sys

import pytest

from vnn_survey.app import launcher


def test_launcher_disables_developer_shortcuts_and_allows_large_backups(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SURVEYFLOW_MAX_UPLOAD_MB", "640")
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setattr(launcher.streamlit_cli, "main", lambda: 0)

    with pytest.raises(SystemExit) as result:
        launcher.main()

    assert result.value.code == 0
    assert "--client.toolbarMode=minimal" in sys.argv
    assert "--server.maxUploadSize=640" in sys.argv
