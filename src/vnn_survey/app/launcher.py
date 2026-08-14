from __future__ import annotations

import os
import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli


def main() -> None:
    app_path = Path(__file__).with_name("main.py")
    address = os.environ.get("SURVEYFLOW_ADDRESS", "localhost")
    port = os.environ.get("SURVEYFLOW_PORT", "8501")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        f"--server.address={address}",
        f"--server.port={port}",
    ]
    raise SystemExit(streamlit_cli.main())
