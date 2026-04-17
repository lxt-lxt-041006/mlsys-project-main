from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ENV_FILE)

api_key = os.getenv("API_KEY", "")
base_url = os.getenv("BASE_URL", "").strip()

client_kwargs = {"api_key": api_key}
if base_url:
    client_kwargs["base_url"] = base_url

client = OpenAI(**client_kwargs)

