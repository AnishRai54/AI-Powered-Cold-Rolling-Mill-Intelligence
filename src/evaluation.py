"""Metadata-backed evaluation utilities."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def load_metadata(path: str = "models/metadata/model_metadata.json") -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
