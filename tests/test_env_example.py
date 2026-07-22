"""Anti-drift test: .env.example stays in sync with Settings.

Every field declared on ``Settings`` must have a matching ``NAME=`` line in
``.env.example`` (uppercased field name). This catches the common drift where a
new setting is added to the config class but the example env file is forgotten
(or vice versa).

``Settings.is_offline`` is a ``@property``, so it is not part of
``model_fields`` and needs no manual exclusion here.
"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def _env_example_keys() -> set[str]:
    """Parse .env.example line by line, collecting keys left of the first '='."""
    keys: set[str] = set()
    for raw_line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name = line.split("=", 1)[0].strip()
        if name:
            keys.add(name)
    return keys


def test_every_setting_present_in_env_example() -> None:
    env_keys = _env_example_keys()
    missing = {
        field_name.upper()
        for field_name in Settings.model_fields
        if field_name.upper() not in env_keys
    }
    assert not missing, f"fields missing from .env.example: {sorted(missing)}"
