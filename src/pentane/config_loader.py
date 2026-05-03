"""
config_loader.py — Single source of truth for all force-field parameters.

Every other module must import CFG from here. No file paths scattered around.
"""
import tomllib
from pathlib import Path

_TOML = Path(__file__).parents[2] / "config" / "trappe_ua.toml"


def load_cfg() -> dict:
    """Read trappe_ua.toml and return the parsed dict."""
    with open(_TOML, "rb") as f:
        return tomllib.load(f)


# Module-level singleton — import CFG from here, never call load_cfg() twice.
CFG = load_cfg()
