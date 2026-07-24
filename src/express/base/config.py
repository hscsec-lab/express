import getpass
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from express import console

REQUIRED_KEYS = ("S3_AK", "S3_SK", "S3_ENDPOINT", "S3_BUCKET")
OPTIONAL_KEYS = ("LOCAL_WORKDIR",)
CONFIG_FILE_NAME = "config.json"


def config_dir() -> Path:
    """Return the OS-standard per-user config directory for express."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / "express"


def config_path() -> Path:
    return config_dir() / CONFIG_FILE_NAME


def default_local_workdir() -> str:
    return str(Path.home() / "models")


def load_config() -> Dict[str, str]:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read config file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid config file {path}: expected a JSON object")
    return {str(k): str(v) for k, v in data.items() if v is not None and str(v) != ""}


def save_config(data: Dict[str, str]) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: data[key] for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS) if data.get(key)}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def apply_config_to_env(data: Dict[str, str]) -> None:
    """Fill missing process env vars from saved config. Existing env wins."""
    for key, value in data.items():
        if value and not os.getenv(key):
            os.environ[key] = value


def missing_required_keys() -> List[str]:
    return [key for key in REQUIRED_KEYS if not os.getenv(key)]


def _prompt_value(label: str, *, default: Optional[str] = None, secret: bool = False) -> str:
    hint = f" [{default}]" if default else ""
    while True:
        if secret:
            # getpass cannot show a default inline; print hint separately.
            if default:
                console.print(f"{label}{hint} (leave blank to keep current)")
                value = getpass.getpass(f"{label}: ")
                value = value.strip() or default
            else:
                value = getpass.getpass(f"{label}: ").strip()
        else:
            raw = input(f"{label}{hint}: ").strip()
            value = raw or (default or "")
        if value:
            return value
        console.print("Value is required.")


def interactive_configure(existing: Optional[Dict[str, str]] = None, *, first_run: bool = False) -> Dict[str, str]:
    existing = existing or {}
    if first_run:
        console.print("Express first-time setup: configure remote storage.")
    else:
        console.print("Express configuration.")
    console.print(f"Config file: {config_path()}")

    data: Dict[str, str] = {}
    data["S3_AK"] = _prompt_value("S3 Access Key (S3_AK)", default=existing.get("S3_AK") or os.getenv("S3_AK"))
    data["S3_SK"] = _prompt_value(
        "S3 Secret Key (S3_SK)",
        default=existing.get("S3_SK") or os.getenv("S3_SK"),
        secret=True,
    )
    data["S3_ENDPOINT"] = _prompt_value(
        "S3 Endpoint (S3_ENDPOINT)",
        default=existing.get("S3_ENDPOINT") or os.getenv("S3_ENDPOINT"),
    )
    data["S3_BUCKET"] = _prompt_value(
        "S3 Bucket (S3_BUCKET)",
        default=existing.get("S3_BUCKET") or os.getenv("S3_BUCKET"),
    )
    data["LOCAL_WORKDIR"] = _prompt_value(
        "Local workdir (LOCAL_WORKDIR)",
        default=existing.get("LOCAL_WORKDIR") or os.getenv("LOCAL_WORKDIR") or default_local_workdir(),
    )
    return data


def ensure_remote_config(*, force_interactive: bool = False) -> Path | None:
    """
    Load saved config into env. On first use (or force), run interactive setup.

    Environment variables always take precedence over the config file.
    Returns the config path when a file is used/written, else None.
    """
    saved = load_config()
    apply_config_to_env(saved)

    missing = missing_required_keys()
    if not force_interactive and not missing:
        return config_path() if saved else None

    if not sys.stdin.isatty():
        where = config_path()
        raise EnvironmentError(
            "Missing S3 configuration and no interactive terminal is available. "
            f"Set S3_AK/S3_SK/S3_ENDPOINT/S3_BUCKET, or create {where}, "
            "or run `express config` in a terminal."
        )

    merged = {**saved}
    for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS):
        if os.getenv(key):
            merged[key] = os.environ[key]

    data = interactive_configure(merged, first_run=not saved and not force_interactive)
    path = save_config(data)
    apply_config_to_env(data)
    console.print(f"Saved config to {path}")
    return path
