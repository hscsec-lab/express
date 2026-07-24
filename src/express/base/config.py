import getpass
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from express import console

REQUIRED_KEYS = ("S3_AK", "S3_SK", "S3_ENDPOINT", "S3_BUCKET")
OPTIONAL_KEYS = ("LOCAL_WORKDIR",)
CONFIG_FILE_NAME = "config.json"
BACK_TOKENS = frozenset({"-", ":back", ":prev"})


@dataclass(frozen=True)
class ConfigField:
    key: str
    label: str
    secret: bool = False


CONFIG_FIELDS = (
    ConfigField("S3_AK", "S3 Access Key"),
    ConfigField("S3_SK", "S3 Secret Key", secret=True),
    ConfigField("S3_ENDPOINT", "S3 Endpoint"),
    ConfigField("S3_BUCKET", "S3 Bucket"),
    ConfigField("LOCAL_WORKDIR", "Local workdir"),
)


class GoBack(Exception):
    """User asked to revisit the previous config field."""


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


def _default_for(field: ConfigField, existing: Dict[str, str]) -> Optional[str]:
    if field.key == "LOCAL_WORKDIR":
        return existing.get(field.key) or os.getenv(field.key) or default_local_workdir()
    return existing.get(field.key) or os.getenv(field.key)


def _echo(text: str = "") -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _read_line_raw(prompt: str, *, secret: bool) -> str:
    """TTY line editor: Ctrl+B goes back, Backspace edits, Enter confirms."""
    import termios
    import tty

    _echo(prompt)
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    buf: List[str] = []
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x03":  # Ctrl+C
                _echo("\n")
                raise KeyboardInterrupt
            if ch == "\x04":  # Ctrl+D
                _echo("\n")
                raise EOFError
            if ch == "\x02":  # Ctrl+B → previous field
                _echo("\n")
                raise GoBack()
            if ch in {"\r", "\n"}:
                _echo("\n")
                return "".join(buf)
            if ch in {"\x7f", "\b"}:
                if buf:
                    buf.pop()
                    _echo("\b \b")
                continue
            if ch == "\x15":  # Ctrl+U clear line
                while buf:
                    buf.pop()
                    _echo("\b \b")
                continue
            if len(ch) == 1 and ch.isprintable():
                buf.append(ch)
                _echo("*" if secret else ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _read_line_fallback(prompt: str, *, secret: bool) -> str:
    """Non-raw fallback; type '-' alone to go back."""
    if secret:
        return getpass.getpass(prompt)
    return input(prompt)


def _read_line(prompt: str, *, secret: bool = False) -> str:
    if sys.stdin.isatty() and sys.platform != "win32":
        try:
            import termios
        except ImportError:
            return _read_line_fallback(prompt, secret=secret)
        try:
            return _read_line_raw(prompt, secret=secret)
        except (OSError, termios.error):
            pass
    return _read_line_fallback(prompt, secret=secret)


def _prompt_field(field: ConfigField, default: Optional[str], *, index: int, total: int) -> str:
    hint = f" [{_mask(default) if field.secret and default else default}]" if default else ""
    prompt = f"[{index}/{total}] {field.label}{hint}: "
    while True:
        raw = _read_line(prompt, secret=field.secret).strip()
        if raw in BACK_TOKENS:
            raise GoBack()
        value = raw or (default or "")
        if value:
            return value
        console.print("不能为空。Ctrl+B 或输入 - 可返回上一项。")


def _mask(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def interactive_configure(existing: Optional[Dict[str, str]] = None, *, first_run: bool = False) -> Dict[str, str]:
    existing = existing or {}
    console.print("Express first-time setup: configure remote storage." if first_run else "Express configuration.")
    console.print(f"Config file: {config_path()}")
    console.print("输错了按 [bold]Ctrl+B[/bold] 回到上一项；也可以输入 [bold]-[/bold] 后回车。")

    answers: Dict[str, str] = {}
    index = 0
    total = len(CONFIG_FIELDS)

    while index < total:
        field = CONFIG_FIELDS[index]
        default = answers.get(field.key) or _default_for(field, existing)
        try:
            answers[field.key] = _prompt_field(field, default, index=index + 1, total=total)
            index += 1
        except GoBack:
            if index == 0:
                console.print("已经是第一项了。")
                continue
            index -= 1
            console.print(f"← 回到 {CONFIG_FIELDS[index].label}")

    return answers


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
