"""Shell tab completion helpers (Typer/Click built-in completion)."""

from __future__ import annotations

import subprocess
import sys
from shutil import which
from typing import Literal, Optional

import typer

ShellName = Literal["bash", "zsh", "fish"]


def _express_executable() -> str:
    return which("express") or sys.argv[0]


def _run_express_completion_flag(flag: str, shell: Optional[ShellName]) -> int:
    args = [_express_executable(), flag]
    if shell is not None:
        args.append(shell)
    return subprocess.run(args, check=False).returncode


def install_completion(shell: Optional[ShellName] = None) -> None:
    """Install tab completion for *express* (subcommands and ``--`` options)."""
    code = _run_express_completion_flag("--install-completion", shell)
    if code != 0:
        raise typer.Exit(code)
    typer.echo(
        "[dim]Restart the shell or run `source ~/.bashrc` / re-open the terminal. "
        "Zsh: ensure `~/.zfunc` is on your fpath if completion was installed there.[/dim]"
    )


def show_completion(shell: Optional[ShellName] = None) -> None:
    """Print completion script to stdout (for manual installation)."""
    code = _run_express_completion_flag("--show-completion", shell)
    raise typer.Exit(code)
