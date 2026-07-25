from pathlib import Path
from typing import List, Optional
import os

import typer

app = typer.Typer(
    name="express",
    no_args_is_help=True
)


@app.command()
def edit(torrent: str, remote_file_name: str):
    """
    Edit a remote file online.
    """
    from express.base.client import Remote
    from express.base.data import Torrent
    from express.functions.edit_file import edit_file

    edit_file(Remote(), Torrent(torrent), remote_file_name)


@app.command()
def push(model_path: Path):
    """
    Push a local model to remote storage.
    """
    from express.base import remote
    from express.base.client import Remote
    from express.base.model import Model

    model: Model = Model(path=model_path)
    remote.push(model, Remote())


@app.command()
def pull(torrent: str, force: bool = typer.Option(False, "--force", "-f", help="Overwrite local files when local and remote hashes do not match.")):
    """
    Pull a remote model by torrent.
    """
    from express.base import remote
    from express.base.client import Remote
    from express.base.data import Torrent

    remote.pull(Torrent(torrent), Remote(), force)


@app.command("delete")
@app.command("rm")
def delete(
        torrent: str,
        yes: bool = typer.Option(False, "--yes", "-y", help="Confirm deletion of this torrent's exclusive remote objects."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Preview what would be deleted without removing objects."),
):
    """
    Delete a remote model. Only removes the torrent index and exclusive chunks;
    chunks still referenced by other models are kept.
    """
    from express.base import remote
    from express.base.client import Remote
    from express.base.data import Torrent

    remote.delete(Torrent(torrent), Remote(), yes=yes, dry_run=dry_run)


@app.command()
def du():
    """
    Show total object count and size of the configured bucket.
    """
    from express.base import remote
    from express.base.client import Remote

    remote.du(Remote())


@app.command()
def config(
        show: bool = typer.Option(False, "--show", help="Show current config path and non-secret values."),
):
    """
    Interactively configure remote storage (also prompted on first remote use).
    """
    from express.base.config import (
        apply_config_to_env,
        config_path,
        ensure_remote_config,
        load_config,
        missing_required_keys,
    )

    if show:
        data = load_config()
        apply_config_to_env(data)
        typer.echo(f"Config file: {config_path()}")
        typer.echo(f"Exists: {config_path().is_file()}")
        for key in ("S3_AK", "S3_ENDPOINT", "S3_BUCKET", "LOCAL_WORKDIR"):
            value = os.getenv(key)
            typer.echo(f"{key}={value if value else '(unset)'}")
        typer.echo(f"S3_SK={'******' if os.getenv('S3_SK') else '(unset)'}")
        missing = missing_required_keys()
        if missing:
            typer.echo(f"Missing: {', '.join(missing)}")
        return

    ensure_remote_config(force_interactive=True)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def info(torrent: str, ctx: typer.Context):
    """
    Show torrent metadata.
    """
    from express.base.data import Torrent
    from express.base.model import _info

    _info(Torrent(torrent), fields=[arg[2:] for arg in ctx.args if arg.startswith('--')])


@app.command()
def create(model_path: Path, authors="default_author", emails="default@email.com", version="0.1.0", tags=None,
           name=None):
    """
    Create model metadata.
    """
    from express.base.model import Model

    if not name:
        name = model_path.name
    model = Model(Path(name))
    model.create_metadata(authors=authors,
                          emails=emails,
                          version=version,
                          tags=tags,
                          name=name)


@app.command()
def init(model_path: Path, authors="default_author", emails="default@email.com", version="0.1.0", tags=None,
         name=None):
    """
    Initialize model metadata (alias of create).
    """
    create(model_path=model_path,
           authors=authors,
           emails=emails,
           version=version,
           tags=tags,
           name=name)


@app.command()
def clear(model_path: Path):
    """
    Remove local model metadata and index files.
    """
    from express.base.model import Model

    model = Model(model_path)
    model.remove_metadata()
    model.remove_index_file()


@app.command()
def ls(torrent: bool = typer.Option(False, "--torrent", "-t", help="Only print torrent ids (script-friendly)")):
    """
    List remote models and their torrents.
    """
    from express.base import remote
    from express.base.client import Remote

    remote.ls(remote=Remote(), torrent=torrent)


@app.command()
def search(
        query: Optional[List[str]] = typer.Argument(None, help="Free text; multiple words are AND-matched"),
        name: Optional[str] = typer.Option(None, "--name", "-n", help="Filter by model name (substring)"),
        tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by tag (substring)"),
        author: Optional[str] = typer.Option(None, "--author", "-a", help="Filter by author (substring)"),
        version: Optional[str] = typer.Option(None, "--version", "-V", help="Filter by version (substring)"),
):
    """
    Search remote models and show matching torrents.

    Examples:
      express search hive
      express search hive 128k
      express search --tag LBM
      express search -a stupidfish -n HIVE0.5
    """
    from express.base import remote
    from express.base.client import Remote

    remote.search(
        remote=Remote(),
        query=" ".join(query) if query else None,
        name=name,
        tag=tag,
        author=author,
        version=version,
    )


@app.command()
def view(
        model_path: Path = typer.Argument(..., help="Path to the model file"),
        diff_with: Optional[Path] = typer.Option(None, "--diff", "-d", help="Path to the model for comparison"),
        calc_fp: bool = typer.Option(False, "--fp", help="Whether to calculate the Singular Value Fingerprint (FP)"),
        calc_er: bool = typer.Option(False, "--er", help="Whether to calculate the Effective Rank (ER)"),
):
    """
    View model structure and optional diagnostics.
    """
    from express.base.model import Model
    from express.functions.view_model import view_model

    model = Model(model_path)
    if diff_with:
        target_model = Model(diff_with)
        view_model(target_model - model, calc_fp, calc_er)
    else:
        view_model(model, calc_fp, calc_er)


@app.command()
def compute(
        args: List[str] = typer.Argument(
            ...,
            help="Format: alias=path [alias=path ...] 'expression' (e.g., A=path/to/m1 B=path/to/m2 'A+B')"
        )
):
    """
    Evaluate multi-model arithmetic expressions.

    Example: A=m1.bin B=m2.bin "(A + B) * 0.5"
    """
    from express.base.model import Model, evaluate_model_expression

    if len(args) < 2:
        raise typer.BadParameter("At least one model mapping and a calculation expression must be provided.")
    mappings, expression = args[:-1], args[-1]

    model_map = {
        name: Model(Path(path))
        for item in mappings
        for name, path in [item.split("=", 1)]
    }

    evaluate_model_expression(expression, model_map)
