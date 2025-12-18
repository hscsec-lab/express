from pathlib import Path

import typer

from express import remote
from express.data import Torrent
from express.model import Model, get_metadata, _info
from express.remote import Remote

app = typer.Typer(
    name="express",
    no_args_is_help=True
)

@app.command()
def push(model_path: Path):
    model: Model = Model(path = model_path)
    remote.push(model,Remote())

@app.command()
def pull(torrent: str,force=False):
    remote.pull(Torrent(torrent),Remote(),force)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def info(torrent: str, ctx: typer.Context):
    _info(Torrent(torrent), fields=[arg[2:] for arg in ctx.args if arg.startswith('--')])

@app.command()
def create(model_name: str):
    model = Model(Path(model_name))
    model.create_metadata(model_name)

@app.command()
def ls(torrent: bool = False):
    remote.ls(remote=Remote(),torrent=torrent)

@app.command()
def search(field: str,value: str):
    remote.search(remote=Remote(),field=field,value=value)
