from pathlib import Path

import typer

from express import remote
from express.data import Torrent
from express.model import Model, get_metadata
from express.remote import Remote

app = typer.Typer(
    name="express",
    no_args_is_help=True
)

@app.command
def push(model_path: Path):
    model: Model = Model(path = model_path)
    remote.push(model,Remote())

@app.command
def pull(torrent: Torrent,force=False):
    remote.pull(torrent,Remote(),force)

@app.command
def info(torrent: Torrent):
    get_metadata(torrent)