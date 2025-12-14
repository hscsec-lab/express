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

@app.command()
def info(torrent: str):
    _info(Torrent(torrent))

@app.command()
def create(model_name: str):
    model = Model(Path(model_name))
    model.create_metadata(model_name)

@app.command()
def ls():
    remote.ls(remote=Remote())