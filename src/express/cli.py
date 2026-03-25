from pathlib import Path

import typer

from src.express import remote
from src.express.data import Torrent
from src.express.model import Model, _info
from src.express.remote import Remote

app = typer.Typer(
    name="express",
    no_args_is_help=True
)


@app.command()
def push(model_path: Path):
    """
    推送模型
    :param model_path:
    :return:
    """
    model: Model = Model(path=model_path)
    remote.push(model, Remote())


@app.command()
def pull(torrent: str, force=False):
    """
    拉取模型
    :param torrent:
    :param force:
    :return:
    """
    remote.pull(Torrent(torrent), Remote(), force)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def info(torrent: str, ctx: typer.Context):
    """
    查看torrent信息
    :param torrent:
    :param ctx:
    :return:
    """
    _info(Torrent(torrent), fields=[arg[2:] for arg in ctx.args if arg.startswith('--')])


@app.command()
def create(model_path: Path, authors="default_author", emails="default@email.com", version="0.1.0", tags=None,
           name=None):
    """
    创建模型
    :param model_path:
    :param authors:
    :param emails:
    :param version:
    :param tags:
    :param name:
    :return:
    """
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
    初始化模型
    :param model_path:
    :param authors:
    :param emails:
    :param version:
    :param tags:
    :param name:
    :return:
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
    清除模型元数据
    :param model_path:
    :return:
    """
    model = Model(model_path)
    model.remove_metadata()
    model.remove_index_file()

@app.command()
def ls(torrent: bool = False):
    """
    获取远程模型列表
    :param torrent:
    :return:
    """
    remote.ls(remote=Remote(), torrent=torrent)


@app.command()
def search(field: str, value: str):
    """
    搜索远程模型
    :param field:
    :param value:
    :return:
    """
    remote.search(remote=Remote(), field=field, value=value)
