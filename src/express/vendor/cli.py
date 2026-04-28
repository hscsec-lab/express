from pathlib import Path
from typing import Annotated

import typer

from express.base import remote
from express.base.data import Torrent
from express.functions.edit_file import edit_file
from express.base.model import Model, _info
from express.base.remote import Remote

app = typer.Typer(
    name="express",
    no_args_is_help=True
)


@app.command()
def edit(torrent: str, remote_file_name: str):
    """
    在线编辑远程文件
    :param torrent:
    :param remote_file_name:
    :return:
    """
    edit_file(Remote(), Torrent(torrent), remote_file_name)


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


@app.command()
def view(model_path: Path, diff_with: Path = None):
    """
    查看模型信息
    :param model_path: 模型路径
    :param diff_with: 对比差异的模型路径，逻辑为Model(diff_with) - Model(model_path)
    :return:
    """
    model = Model(model_path)
    if diff_with:
        target_model = Model(diff_with)
        (target_model - model).view_gui()
    else:
        model.view_gui()
