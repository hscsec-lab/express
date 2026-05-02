from pathlib import Path
from typing import Annotated, List

import typer

from express.base import remote
from express.base.data import Torrent
from express.functions.edit_file import edit_file
from express.base.model import Model, _info, evaluate_model_expression
from express.base.remote import Remote
from express.functions.view_model import view_model

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
def view(model_path: Path, diff_with: Path = None, calc_fp: bool = False, calc_er: bool = False):
    """
    查看模型信息
    :param model_path: 模型路径
    :param diff_with: 对比差异的模型路径，逻辑为Model(diff_with) - Model(model_path)
    :return:
    """
    model = Model(model_path)
    if diff_with:
        target_model = Model(diff_with)
        view_model(target_model - model,calc_fp,calc_er)
    else:
        view_model(model, calc_fp, calc_er)


@app.command()
def compute(
        args: List[str] = typer.Argument(..., help="格式: 别名=路径 [别名=路径 ...] '表达式'")
):
    """
    支持多模型复杂的四则运算。
    示例: A=m1.bin B=m2.bin "(A + B) * 0.5"
    """
    if len(args) < 2:
        raise typer.BadParameter("需提供至少一个模型映射和计算表达式。")

    mappings, expression = args[:-1], args[-1]

    model_map = {
        name: Model(Path(path))
        for item in mappings
        for name, path in [item.split("=", 1)]
    }

    result = evaluate_model_expression(expression, model_map)
