from pathlib import Path
from typing import Annotated, List, Optional
import os

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
def pull(torrent: str, force: bool = typer.Option(False, "--force", "-f", help="Automatically overwrite local files when local files and cloud hashes do not match.")):
    """
    拉取模型
    :param torrent:
    :param force:
    :return:
    """
    remote.pull(Torrent(torrent), Remote(), force)


@app.command("delete")
@app.command("rm")
def delete(
        torrent: str,
        yes: bool = typer.Option(False, "--yes", "-y", help="Confirm deletion of this torrent's exclusive remote objects."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Preview what would be deleted without removing objects."),
):
    """
    删除远端模型（仅删除该 torrent 独有对象；仍被其他模型引用的 chunk 会保留）
    """
    remote.delete(Torrent(torrent), Remote(), yes=yes, dry_run=dry_run)


@app.command()
def du():
    """
    查询当前存储桶总大小与总文件数量
    """
    remote.du(Remote())


@app.command()
def config(
        show: bool = typer.Option(False, "--show", help="Show current config path and non-secret values."),
):
    """
    交互式配置远端存储（首次使用远端命令时也会自动引导）
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
def ls(torrent: bool = typer.Option(False, "--torrent", "-t", help="Only print torrent ids (script-friendly)")):
    """
    列出远程模型及其 torrent
    """
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
    搜索远程模型并显示对应 torrent。

    示例:
      express search hive
      express search hive 128k
      express search --tag LBM
      express search -a stupidfish -n HIVE0.5
    """
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
    查看模型信息
    :param model_path: 模型路径
    :param diff_with: 对比差异的模型路径，逻辑为Model(diff_with) - Model(model_path)
    :param calc_fp:
    :param calc_er:
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
        args: List[str] = typer.Argument(
            ...,
            help="Format: alias=path [alias=path ...] 'expression' (e.g., A=path/to/m1 B=path/to/m2 'A+B')"
        )
):
    """
    支持多模型复杂的四则运算。
    示例: A=m1.bin B=m2.bin "(A + B) * 0.5"
    """
    if len(args) < 2:
        raise typer.BadParameter("At least one model mapping and a calculation expression must be provided.")
    mappings, expression = args[:-1], args[-1]

    model_map = {
        name: Model(Path(path))
        for item in mappings
        for name, path in [item.split("=", 1)]
    }

    result = evaluate_model_expression(expression, model_map)
