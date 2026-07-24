from pathlib import Path

from express.base.model import Model, Metadata


def test_create_metadata_and_index(tmp_path):
    model_dir = tmp_path / "demo"
    model_dir.mkdir()
    (model_dir / "readme.txt").write_text("hello", encoding="utf-8")

    model = Model(model_dir)
    model.create_metadata(
        authors="tester",
        emails="t@example.com",
        version="0.1.0",
        tags=["unit"],
        name="demo",
    )
    model.write_index_file()

    assert model.is_metadata_file_exists()
    meta = model.get_metadata()
    assert meta.name == "demo"
    assert meta.version == "0.1.0"
    assert any(item.file_name == "readme.txt" for item in model.folder_index.folder_index)
    assert Metadata.from_torrent(meta.get_torrent()).name == "demo"
