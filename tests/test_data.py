from express.base.data import get_torrent, from_torrent, search_models, search_exact
from express.base.model import Metadata


def test_torrent_roundtrip():
    meta = Metadata(
        name="demo-model",
        authors="alice",
        emails="a@example.com",
        version="1.2.3",
        tags=["chat", "lbm"],
    )
    torrent = get_torrent(meta)
    restored = from_torrent(torrent, Metadata)
    assert restored.model_dump() == meta.model_dump()


def test_search_models_free_text_and_filters():
    items = [
        Metadata(authors="stupidfish", emails="a@b.c", version="0.1.1", tags=["chat", "no-think"], name="HIVE0.5-6B-1115-sft"),
        Metadata(authors="stupidfish", emails="a@b.c", version="0.1.3", tags=["128k", "chat", "LBM"], name="HIVE0.5-6B-20260317131457-sft"),
        Metadata(authors="bob", emails="x@y.z", version="0.1.0", tags=["MoE"], name="Other-Model"),
    ]
    assert len(search_models(items, query="hive")) == 2
    assert len(search_models(items, query="hive 128k")) == 1
    assert len(search_models(items, tag="LBM")) == 1
    assert len(search_models(items, author="bob")) == 1
    assert len(search_models(items, name="other")) == 1
    assert search_models(items, query="missing-token") == []


def test_search_exact_case_insensitive():
    items = [Metadata(name="Alpha", authors="Bob", version="0.1.0")]
    assert search_exact(items, "authors", "bob", case_sensitive=False)
    assert not search_exact(items, "authors", "bob", case_sensitive=True)
