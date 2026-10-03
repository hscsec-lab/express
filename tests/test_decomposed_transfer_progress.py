from unittest.mock import MagicMock

from express.base.progress import TransferAmountColumn, TransferSession


def test_decomposed_file_tracks_parent_and_slots():
    with TransferSession("Push", total_files=1) as session:
        session.begin_decomposed_file("shard.safetensors", 100, slot_count=2)
        assert session._parent_file_id is not None
        assert len(session._slot_ids) == 2

        session.on_slot_start(0, "layer.0.weight", 60)
        session.on_slot_progress(0, 10)
        session.on_slot_progress(0, 20)

        parent = session._task_by_id(session._parent_file_id)
        slot0 = session._task_by_id(session._slot_ids[0])
        assert parent is not None and parent.completed == 30
        assert slot0 is not None and slot0.completed == 30

        session.on_slot_done(0)
        session.finish_decomposed_file()
        assert session._parent_file_id is None
        assert session._slot_ids == []


def test_transfer_amount_column_without_total():
    column = TransferAmountColumn()
    task = MagicMock()
    task.fields = {"unit": "bytes"}
    task.total = 0
    task.completed = 12
    text = column.render(task)
    assert str(text)


def test_decomposed_invalid_slot_is_ignored():
    with TransferSession("Push", total_files=1) as session:
        session.begin_decomposed_file("shard.safetensors", 10, slot_count=1)
        session.on_slot_progress(-1, 5)
        session.on_slot_progress(3, 5)
        session.on_slot_skip(2, 5)


def test_task_by_id_missing_returns_none():
    session = TransferSession("Push", total_files=0)
    session.progress.start()
    assert session._task_by_id(999_999) is None
    session.progress.stop()


def test_decomposed_skip_advances_parent():
    with TransferSession("Push", total_files=1) as session:
        session.begin_decomposed_file("shard.safetensors", 50, slot_count=1)
        session.on_slot_start(0, "bias", 50)
        session.on_slot_skip(0, 50)
        parent = session._task_by_id(session._parent_file_id)
        assert parent is not None and parent.completed == 50
