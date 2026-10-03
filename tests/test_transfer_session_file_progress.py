from express.base.progress import TransferSession


def test_blob_finish_does_not_advance_overall_until_complete_file():
    with TransferSession("Push", total_files=2) as session:
        session.on_start("model.safetensors", 100)
        session.on_finish(advance_overall=False)
        session.on_start("model.safetensors", 50)
        session.on_finish(advance_overall=False)
        assert session._files_done == 0
        session.complete_file()
        assert session._files_done == 1
