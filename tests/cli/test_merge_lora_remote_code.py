from axolotl.cli.merge_lora import _copy_remote_code_files


def test_remote_code_copy_keeps_merged_metadata_and_nested_sources(
    tmp_path, monkeypatch
):
    base = tmp_path / "base"
    output = base / "merged"
    (base / "nested").mkdir(parents=True)
    (base / "modeling_model.py").write_text("from .nested.helper import value\n")
    (base / "nested" / "helper.py").write_text("value = 1\n")
    (base / "config.json").write_text('{"vocab_size": 1}\n')
    (base / "model.safetensors").write_bytes(b"base")
    output.mkdir()
    (output / "config.json").write_text('{"vocab_size": 2}\n')
    (output / "tokenizer.json").write_text("merged-tokenizer\n")
    (output / "model.safetensors").write_bytes(b"merged")
    monkeypatch.chdir(tmp_path)
    _copy_remote_code_files(base.resolve(), output.relative_to(tmp_path))
    assert (
        output / "modeling_model.py"
    ).read_text() == "from .nested.helper import value\n"
    assert (output / "nested" / "helper.py").read_text() == "value = 1\n"
    assert (output / "config.json").read_text() == '{"vocab_size": 2}\n'
    assert (output / "tokenizer.json").read_text() == "merged-tokenizer\n"
    assert (output / "model.safetensors").read_bytes() == b"merged"
