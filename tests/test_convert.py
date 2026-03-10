"""Unit tests for src/axolotl/convert.py"""

import json

import pytest

from axolotl.convert import (
    FileReader,
    FileWriter,
    JsonlSerializer,
    JsonParser,
    JsonToJsonlConverter,
    StdoutWriter,
)


class TestJsonParser:
    def test_parse_valid_json_array(self):
        parser = JsonParser()
        result = parser.parse('[{"key": "value"}]')
        assert result == [{"key": "value"}]

    def test_parse_valid_json_object(self):
        parser = JsonParser()
        result = parser.parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_invalid_json_raises(self):
        parser = JsonParser()
        with pytest.raises(json.JSONDecodeError):
            parser.parse("not valid json")


class TestJsonlSerializer:
    def test_serialize_single_item(self):
        serializer = JsonlSerializer()
        result = serializer.serialize([{"a": 1}])
        assert result == '{"a": 1}'

    def test_serialize_multiple_items(self):
        serializer = JsonlSerializer()
        result = serializer.serialize([{"a": 1}, {"b": 2}])
        lines = result.split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}

    def test_serialize_empty_list(self):
        serializer = JsonlSerializer()
        result = serializer.serialize([])
        assert result == ""


class TestFileReaderWriter:
    def test_read_write_roundtrip(self, tmp_path):
        test_file = tmp_path / "test.txt"
        content = '{"hello": "world"}'
        writer = FileWriter(str(test_file))
        writer.write(content)

        reader = FileReader()
        result = reader.read(str(test_file))
        assert result == content


class TestStdoutWriter:
    def test_write_to_stdout(self, capsys):
        writer = StdoutWriter()
        writer.write("hello")
        captured = capsys.readouterr()
        assert captured.out == "hello\n"


class TestJsonToJsonlConverter:
    def test_convert_json_to_jsonl(self, tmp_path):
        input_data = [{"name": "Alice"}, {"name": "Bob"}]
        input_file = tmp_path / "input.json"
        output_file = tmp_path / "output.jsonl"

        input_file.write_text(json.dumps(input_data), encoding="utf-8")

        converter = JsonToJsonlConverter(
            FileReader(), FileWriter(str(output_file)), JsonParser(), JsonlSerializer()
        )
        converter.convert(str(input_file))

        result = output_file.read_text(encoding="utf-8")
        lines = result.split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"name": "Alice"}
        assert json.loads(lines[1]) == {"name": "Bob"}
