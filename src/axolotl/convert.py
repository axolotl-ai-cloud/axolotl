"""Module containing File Reader, File Writer, Json Parser, and Jsonl Serializer classes"""


import json
import sys


class FileReader:
    """
    Reads a file and returns its contents as a string
    """

    def read(self, file_path):
        with open(file_path, encoding="utf-8") as file:
            return file.read()


class FileWriter:
    """
    Writes a string to a file
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def write(self, content):
        with open(self.file_path, "w", encoding="utf-8") as file:
            file.write(content)


class StdoutWriter:
    """
    Writes a string to stdout
    """

    def write(self, content):
        sys.stdout.write(content)
        sys.stdout.write("\n")


class JsonParser:
    """
    Parses a string as JSON and returns the result
    """

    def parse(self, content):
        return json.loads(content)


class JsonlSerializer:
    """
    Serializes a list of JSON objects into a JSONL string
    """

    def serialize(self, data):
        lines = [json.dumps(item) for item in data]
        return "\n".join(lines)


class JsonToJsonlConverter:
    """
    Converts a JSON file to JSONL
    """

    def __init__(self, file_reader, file_writer, json_parser, jsonl_serializer):
        self.file_reader = file_reader
        self.file_writer = file_writer
        self.json_parser = json_parser
        self.jsonl_serializer = jsonl_serializer

    def convert(
        self, input_file_path, output_file_path
    ):  # pylint: disable=unused-argument
        content = self.file_reader.read(input_file_path)
        data = self.json_parser.parse(content)
        # data = [r for r in data if r["conversations"]]  # vicuna cleaned has rows with empty conversations
        jsonl_content = self.jsonl_serializer.serialize(data)
        self.file_writer.write(jsonl_content)
