import json
import sys


class FileReader:
    def read(self, file_path):
        with open(file_path, "r") as file:
            return file.read()


class FileWriter:
    def __init__(self, file_path):
        self.file_path = file_path

    def write(self, content):
        with open(self.file_path, "w") as file:
            file.write(content)


class StdoutWriter:
    def write(self, content):
        sys.stdout.write(content)
        sys.stdout.write("\n")


class JsonParser:
    def parse(self, content):
        return json.loads(content)


class JsonlSerializer:
    def serialize(self, data):
        lines = [json.dumps(item) for item in data]
        return "\n".join(lines)


class JsonToJsonlConverter:
    def __init__(self, file_reader, file_writer, json_parser, jsonl_serializer):
        self.file_reader = file_reader
        self.file_writer = file_writer
        self.json_parser = json_parser
        self.jsonl_serializer = jsonl_serializer

    def convert(self, input_file_path, output_file_path):
        content = self.file_reader.read(input_file_path)
        data = self.json_parser.parse(content)
        jsonl_content = self.jsonl_serializer.serialize(data)
        self.file_writer.write(jsonl_content)


