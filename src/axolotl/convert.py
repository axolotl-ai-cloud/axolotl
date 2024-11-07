"""Module containing File Reader, File Writer, Json Parser, Csv Parser, and Jsonl Serializer classes"""

import csv
import json
import sys


class FileReader:
    """
    Reads a file and returns its contents as a string
    """

    def read(self, file_path):
        with open(file_path, mode="r", encoding="utf-8") as file:
            return file.read()


class FileWriter:
    """
    Writes a string to a file
    """

    def write(self, content, file_path):
        with open(file_path, mode="w", encoding="utf-8") as file:
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


class CsvParser:
    """
    Reads a CSV file and returns its contents as a list of dictionaries
    """

    def read(self, file_path):
        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return list(reader)


class JsonlSerializer:
    """
    Serializes a list of JSON objects into a JSONL string
    """

    def serialize(self, data):
        lines = [json.dumps(item) for item in data]
        return "\n".join(lines)


class ConverterToJsonl:
    """
    Converts to JSONL
    """

    def __init__(self, parser, writer, jsonl_serializer=JsonlSerializer()):
        self.csv_parser = parser
        self.file_writer = writer
        self.jsonl_serializer = jsonl_serializer

    def convert(self, input_file_path, output_file_path):
        # Read data from the CSV file
        data = self.csv_parser.read(input_file_path)

        # Serialize the data to JSONL format
        jsonl_content = self.jsonl_serializer.serialize(data)

        # Write the JSONL content to the output file
        self.file_writer.write(jsonl_content, output_file_path)
