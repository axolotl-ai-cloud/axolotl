"""This module contains utility methods related to filesystem and storage operations"""

from datetime import datetime, timezone
from os.path import join
from typing import Optional


def gen_timestamp_file(*args, suffix: str, timestamp: Optional[datetime] = None) -> str:
    """Generate a file path with an embedded timestamp.

    Parameters
    ----------
    *args : str
        Folder path elements as separate string arguments.
    suffix : str
        The suffix to be appended to the timestamp in the filename. Should
        include the extension if one is desired
    timestamp : datetime, optional
        The datetime instance used for generating the timestamp.
        Uses the current UTC time if not provided, by default None.

    Returns
    -------
    str
        The generated file path containing the timestamped filename.
    """

    # If timestamp is not provided, use the current UTC time
    derived_timestamp = timestamp if timestamp else datetime.now(timezone.utc)

    # Define the format for the timestamp. 'Z' is only appended if the time is in UTC.
    time_format = "%Y%m%dT%H%M%S"
    time_format += "Z" if derived_timestamp.tzinfo == timezone.utc else ""

    # Use the strftime function to format the datetime object as a string, and join this with the suffix
    timestamp_string = f"{derived_timestamp.strftime(time_format)}{suffix}"

    # Return the file path by joining the provided folder path elements with the timestamped filename
    return join(*args, timestamp_string)
