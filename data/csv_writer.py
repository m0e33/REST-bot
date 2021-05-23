"""Write data to CSV"""
import csv
from typing import List


def write(path: str, data, fieldnames: List[str]):
    """Method for writing a List of Dicts to csv"""

    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow(dict((k, entry[k]) for k in fieldnames))
