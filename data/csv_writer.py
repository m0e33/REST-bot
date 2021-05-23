"""Write data to CSV"""
import csv
from typing import List


def read_csv_to_json_array(path: str, fieldnames: List[str]):
    """Helper method for reading csv rows into array of json objects"""

    json_array = []
    with open(path) as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for rows in reader:
            json_object = {}
            for key in fieldnames:
                json_object[key] = rows[key]
            json_array.append(json_object)

    return json_array


def write_csv(path: str, data, fieldnames: List[str]):
    """Helper method for writing a List of Dicts to csv"""

    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow(dict((k, entry[k]) for k in fieldnames))
