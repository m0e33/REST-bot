import csv
from typing import List


class CSVWriter:
    def __init__(self) -> None:
        pass

    def write(self, path: str, data, fieldnames: List[str]):

        with open(path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
            writer.writeheader()
            for entry in data:
                writer.writerow(dict((k, entry[k]) for k in fieldnames))
