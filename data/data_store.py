from data.API_adapter import APIAdapter
from .csv_writer import CSVWriter
import os, shutil

RELEVANT_HIST_FIELDS = ["date", "open", "close", "high", "low", "vwap"]


class DataStore:
    STORAGE_PATH = "./data/storage/"

    def __init__(self) -> None:
        self.api = APIAdapter()
        self.writer = CSVWriter()

    def get_price_data(self, symbol: str, start: str, end: str):
        path = self._get_path(symbol, start, end)
        if self._check_file(path):
            # load file
            pass
        else:
            prices = self.api.get_historical_prices(symbol, start, end)
            self.writer.write(path, prices, RELEVANT_HIST_FIELDS)

    def get_press_release_data():
        pass

    def flush(self):
        for filename in os.listdir(DataStore.STORAGE_PATH):
            file_path = os.path.join(DataStore.STORAGE_PATH, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

    def _check_file(self, name: str):
        return os.path.exists(name)

    def _get_path(self, symbol: str, start: str, end: str, type: str = "csv"):
        return f"{DataStore.STORAGE_PATH}{symbol}_{start}_{end}.{type}"
