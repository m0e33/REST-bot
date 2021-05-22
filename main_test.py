from data.data_store import DataStore
from data.API_adapter import APIAdapter


if __name__ == "__main__":
  data = DataStore()
  data.flush()
  data.get_price_data('AAPL', '2021-01-01', '2021-05-22')