
from urllib.request import urlopen
import json

def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

url = ("https://financialmodelingprep.com/api/v3/press-releases/AAPL?limit=100&apikey=24bb344efcd23043c520dd9489d6f29c")
print(get_jsonparsed_data(url))
