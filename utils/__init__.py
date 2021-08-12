import yaml


def load_symbols(limit):
    with open("symbols.yaml", 'r') as stream:
        symbols = yaml.safe_load(stream)
    return list(symbols['symbols'].keys())[:limit]
