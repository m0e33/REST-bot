import yaml


def load_symbols(limit):
    with open("symbols.yaml", 'r') as stream:
        symbols = yaml.safe_load(stream)

    if limit is None:
        return list(symbols['symbols'].keys())
    else:
        return list(symbols['symbols'].keys())[:limit]
