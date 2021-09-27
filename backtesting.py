import pandas as pd
from datetime import datetime
import numpy as np
from backtesting.strategy import *

NUMBER_OF_SYMBOLS = 10
SYMBOLS = [f'symbol_{i}' for i in range(NUMBER_OF_SYMBOLS)]

def _create_df(number_of_symbols: int):
    date_rng = pd.date_range(start='1/1/2021', end='08/30/2021', freq='D')
    df = pd.DataFrame(date_rng, columns=["date"])
    for i in range(number_of_symbols):
        df[SYMBOLS[i]] = np.random.uniform(0, 100, size=(len(date_rng)))
    return df

if __name__=="__main__":
    ticks_df = _create_df(NUMBER_OF_SYMBOLS)
    preds_df = _create_df(NUMBER_OF_SYMBOLS)

    strategy = Strategy(4, SYMBOLS)

    for index, ticks in ticks_df.iterrows():
        date = ticks['date']
        preds = preds_df.iloc[index]

        ticks_per_symbol = {symbol: tick for symbol, tick in ticks.iteritems()}
        preds_per_symbol = {symbol: pred for symbol, pred in preds.iteritems()}

        # pop date column
        ticks_per_symbol.pop("date", None)
        preds_per_symbol.pop("date", None)

        strategy.on_tick_update(date, ticks_per_symbol, preds_per_symbol)

    strategy.finish()