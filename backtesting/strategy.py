from typing import Dict, List, NamedTuple
from dataclasses import dataclass
import logging
from collections import namedtuple

logging.getLogger().setLevel(logging.INFO)
State = namedtuple('State', 'date ticks preds')

@dataclass
class Position:
    symbol: str
    open_price: float
    volume: float
    prediction: float
    open_date: str
    open_fee: float
    close_fee: float = 0
    close_date: str = ""
    close_price: float = 0
    open: bool = True

class Strategy:
    def __init__(self, k: int, symbols: List[str]):
        self._k = k
        self._symbols = symbols
        self._positions: Dict[str, List[Position]] = {symbol: [] for symbol in self._symbols}
        self._state: NamedTuple

    def finish(self):
        # close all open positions
        for symbol, positions in self._positions.items():
            if positions[-1].open:
                positions[-1].open = False
                positions[-1].close_price = self._state.ticks[symbol]

        self._calculate_profit()


    def on_tick_update(self, date, ticks: Dict[str, float], preds: Dict[str, float]):
        self._state = State(date, ticks, preds)

        # sort dictionary by prediction value
        sorted_preds = {symbol: pred for symbol, pred in sorted(preds.items(), key=lambda symbol_pred_mapping: symbol_pred_mapping[1], reverse=True)}

        # split into buy symbols and sell symbols
        buy_symbols = list(sorted_preds.keys())[:self._k]
        sell_symbols = list(sorted_preds.keys())[self._k:]

        for symbol in buy_symbols:
            # if this is the first position or last position already closed - open new position
            if not self._positions[symbol] or not self._positions[symbol][-1].open:
                logging.info(f"Open position for {symbol}")
                price = ticks[symbol]
                pred = preds[symbol]
                open_fee = price * 0.015
                new_pos = Position(symbol, price, 3, pred, date, open_fee)
                self._positions[symbol].append(new_pos)
            else:
                # there is already an open position
                continue

        for symbol in sell_symbols:
            # if there is an open position - close position
            if self._positions[symbol] and self._positions[symbol][-1].open:
                logging.info(f"Close open position for {symbol}")
                pos = self._positions[symbol][-1]
                pos.open = False
                pos.close_date = date
                pos.close_fee = ticks[symbol] * 0.025
                pos.close_price = ticks[symbol]
            else:
                # there is no open position for this symbol, thus we can not close a position
                continue

    def _calculate_profit(self):
        brutto_profit = {symbol: 0 for symbol in self._symbols}
        fees = {symbol: 0 for symbol in self._symbols}
        for symbol, positions in self._positions.items():
            brutto_profit[symbol] += sum(map(lambda pos: pos.close_price - pos.open_price, positions))
            fees[symbol] += sum(map(lambda pos: pos.open_fee + pos.close_fee, positions))

        netto_profit = {symbol: brutto_profit[symbol] - fees[symbol] for symbol in self._symbols}
        print("------------------------RESULT------------------------")
        print("Brutto Profit per Symbol")
        for symbol, br_profit in brutto_profit.items():
            print("\t{:<11} | {:<4}".format(symbol, br_profit))

        print("\nNetto Profit per Symbol")
        for symbol, ne_profit in netto_profit.items():
            print("\t{:<11} | {:<4}".format(symbol, ne_profit))

        print(f"\nOver all Brutto Profit: {sum(brutto_profit.values())}")
        print(f"Over all Netto Profit: {sum(brutto_profit.values()) - sum(fees.values())}")

        print(f"\nNumber of positions per symbol")
        for symbol, positions in self._positions.items():
            print("\t{:11} | {:<4}".format(symbol, len(positions)))

        print(f"Number of transactions: {sum([len(positions) for _, positions in self._positions.items()]) * 2}")

        print("-------------------------------------------------------")