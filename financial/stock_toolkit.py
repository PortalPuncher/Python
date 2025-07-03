"""A minimal toolkit for stock analysis.

This module demonstrates a tiny application that combines several
features:

2. Stock screener
4. Social sentiment tracker
6. Backtesting simulator
7. AI-powered stock recommender
8. Income & dividend tracker
9. Educational investment guide

All data is randomly generated to keep this example self-contained.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class Stock:
    symbol: str
    price: float
    pe_ratio: float
    dividend_yield: float
    sector: str


# Sample in-memory "database" of stocks
STOCKS = [
    Stock("AAPL", 150.0, 28.0, 0.006, "Technology"),
    Stock("MSFT", 310.0, 34.0, 0.010, "Technology"),
    Stock("GOOGL", 2800.0, 31.0, 0.000, "Technology"),
    Stock("AMZN", 3300.0, 58.0, 0.000, "Consumer"),
    Stock("TSLA", 700.0, 110.0, 0.000, "Auto"),
]


def screen_stocks(pe_ratio: float | None = None, dividend_yield: float | None = None,
                  sector: str | None = None) -> pd.DataFrame:
    """Return stocks matching given criteria.

    >>> screen_stocks(pe_ratio=30).shape[0] > 0
    True
    >>> screen_stocks(dividend_yield=0.005).shape[0]
    2
    >>> screen_stocks(sector="Auto").iloc[0].symbol
    'TSLA'
    """

    data = pd.DataFrame(STOCKS)
    if pe_ratio is not None:
        data = data[data["pe_ratio"] <= pe_ratio]
    if dividend_yield is not None:
        data = data[data["dividend_yield"] >= dividend_yield]
    if sector is not None:
        data = data[data["sector"] == sector]
    return data.reset_index(drop=True)


_POSITIVE = {"buy", "great", "growth", "bull"}
_NEGATIVE = {"sell", "bad", "bear"}


def social_sentiment(comments: Iterable[str]) -> float:
    """Return simple sentiment score from comments.

    >>> round(social_sentiment(["buy", "bad", "great"]), 2)
    0.33
    >>> social_sentiment(["sell", "sell"]) < 0
    True
    """

    score = 0
    for comment in comments:
        words = comment.lower().split()
        score += sum(word in _POSITIVE for word in words)
        score -= sum(word in _NEGATIVE for word in words)
    if not comments:
        return 0.0
    return score / len(list(comments))


def _random_price_series(symbol: str, periods: int = 100) -> pd.Series:
    rng = np.random.default_rng(abs(hash(symbol)) % 2**32)
    returns = rng.normal(0.001, 0.02, periods)
    return pd.Series(100 * (1 + returns).cumprod())


def backtest_moving_average(symbol: str, short: int = 5, long: int = 20) -> float:
    """Simple moving-average crossover backtest.

    Returns total return of the strategy in percent.

    >>> result = backtest_moving_average("AAPL", 5, 10)
    >>> result > -100
    True
    """

    prices = _random_price_series(symbol)
    short_ma = prices.rolling(window=short).mean()
    long_ma = prices.rolling(window=long).mean()
    position = (short_ma > long_ma).astype(int).shift(1)
    returns = prices.pct_change().fillna(0)
    strategy = (position * returns).cumsum()
    return float(strategy.iloc[-1] * 100)


def recommend_stocks(symbols: list[str]) -> dict[str, float]:
    """Return naive expected returns for symbols.

    >>> recs = recommend_stocks(["AAPL", "MSFT"])
    >>> set(recs).issuperset({"AAPL", "MSFT"})
    True
    """

    data = pd.DataFrame(STOCKS)
    if symbols:
        data = data[data["symbol"].isin(symbols)]
    x = data[["pe_ratio", "dividend_yield"]]
    rng = np.random.default_rng(42)
    target = rng.normal(0.01, 0.05, len(data))
    model = LinearRegression().fit(x, target)
    predictions = model.predict(x)
    return dict(zip(data["symbol"], predictions))


def annual_dividend(symbol: str, shares: int) -> float:
    """Return estimated annual dividend income for given shares."""

    for stock in STOCKS:
        if stock.symbol == symbol:
            return stock.price * stock.dividend_yield * shares
    message = f"Unknown symbol: {symbol}"
    raise ValueError(message)


GUIDE = """
Basic Investment Guide
----------------------
- Diversify your holdings.
- Focus on long-term growth.
- Reinvest dividends.
"""


def get_investment_guide() -> str:
    """Return a simple educational investment guide.

    >>> "long-term" in get_investment_guide()
    True
    """

    return GUIDE


if __name__ == "__main__":
    import doctest

    doctest.testmod()
