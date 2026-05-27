"""
Portfolio optimization use case for PSO.
Defines the fitness function (negative Sharpe ratio) and loads
historical price data for 10 S&P 500 stocks via yfinance.
"""

import numpy as np
import yfinance as yf

# ── Configuration ─────────────────────────────────────────────────────────────

TICKERS    = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "JNJ", "V", "PG", "META"]
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"
RISK_FREE  = 0.02

# ── Data loading ──────────────────────────────────────────────────────────────

def load_returns() -> np.ndarray:
    print(f"Downloading data for {TICKERS} from {START_DATE} to {END_DATE}...")
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    prices = data["Close"][TICKERS].dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna().values
    print(f"Data loaded: {log_returns.shape[0]} trading days, {log_returns.shape[1]} assets.")
    return log_returns


# ── Closure-based fitness function (V0, V1, V3, V4) ──────────────────────────

def make_fitness_fn(log_returns: np.ndarray):
    """
    Returns negative Sharpe ratio as fitness function.
    NOT picklable — cannot be used with V2 multiprocessing.
    """
    mean_returns = log_returns.mean(axis=0)
    cov_matrix   = np.cov(log_returns.T)

    def fitness_fn(x: np.ndarray) -> float:
        w = np.abs(x)
        w = w / w.sum()
        portfolio_return = np.dot(w, mean_returns) * 252
        portfolio_vol    = np.sqrt(w @ cov_matrix @ w) * np.sqrt(252)
        if portfolio_vol < 1e-10:
            return 0.0
        return -(portfolio_return - RISK_FREE) / portfolio_vol

    return fitness_fn


# ── Multiprocessing-compatible fitness function (V2) ─────────────────────────
# On Windows, spawned processes do not inherit parent globals.
# The initializer pattern passes data explicitly to each worker process.

def _worker_init(mean_returns_arr, cov_matrix_arr):
    """Initializer for each worker process — sets globals from passed arrays."""
    global _mean_returns, _cov_matrix
    _mean_returns = mean_returns_arr
    _cov_matrix   = cov_matrix_arr


def fitness_fn_mp(x: np.ndarray) -> float:
    """Module-level picklable fitness function for V2 multiprocessing."""
    w = np.abs(x)
    w = w / w.sum()
    portfolio_return = np.dot(w, _mean_returns) * 252
    portfolio_vol    = np.sqrt(w @ _cov_matrix @ w) * np.sqrt(252)
    if portfolio_vol < 1e-10:
        return 0.0
    return -(portfolio_return - RISK_FREE) / portfolio_vol