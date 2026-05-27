# Use Case — Portfolio Optimization with PSO

## Problem

Given a universe of 10 S&P 500 stocks, find the optimal portfolio weights
that maximize the Sharpe ratio (return / risk). PSO minimizes the negative
Sharpe ratio, which is equivalent to maximizing it.

## Objective function

    f(w) = -Sharpe(w) = -(mean_return(w) - risk_free) / std(w)

where w is a vector of 10 weights (one per asset), constrained to [0, 1].
The weights are normalized internally so they always sum to 1.

## Assets

AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, JNJ, V, PG, META

## Data

Daily closing prices from 2020-01-01 to 2024-12-31 downloaded via yfinance.
Log returns are computed and used to estimate expected return and covariance.

## Strategies compared

V0 Sequential, V1 Threading, V2 Multiprocessing, V3 Async, V4 Vectorized.

## How to run

    python use_case/run_use_case.py

## Results

Saved to use_case/results/ as JSON files, one per strategy.