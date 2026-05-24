# Skills

This repository supports the following capabilities:

- Historical ETF price data ingestion via Yahoo Finance (`yfinance`).
- Feature engineering with technical indicators: moving averages, volatility, returns, and volume changes.
- LSTM time-series modeling for next-day adjusted close price prediction.
- Time series cross-validation and hyperparameter search.
- Backtest analysis using a simple directional strategy.
- Unit tests for key data processing and model-building functions.

## Relevant directories

- `src/`: Python package code for the ETF pricing model.
- `tests/`: Test suite for validating the model and preprocessing logic.
- `output/`: Generated backtest CSV files (created during script execution).
