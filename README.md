# LSTM-Pricing_Model

This repository contains an LSTM-based ETF price prediction script that fetches historical data from Yahoo Finance, trains on the past 5 years, validates with time-series cross-validation, and backtests on the latest 3 months.

## Files

- `src/etf_price_prediction.py`: Fetches ETF price data for `LQD`, `HYG`, and `BKLN`, trains an LSTM model, performs hyperparameter tuning, evaluates test performance, and runs a simple backtest strategy.
- `tests/test_etf_price_prediction.py`: Unit tests for the model package.
- `requirements.txt`: Python dependencies for the project.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the prediction and backtest script:

```bash
python src/etf_price_prediction.py
```

3. Run unit tests:

```bash
python -m pytest tests/test_etf_price_prediction.py
```

4. Output backtest details will be saved in the `output/` directory.

## Notes

- The model uses the last 5 years of data for training and time-series cross-validation.
- The final test window is the latest 3 months of data.
- Backtesting uses a simple signal: long when the predicted next-day price is higher than today's price, otherwise stay in cash.
