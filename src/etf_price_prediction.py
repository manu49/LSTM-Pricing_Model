import os
import warnings
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def fetch_etf_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} from {start_date} to {end_date}.")
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    df = df.dropna()
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['return'] = df['Adj Close'].pct_change()
    df['ma10'] = df['Adj Close'].rolling(window=10).mean()
    df['ma21'] = df['Adj Close'].rolling(window=21).mean()
    df['volatility'] = df['return'].rolling(window=10).std()
    df['volume_change'] = df['Volume'].pct_change()
    df = df.dropna()
    return df


def create_sequences(values: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(n_steps, len(values)):
        X.append(values[i - n_steps:i])
        y.append(values[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model(n_steps: int, n_features: int, units: int = 50, dropout: float = 0.2, lr: float = 1e-3) -> Sequential:
    model = Sequential(
        [
            LSTM(units, input_shape=(n_steps, n_features), return_sequences=False),
            Dropout(dropout),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model


def train_with_cv(X: np.ndarray, y: np.ndarray, params: dict, splits: int = 3) -> dict:
    tscv = TimeSeriesSplit(n_splits=splits)
    losses = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_lstm_model(X_train.shape[1], X_train.shape[2], **params)
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            verbose=0,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=1e-4)],
        )
        loss = model.evaluate(X_val, y_val, verbose=0)
        losses.append(loss)
        print(f"  Fold {fold}/{splits}: validation loss = {loss:.6f}")

    average_loss = np.mean(losses)
    print(f"  Average validation loss = {average_loss:.6f}\n")
    return {'params': params, 'val_loss': average_loss}


def grid_search_hyperparameters(X: np.ndarray, y: np.ndarray, parameter_grid: list[dict]) -> dict:
    best = None
    print("Starting hyperparameter grid search...")
    for params in parameter_grid:
        print(f"Testing params: {params}")
        result = train_with_cv(X, y, params)
        if best is None or result['val_loss'] < best['val_loss']:
            best = result
    print(f"Best params: {best['params']} with validation loss {best['val_loss']:.6f}\n")
    return best['params']


def build_feature_matrix(df: pd.DataFrame, scaler: MinMaxScaler = None) -> tuple[np.ndarray, MinMaxScaler]:
    features = df[['Adj Close', 'return', 'ma10', 'ma21', 'volatility', 'volume_change']].copy()
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        values = scaler.fit_transform(features)
    else:
        values = scaler.transform(features)
    return values, scaler


def invert_scaled_values(scaler: MinMaxScaler, scaled_values: np.ndarray) -> np.ndarray:
    dummy = np.zeros((scaled_values.shape[0], scaler.n_features_in_))
    dummy[:, 0] = scaled_values.flatten()
    inverted = scaler.inverse_transform(dummy)
    return inverted[:, 0]


def evaluate_test_set(model: Sequential, X_test: np.ndarray, y_test: np.ndarray, scaler: MinMaxScaler) -> dict:
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = invert_scaled_values(scaler, y_pred_scaled)
    y_true = invert_scaled_values(scaler, y_test.reshape(-1, 1))

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
    }


def backtest_strategy(dates: pd.DatetimeIndex, prices: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(index=dates, data={'price': prices, 'prediction': predictions})
    df['price_return'] = df['price'].pct_change()
    df['signal'] = 0
    df.loc[df.index[1:], 'signal'] = (df['prediction'].iloc[1:].values > df['price'].iloc[:-1].values).astype(int)
    df['strategy_return'] = df['signal'] * df['price_return']
    df['cum_market'] = (1 + df['price_return'].fillna(0)).cumprod()
    df['cum_strategy'] = (1 + df['strategy_return'].fillna(0)).cumprod()
    df['direction'] = (df['price'].diff() > 0).astype(int)
    df['pred_direction'] = (df['prediction'].diff() > 0).astype(int)
    df['direction_accuracy'] = (df['direction'] == df['pred_direction']).astype(int)
    return df


def plot_results(dates: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray, ticker: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label='Actual Adj Close', linewidth=1.3)
    plt.plot(dates, y_pred, label='Predicted Adj Close', linewidth=1.2)
    plt.title(f'{ticker} Price Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_for_ticker(ticker: str, test_months: int = 3) -> None:
    today = pd.Timestamp.today().normalize()
    end_date = today + timedelta(days=1)
    start_date = today - pd.DateOffset(years=5, months=test_months)
    train_end_date = today - pd.DateOffset(months=test_months)

    print(f"\n=== {ticker} ===")
    print(f"Fetching data from {start_date.date()} to {today.date()}...")
    df = fetch_etf_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    df_features = add_technical_features(df)

    train_df = df_features[df_features.index < train_end_date]
    test_df = df_features[df_features.index >= train_end_date]
    if len(test_df) < 20:
        raise ValueError("Not enough test data for the last period.")

    train_values, scaler = build_feature_matrix(train_df)
    test_values, _ = build_feature_matrix(test_df, scaler=scaler)

    n_steps = 60
    X_train, y_train = create_sequences(train_values, n_steps)
    X_test, y_test = create_sequences(test_values, n_steps)
    test_dates = test_df.index[n_steps:]

    parameter_grid = [
        {'units': 32, 'dropout': 0.1, 'lr': 1e-3, 'batch_size': 32, 'epochs': 25},
        {'units': 50, 'dropout': 0.2, 'lr': 1e-3, 'batch_size': 32, 'epochs': 25},
        {'units': 50, 'dropout': 0.2, 'lr': 5e-4, 'batch_size': 64, 'epochs': 30},
    ]

    best_params = grid_search_hyperparameters(X_train, y_train, parameter_grid)
    model = build_lstm_model(X_train.shape[1], X_train.shape[2], **best_params)
    model.fit(
        X_train,
        y_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        validation_split=0.1,
        verbose=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=1e-4)],
    )

    results = evaluate_test_set(model, X_test, y_test, scaler)
    print(f"Test RMSE: {results['rmse']:.4f}")
    print(f"Test MAE: {results['mae']:.4f}")
    print(f"Test MAPE: {results['mape']:.2f}%")

    backtest_df = backtest_strategy(test_dates, results['y_true'], results['y_pred'])
    direction_acc = backtest_df['direction_accuracy'].mean() * 100
    market_return = backtest_df['cum_market'].iloc[-1] - 1
    strategy_return = backtest_df['cum_strategy'].iloc[-1] - 1

    print(f"Direction accuracy: {direction_acc:.2f}%")
    print(f"Buy & hold return over test window: {market_return:.2%}")
    print(f"Simple strategy return over test window: {strategy_return:.2%}\n")

    plot_results(test_dates, results['y_true'], results['y_pred'], ticker)

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    backtest_path = os.path.join(output_dir, f'{ticker}_backtest.csv')
    backtest_df.to_csv(backtest_path)
    print(f"Saved backtest details to {backtest_path}")


def main() -> None:
    tickers = ['LQD', 'HYG', 'BKLN']
    for ticker in tickers:
        run_for_ticker(ticker)


if __name__ == '__main__':
    main()
