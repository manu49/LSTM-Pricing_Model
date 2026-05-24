import pandas as pd
import numpy as np
import pytest

from src import etf_price_prediction as model


def make_sample_df(n_days: int = 40) -> pd.DataFrame:
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
    data = {
        'Open': np.linspace(100, 110, n_days),
        'High': np.linspace(101, 111, n_days),
        'Low': np.linspace(99, 109, n_days),
        'Close': np.linspace(100, 110, n_days),
        'Adj Close': np.linspace(100, 110, n_days),
        'Volume': np.linspace(200_000, 300_000, n_days),
    }
    return pd.DataFrame(data, index=dates)


def test_add_technical_features_creates_expected_columns():
    df = make_sample_df()
    result = model.add_technical_features(df)

    assert 'return' in result.columns
    assert 'ma10' in result.columns
    assert 'ma21' in result.columns
    assert 'volatility' in result.columns
    assert 'volume_change' in result.columns
    assert not result.isna().any().any()
    assert result.shape[0] == len(df) - 21


def test_create_sequences_shapes_and_values():
    values = np.arange(30).reshape(15, 2).astype(float)
    n_steps = 5
    X, y = model.create_sequences(values, n_steps)

    assert X.shape == (10, n_steps, 2)
    assert y.shape == (10,)
    assert np.array_equal(X[0], values[0:5])
    assert y[0] == values[5, 0]


def test_build_feature_matrix_scaler_ranges_and_inverse_transform():
    df = make_sample_df()
    df = model.add_technical_features(df)
    values, scaler = model.build_feature_matrix(df)

    assert values.min() >= 0.0
    assert values.max() <= 1.0
    assert values.shape[1] == 6

    sample_scaled = np.array([[0.0], [1.0]])
    inverted = model.invert_scaled_values(scaler, sample_scaled)
    assert inverted.shape == (2,)


def test_build_lstm_model_output_shape():
    n_steps = 10
    n_features = 6
    units = 16
    model_obj = model.build_lstm_model(n_steps, n_features, units=units, dropout=0.1, lr=1e-3)

    assert model_obj.input_shape == (None, n_steps, n_features)
    assert model_obj.output_shape == (None, 1)


def test_fetch_etf_data_uses_yfinance_download(monkeypatch):
    sample_df = make_sample_df(10)

    def fake_download(ticker, start, end, progress):
        assert ticker == 'TEST'
        assert start == '2023-01-01'
        assert end == '2023-01-15'
        assert progress is False
        return sample_df

    monkeypatch.setattr(model.yf, 'download', fake_download)
    fetched = model.fetch_etf_data('TEST', '2023-01-01', '2023-01-15')

    assert isinstance(fetched, pd.DataFrame)
    assert list(fetched.columns) == ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    assert fetched.shape[0] == 10
