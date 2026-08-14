"""Tests for engine.data — CSV loading, date parsing/filtering, alignment."""

from __future__ import annotations

import numpy as np
import pytest

from engine.data import (
    _apply_date_filter,
    _parse_month_year,
    _ym_key,
    load_all_returns,
    load_portfolio_csv,
)


def test_parse_month_year():
    assert _parse_month_year("03/2015") == (2015, 3)
    assert _parse_month_year("12/1999") == (1999, 12)


def test_ym_key():
    assert _ym_key("2015-03") == (2015, 3)


def test_apply_date_filter_no_bounds_returns_all():
    dates = [(2020, 1), (2020, 2), (2020, 3)]
    returns = np.array([0.01, 0.02, 0.03])
    out = _apply_date_filter(dates, returns, None, None)
    np.testing.assert_array_equal(out, returns)


def test_apply_date_filter_inclusive_bounds():
    dates = [(2020, 1), (2020, 2), (2020, 3), (2020, 4)]
    returns = np.array([0.01, 0.02, 0.03, 0.04])
    out = _apply_date_filter(dates, returns, "2020-02", "2020-03")
    np.testing.assert_array_equal(out, np.array([0.02, 0.03]))


def test_apply_date_filter_empty_result_raises():
    dates = [(2020, 1), (2020, 2)]
    returns = np.array([0.01, 0.02])
    with pytest.raises(ValueError):
        _apply_date_filter(dates, returns, "2021-01", "2021-12")


def test_load_portfolio_csv_normalises_weights(tmp_path):
    csv_path = tmp_path / "portfolio.csv"
    csv_path.write_text("TICKER,PCT\nMWEQ,0.6\nGOVH,0.6\n")
    weights = load_portfolio_csv(str(csv_path))
    assert weights["MWEQ"] == pytest.approx(0.5)
    assert weights["GOVH"] == pytest.approx(0.5)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_load_all_returns_aligns_to_shortest_history():
    portfolio = {"MWEQ": 0.5, "GOVH": 0.5}
    weights, ret_matrix = load_all_returns(portfolio, use_after_ter=True)
    assert ret_matrix.ndim == 2
    assert ret_matrix.shape[1] == 2
    assert weights.shape == (2,)
    # weights must be in alphabetical ticker order: GOVH, MWEQ
    assert weights[0] == pytest.approx(0.5)
    assert weights[1] == pytest.approx(0.5)


def test_load_all_returns_date_filter_shrinks_history():
    portfolio = {"MWEQ": 1.0}
    _, full = load_all_returns(portfolio, use_after_ter=True)
    _, filtered = load_all_returns(
        portfolio, use_after_ter=True, date_start="2015-01", date_end="2018-12",
    )
    assert filtered.shape[0] < full.shape[0]
    assert filtered.shape[0] <= 48
