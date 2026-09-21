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


def test_load_all_returns_aligns_on_common_calendar_months():
    portfolio = {"MWEQ": 0.5, "GOVH": 0.5}
    weights, ret_matrix = load_all_returns(portfolio, use_after_ter=True)
    assert ret_matrix.ndim == 2
    assert ret_matrix.shape[1] == 2
    assert weights.shape == (2,)
    # weights must be in alphabetical ticker order: GOVH, MWEQ
    assert weights[0] == pytest.approx(0.5)
    assert weights[1] == pytest.approx(0.5)

    # Alignment must be by calendar month, not by row position: two
    # tickers whose histories end on different months must not be
    # silently paired row-for-row (that scrambles cross-asset correlation).
    from engine.data import load_returns
    dates_a, ret_a = load_returns("MWEQ", use_after_ter=True)
    dates_b, ret_b = load_returns("GOVH", use_after_ter=True)
    common = sorted(set(dates_a) & set(dates_b))
    assert ret_matrix.shape[0] == len(common)
    lookup_a = dict(zip(dates_a, ret_a))
    lookup_b = dict(zip(dates_b, ret_b))
    expected = np.column_stack([
        [lookup_b[m] for m in common],   # GOVH (alphabetically first)
        [lookup_a[m] for m in common],   # MWEQ
    ])
    np.testing.assert_allclose(ret_matrix, expected)


def test_load_all_returns_correlation_matches_date_aligned_reference():
    # WRDA and HPRD end on different calendar months (HPRD's history is
    # much shorter) — a row-tail alignment would pair unrelated months
    # and destroy the true correlation between the two series.
    from engine.data import load_returns
    portfolio = {"WRDA": 0.5, "HPRD": 0.5}
    weights, ret_matrix = load_all_returns(portfolio, use_after_ter=True)

    dates_h, ret_h = load_returns("HPRD", use_after_ter=True)
    dates_w, ret_w = load_returns("WRDA", use_after_ter=True)
    common = sorted(set(dates_h) & set(dates_w))
    lookup_h = dict(zip(dates_h, ret_h))
    lookup_w = dict(zip(dates_w, ret_w))
    expected_corr = np.corrcoef(
        [lookup_h[m] for m in common], [lookup_w[m] for m in common],
    )[0, 1]

    got_corr = np.corrcoef(ret_matrix.T)[0, 1]
    assert got_corr == pytest.approx(expected_corr, abs=1e-9)
    # Sanity: the true correlation here is well above what a scrambled
    # row-tail alignment would produce (empirically ~0.02).
    assert got_corr > 0.5


def test_load_all_returns_date_filter_shrinks_history():
    portfolio = {"MWEQ": 1.0}
    _, full = load_all_returns(portfolio, use_after_ter=True)
    _, filtered = load_all_returns(
        portfolio, use_after_ter=True, date_start="2015-01", date_end="2018-12",
    )
    assert filtered.shape[0] < full.shape[0]
    assert filtered.shape[0] <= 48


def test_load_portfolios_on_common_window_uses_one_shared_window():
    """Portfolios being RANKED against each other must be measured on the
    same months. Loaded on their own windows, a portfolio built from
    long-history assets is scored over a different period than one holding
    a late-starting asset — so part of any gap is the period, not the
    portfolio, and the tool silently rewards holding assets with more
    history."""
    from engine.data import load_portfolios_on_common_window

    a = {"WEBN": 0.73, "GOVH": 0.27}
    b = {"WEBN": 0.4, "GOVH": 0.2, "XDWS": 0.4}   # XDWS starts much later
    tickers, mat, weights = load_portfolios_on_common_window(
        {"a": a, "b": b}, use_after_ter=True,
    )

    assert tickers == ["GOVH", "WEBN", "XDWS"]
    assert mat.shape == (mat.shape[0], 3)
    # one matrix, therefore one window, for both portfolios
    for name, pf in (("a", a), ("b", b)):
        assert weights[name].shape == (3,)
        assert weights[name].sum() == pytest.approx(1.0)
        for t, w in zip(tickers, weights[name]):
            assert w == pytest.approx(pf.get(t, 0.0))

    # The shared window must be the INTERSECTION — strictly shorter than
    # what portfolio 'a' would get on its own tickers, which is exactly
    # the advantage being removed.
    _, own_a = load_all_returns(a, use_after_ter=True)
    assert mat.shape[0] < own_a.shape[0]


def test_load_portfolios_on_common_window_single_portfolio_matches_load_all_returns():
    """With one portfolio there's nothing to be fair between, so the shared
    window degenerates to that portfolio's own window — the common-window
    loader must not silently narrow it further."""
    from engine.data import load_portfolios_on_common_window

    pf = {"WEBN": 0.73, "GOVH": 0.27}
    tickers, mat, weights = load_portfolios_on_common_window(
        {"only": pf}, use_after_ter=True,
    )
    own_w, own_mat = load_all_returns(pf, use_after_ter=True)
    assert mat.shape == own_mat.shape
    np.testing.assert_allclose(mat, own_mat)
    np.testing.assert_allclose(weights["only"], own_w)
