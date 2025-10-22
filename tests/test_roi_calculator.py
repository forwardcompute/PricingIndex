import math
import pandas as pd
import pytest

from roi_calculator import (
    calculate_total_cost,
    calculate_basic_roi,
    calculate_utilisation_adjusted_roi,
    print_roi_summary,
)

@pytest.fixture
def sample_df():
    # Minimal, clean sample (no zeros to avoid div-by-zero in ROI)
    return pd.DataFrame([
        {"provider": "A", "region": "US-West",   "price_hourly_usd": 2.00, "gpu_count": 1},
        {"provider": "B", "region": "US-Central","price_hourly_usd": 4.00, "gpu_count": 2},
        {"provider": "C", "region": "US-East",   "price_hourly_usd": 1.25, "gpu_count": 3},
    ])

def test_calculate_total_cost_adds_columns(sample_df):
    out = calculate_total_cost(sample_df, duration_hours=5)
    assert "total_cost_usd" in out.columns
    assert "duration_hours" in out.columns
    # Row-wise checks
    assert out.loc[0, "total_cost_usd"] == pytest.approx(2.00 * 1 * 5)
    assert out.loc[1, "total_cost_usd"] == pytest.approx(4.00 * 2 * 5)
    assert out.loc[2, "total_cost_usd"] == pytest.approx(1.25 * 3 * 5)

def test_calculate_total_cost_requires_columns():
    with pytest.raises(ValueError):
        calculate_total_cost(pd.DataFrame([{"price_hourly_usd": 1.0}]))
    with pytest.raises(ValueError):
        calculate_total_cost(pd.DataFrame([{"gpu_count": 1}]))

def test_calculate_basic_roi(sample_df):
    out = calculate_basic_roi(sample_df)
    assert "roi_basic" in out.columns
    assert out.loc[0, "roi_basic"] == pytest.approx(1 / 2.00)
    assert out.loc[1, "roi_basic"] == pytest.approx(1 / 4.00)
    assert out.loc[2, "roi_basic"] == pytest.approx(1 / 1.25)

def test_calculate_basic_roi_requires_price():
    with pytest.raises(ValueError):
        calculate_basic_roi(pd.DataFrame([{"provider": "X"}]))

def test_utilisation_adjusted_roi_absent_column(sample_df, capsys):
    out = calculate_utilisation_adjusted_roi(sample_df)
    # Should add column with None and print a warning
    assert "roi_utilisation_adj" in out.columns
    assert out["roi_utilisation_adj"].isna().all()
    captured = capsys.readouterr().out
    assert "No 'utilisation_rate' column found" in captured

def test_utilisation_adjusted_roi_with_column(sample_df):
    df = sample_df.copy()
    df["utilisation_rate"] = [0.9, 0.5, 0.8]  # 90%, 50%, 80%
    out = calculate_utilisation_adjusted_roi(df)
    assert out.loc[0, "roi_utilisation_adj"] == pytest.approx(0.9 / 2.00)
    assert out.loc[1, "roi_utilisation_adj"] == pytest.approx(0.5 / 4.00)
    assert out.loc[2, "roi_utilisation_adj"] == pytest.approx(0.8 / 1.25)

def test_print_roi_summary_does_not_crash(sample_df, capsys):
    # Compose full pipeline and ensure printing works
    out = calculate_total_cost(sample_df, duration_hours=1)
    out = calculate_basic_roi(out)
    print_roi_summary(out, top_n=3)
    text = capsys.readouterr().out
    assert "Top Providers by ROI" in text
    # Should include provider names and formatted fields
    assert "A" in text and "B" in text and "C" in text
    assert "/hr" in text
