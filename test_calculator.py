import numpy as np
import pandas as pd
import importlib

def test_weighted_median_and_regional_index():
    calc = importlib.import_module("index_calculator")
    # Simple 1-region order book: prices 6, 7 with quantities 100, 100
    ob = pd.DataFrame({"price":[6.0,7.0],"q":[100,100],"num_providers":[1,1]})
    idx, liq = calc.compute_regional_index(ob, lam=3.0)
    # Symmetric around m_r=6.5; with exponential tilt, index should be near ~6.5
    assert 6.1 < idx < 6.9
    assert liq > 0

def test_end_to_end_indices(sample_scrape_df):
    calc = importlib.import_module("index_calculator")
    res = calc.calculate_hcpi(sample_scrape_df, verbose=False)
    assert res["us_index"] is not None
    # Must have regional indices for regions present
    assert set(res["regional_indices"].keys()).issubset({"US-West","US-Central","US-East"})
    # Category indices present (even if some None)
    assert "Big 3 Hyperscalers" in res["category_indices"]
    # Report does not crash
    txt = calc.format_hcpi_report(res)
    assert "US INDEX" in txt
