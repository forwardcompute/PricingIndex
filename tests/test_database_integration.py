import pandas as pd
import importlib

def test_insert_and_query_roundtrip(sample_scrape_df, tmp_path):
    dbmod = importlib.import_module("database_integration")
    url = f"sqlite:///{tmp_path}/hcpi.db"
    db = dbmod.HCPIDatabase(url)

    # Insert prices
    n = db.insert_scraped_prices(sample_scrape_df)
    assert n == len(sample_scrape_df)

    # Duplicate insert should be mostly ignored by unique constraint
    n2 = db.insert_scraped_prices(sample_scrape_df)
    assert n2 == 0 or n2 < len(sample_scrape_df)

    # Insert HCPI result (use a minimal fake result)
    calc = importlib.import_module("index_calculator")
    result = calc.calculate_hcpi(sample_scrape_df, verbose=False)
    hid = db.insert_hcpi_result(result, api_sources_count=2)
    assert isinstance(hid, int)

    # Queries
    latest = db.get_latest_hcpi()
    assert latest and "us_index" in latest

    prov_df = db.get_all_providers_latest()
    assert not prov_df.empty
    assert set(prov_df["provider"]).issubset(set(sample_scrape_df["provider"]))

def test_save_to_database_wrapper(sample_scrape_df, tmp_path, mocker):
    dbmod = importlib.import_module("database_integration")
    calc = importlib.import_module("index_calculator")
    url = f"sqlite:///{tmp_path}/hcpi.db"
    res = calc.calculate_hcpi(sample_scrape_df, verbose=False)
    # Should not raise
    dbmod.save_to_database(sample_scrape_df, res, database_url=url)
