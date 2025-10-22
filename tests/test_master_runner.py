import pandas as pd
import importlib

def test_master_runner_single_cycle(tmp_path, mocker):
    runner = importlib.import_module("master_runner")
    scrapers = importlib.import_module("scrapers_all_providers")
    calc = importlib.import_module("index_calculator")
    dbmod = importlib.import_module("database_integration")

    # Fake scrape df
    df = pd.DataFrame([{
        "provider":"AWS","region":"US-East","gpu_model":"H100","product":"p5.48xlarge","term":"on_demand",
        "price_hourly_usd":10.0,"gpu_count":800,"timestamp":"2025-01-01T12:00:00+00:00",
        "source":"aws_price_list_api","source_url":"u","listing_id":"L1"
    }])

    mocker.patch.object(scrapers, "scrape_all_providers", return_value=df)
    mocker.patch.object(calc, "save_hcpi_results", return_value=("full.json","summary.json"))

    # Use real calculator but quiet
    out_df, out_res = runner.run_single_scrape_and_calculate(
        database_url=f"sqlite:///{tmp_path}/hcpi.db",
        verbose=False
    )
    assert out_df is not None and not out_df.empty
    assert out_res is not None and out_res.get("us_index") is not None
