# tests/conftest.py
import pytest
import pandas as pd
from datetime import datetime, timezone

@pytest.fixture
def sample_scrape_df():
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return pd.DataFrame([
        # US-West
        {"provider":"A","region":"US-West","gpu_model":"H100","product":"x","term":"on_demand",
         "price_hourly_usd":2.10,"gpu_count":3,"timestamp":now,"source":"test","source_url":"-", "listing_id":"aw"},
        {"provider":"B","region":"US-West","gpu_model":"H100","product":"x","term":"on_demand",
         "price_hourly_usd":3.00,"gpu_count":2,"timestamp":now,"source":"test","source_url":"-", "listing_id":"bw"},
        # US-Central
        {"provider":"C","region":"US-Central","gpu_model":"H100","product":"x","term":"on_demand",
         "price_hourly_usd":2.50,"gpu_count":4,"timestamp":now,"source":"test","source_url":"-", "listing_id":"cc"},
        # US-East
        {"provider":"D","region":"US-East","gpu_model":"H100","product":"x","term":"on_demand",
         "price_hourly_usd":2.80,"gpu_count":1,"timestamp":now,"source":"test","source_url":"-", "listing_id":"de"},
    ])
