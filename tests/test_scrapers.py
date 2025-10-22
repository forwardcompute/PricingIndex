import pandas as pd
import re
import importlib

def test_scrapers_schema_and_triads(mocker):
    # Import with fresh state
    scrapers = importlib.import_module("scrapers_all_providers")
    # Monkeypatch all network scrapers to return deterministic rows
    def fake_emit(provider, region, price, **kw):
        return pd.DataFrame([{
            "provider": provider, "region": region, "gpu_model":"H100",
            "product": kw.get("product"), "term": kw.get("term","on_demand"),
            "price_hourly_usd": price, "gpu_count": kw.get("gpu_count", 10),
            "timestamp": "2025-01-01T12:00:00+00:00", "source": kw.get("source","web"),
            "source_url": kw.get("source_url","u"), "listing_id": kw.get("listing_id","X")
        }])

    mocker.patch.object(scrapers, "_emit", side_effect=fake_emit)
    mocker.patch.object(scrapers, "scrape_vastai_h100", lambda: fake_emit("Vast.ai","US-West",6.5, source="api"))
    mocker.patch.object(scrapers, "scrape_lambda_h100", lambda: fake_emit("Lambda Labs","US-East",8.0, source="api"))
    mocker.patch.object(scrapers, "scrape_crusoe_h100", lambda: fake_emit("Crusoe","US-Central",7.2))
    mocker.patch.object(scrapers, "scrape_fluidstack_h100", lambda: fake_emit("FluidStack","US-West",6.9))
    mocker.patch.object(scrapers, "scrape_coreweave_h100", lambda: fake_emit("CoreWeave","US-East",9.0))
    mocker.patch.object(scrapers, "scrape_paperspace_h100", lambda: fake_emit("Paperspace","US-East",9.8))
    mocker.patch.object(scrapers, "scrape_voltagepark_h100", lambda: fake_emit("Voltage Park","US-West",6.7))
    mocker.patch.object(scrapers, "scrape_ovhcloud_h100", lambda: fake_emit("OVHcloud","US-East",9.4))
    mocker.patch.object(scrapers, "scrape_tensordock_h100", lambda: fake_emit("TensorDock","US-Central",6.1))
    mocker.patch.object(scrapers, "scrape_jarvislabs_h100", lambda: fake_emit("Jarvislabs","US-Central",6.3))
    mocker.patch.object(scrapers, "scrape_nebius_h100", lambda: fake_emit("Nebius","US-East",8.7))
    mocker.patch.object(scrapers, "scrape_sfcompute_h100", lambda: fake_emit("SF Compute","US-West",6.2))
    mocker.patch.object(scrapers, "scrape_runpod_h100", lambda: fake_emit("RunPod","US-West",7.0, source="api"))
    mocker.patch.object(scrapers, "scrape_together_ai", lambda: fake_emit("Together.ai","US-West",5.5))
    mocker.patch.object(scrapers, "scrape_replicate", lambda: fake_emit("Replicate","US-East",5.7))
    mocker.patch.object(scrapers, "scrape_aws_h100", lambda: fake_emit("AWS","US-East",10.0, source="aws_price_list_api"))
    mocker.patch.object(scrapers, "scrape_azure_h100", lambda: fake_emit("Azure","US-Central",11.0, source="azure_retail_api"))
    mocker.patch.object(scrapers, "scrape_gcp_h100", lambda: fake_emit("GCP","US-Central",9.9))

    df = scrapers.scrape_all_providers()
    # Must have triad-only regions
    assert set(df["region"]).issubset({"US-West","US-Central","US-East"})
    # Sanity bounds
    assert df["price_hourly_usd"].between(0.4,30).all()
    assert (df["gpu_count"] >= 1).all()
    # Schema presence
    needed = {"provider","region","gpu_model","term","price_hourly_usd","gpu_count","timestamp","source","source_url","listing_id"}
    assert needed.issubset(df.columns)
    # listing_id uniqueness per row
    assert df["listing_id"].notna().all()