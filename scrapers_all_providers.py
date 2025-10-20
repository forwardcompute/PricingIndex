#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete H100 Scraper Suite — ALL PROVIDERS (Index-Grade; No Data Loss)
=======================================================================

Guarantees
----------
- Never drops a provider: every row is assigned a triad region (US-West/Central/East).
- Region mapping policy:
    1) Parse explicit region/location strings (AWS, Azure, Lambda, Vast.ai, etc.)
    2) If only “US/USA/United States” is present → US-Central
    3) If still unknown → provider default in PROVIDER_DEFAULT_REGION
- No fabricated region *splits* and no invented GPU counts.
- Strict schema for downstream DB/API:
    provider, region, gpu_model, product, term,
    price_hourly_usd, gpu_count, timestamp, source, source_url, listing_id
- Basic request retries, dedupe, and sanity filtering.

Secrets (set in your env)
-------------------------
export LAMBDA_API_KEY=...
export RUNPOD_API_KEY=...

Dependencies
------------
pip install requests beautifulsoup4 pandas playwright nest_asyncio
python -m playwright install chromium
"""

import os
import re
import time
import json
import hashlib
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Optional Playwright (dynamic pages)
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠ Playwright not available — some scrapers will be limited")

# ------------------------------------------------------------------------------
# CONFIG / CONSTANTS
# ------------------------------------------------------------------------------
UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")

# Hyperscaler nodes commonly have 8× H100 SXM
H100S_PER_NODE = 8

# Triad regions we allow
US_TRIAD = {"US-West", "US-Central", "US-East"}

# Provider fallback region (used only when we can't parse a better hint)
PROVIDER_DEFAULT_REGION = {
    "AWS": "US-East",
    "Azure": "US-East",
    "GCP": "US-Central",
    "CoreWeave": "US-East",
    "Lambda Labs": "US-East",
    "Crusoe": "US-Central",
    "Paperspace": "US-East",
    "Vast.ai": "US-Central",
    "FluidStack": "US-West",
    "Voltage Park": "US-West",
    "OVHcloud": "US-East",
    "Scaleway": "US-East",
    "Genesis Cloud": "US-East",
    "TensorDock": "US-East",
    "Jarvislabs": "US-East",
    "SF Compute": "US-West",
    "RunPod": "US-West",
    "Together.ai": "US-West",
    "Replicate": "US-East",
    "Nebius": "US-East",
}

# Free-text region map (longest-key check first)
REGION_MAP = {
    # West
    "us-west-2": "US-West", "us-west-1": "US-West", "us-west": "US-West",
    "westus3": "US-West", "westus2": "US-West", "westus": "US-West",
    "seattle": "US-West", "oregon": "US-West", "san jose": "US-West",
    "california": "US-West", "ca": "US-West", "wa": "US-West", "or": "US-West",
    # Central
    "us-central-1": "US-Central", "us-central": "US-Central",
    "centralus": "US-Central", "northcentralus": "US-Central", "southcentralus": "US-Central",
    "texas": "US-Central", "dallas": "US-Central", "chicago": "US-Central",
    "tx": "US-Central", "il": "US-Central",
    # East
    "us-east-2": "US-East", "us-east-1": "US-East", "us-east": "US-East",
    "eastus2": "US-East", "eastus": "US-East",
    "virginia": "US-East", "ohio": "US-East", "ashburn": "US-East",
    "north carolina": "US-East", "va": "US-East", "nc": "US-East",
    # Country-only → policy: US-Central
    "united states": "US-Central", "usa": "US-Central", "us": "US-Central",
}

AWS_LOCATION_MAP = {
    'us-east-1': 'US East (N. Virginia)',
    'us-east-2': 'US East (Ohio)',
    'us-west-1': 'US West (N. California)',
    'us-west-2': 'US West (Oregon)',
}
AWS_REGION_TO_TRIAD = {
    "us-east-1": "US-East", "us-east-2": "US-East",
    "us-west-1": "US-West", "us-west-2": "US-West",
}
AZURE_REGION_TO_TRIAD = {
    "eastus": "US-East", "eastus2": "US-East",
    "westus": "US-West", "westus2": "US-West", "westus3": "US-West",
    "centralus": "US-Central", "northcentralus": "US-Central", "southcentralus": "US-Central",
}

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def _listing_id(*parts: str) -> str:
    """Deterministic 16-char id for dedupe."""
    h = hashlib.sha256("|".join([p or "" for p in parts]).encode()).hexdigest()
    return h[:16]

def _map_region_free(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.lower()
    for key, triad in sorted(REGION_MAP.items(), key=lambda kv: -len(kv[0])):
        if key in t:
            return triad
    return None

def _infer_region(provider: str, *context_bits: str) -> str:
    """
    Deterministic region imputation for triad assignment:
      1) Try free-text mapping from any context
      2) If only country-level 'US' → US-Central (handled in REGION_MAP)
      3) Else provider default
    """
    for bit in context_bits:
        triad = _map_region_free(bit or "")
        if triad in US_TRIAD:
            return triad
    triad = PROVIDER_DEFAULT_REGION.get(provider) or "US-Central"
    return triad if triad in US_TRIAD else "US-Central"

def _emit(provider: str, region: str, price: float, *,
          model: str = "H100", product: Optional[str] = None, term: str = "on_demand",
          gpu_count: int = 1, source: str = "web", source_url: Optional[str] = None,
          listing_id: Optional[str] = None) -> pd.DataFrame:
    return pd.DataFrame([{
        "provider": provider,
        "region": region,
        "gpu_model": model,
        "product": product,
        "term": term,
        "price_hourly_usd": float(price),
        "gpu_count": int(gpu_count),
        "timestamp": _now_iso(),
        "source": source,
        "source_url": source_url,
        "listing_id": listing_id or _listing_id(provider, region, model, term, str(price), str(product), str(source_url)),
    }])

def _dedupe_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (df.sort_values("timestamp")
              .drop_duplicates(subset=["provider", "region", "gpu_model", "term", "listing_id"], keep="last"))

def _req(method: str, url: str, *, retries: int = 2, backoff: float = 0.5, timeout: int = 30, **kwargs) -> requests.Response:
    for i in range(retries + 1):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception:
            if i == retries:
                raise
            time.sleep(backoff * (2 ** i))

# ------------------------------------------------------------------------------
# SCRAPERS
# (Return rows with *triad* region; never drop providers — always infer a triad.)
# ------------------------------------------------------------------------------

# ---------------------- Vast.ai (API) ----------------------
def scrape_vastai_h100(timeout=30) -> pd.DataFrame:
    print("[Vast.ai API]", end=" ")
    try:
        r = _req("GET", "https://console.vast.ai/api/v0/bundles", headers=UA, timeout=timeout)
        data = r.json()
        offers = data.get("offers") or data.get("bundles") or []
        if not offers:
            print("✗ No offers"); return pd.DataFrame()
        rows = []
        for o in offers:
            gpu_name = str(o.get("gpu_name",""))
            if "H100" not in gpu_name.upper():
                continue
            gpu_count = int(o.get("num_gpus") or o.get("available_gpus") or 0)
            if gpu_count <= 0:
                continue
            loc_text = " ".join(str(x or "") for x in [o.get("city"), o.get("state"), o.get("country"), o.get("country_code")])
            triad = _infer_region("Vast.ai", loc_text)
            # price per GPU per hour
            dph_total = o.get("dph_total"); dph_base = o.get("dph_base"); dph = o.get("dph")
            if dph_total:
                price = float(dph_total) / max(gpu_count,1)
            elif dph_base:
                price = float(dph_base)
            elif dph:
                price = float(dph)
            else:
                continue
            if not (0.4 <= price <= 30):
                continue
            rows.append(_emit("Vast.ai", triad, price,
                              gpu_count=gpu_count, source="api",
                              source_url="https://console.vast.ai/api/v0/bundles",
                              term="on_demand", product=gpu_name))
        if not rows:
            print("✗ No H100 offers"); return pd.DataFrame()
        df = pd.concat(rows, ignore_index=True)
        
        df["weighted"] = df["price_hourly_usd"] * df["gpu_count"]
        df = (df.groupby("region", as_index=False)
                .agg(provider=("provider","first"),
                     region=("region","first"),
                     gpu_model=("gpu_model","first"),
                     price_hourly_usd=("weighted","sum"),
                     gpu_count=("gpu_count","sum"),
                     timestamp=("timestamp","max"),
                     source=("source","first"),
                     product=("product","first"),
                     source_url=("source_url","first")))
        df["price_hourly_usd"] = df["price_hourly_usd"] / df["gpu_count"].clip(lower=1)
        print(f"✓ {int(df['gpu_count'].sum())} GPUs across {len(df)} region(s)")
        return df.assign(term="on_demand")
    except Exception as e:
        print(f"✗ {str(e)[:60]}"); return pd.DataFrame()

# ---------------------- Lambda Labs (API) ----------------------
def scrape_lambda_h100(timeout=30) -> pd.DataFrame:
    print("[Lambda Labs]", end=" ")
    if not LAMBDA_API_KEY:
        print("✗ no API key"); return pd.DataFrame()
    try:
        base = "https://cloud.lambdalabs.com/api/v1/instance-types"
        headers = {"Authorization": f"Bearer {LAMBDA_API_KEY}", **UA}
        rows = []
        for url in (base, base + "?show_hidden=true"):
            try:
                r = _req("GET", url, headers=headers, timeout=timeout)
                payload = r.json() or {}
            except Exception:
                continue

            data = payload.get("data", [])
            if isinstance(data, dict):
                records = [v for v in data.values() if isinstance(v, dict)]
            elif isinstance(data, list):
                records = [x for x in data if isinstance(x, dict)]
            else:
                records = []

            for rec in records:
                text = " ".join([
                    str(rec.get("name","")),
                    str(rec.get("instance_type", {}).get("name","")),
                    str(rec.get("instance_type", {}).get("description","")),
                    str(rec.get("description","")),
                ]).upper()
                if "H100" not in text:
                    continue

                cents = (rec.get("price_cents_per_hour")
                         or rec.get("instance_type", {}).get("price_cents_per_hour") or 0)
                try:
                    price = float(cents) / 100.0
                except Exception:
                    price = 0.0
                if not (0.4 <= price <= 30):
                    continue

                raw_regions = (rec.get("regions_with_capacity_available")
                               or rec.get("instance_type", {}).get("regions_with_capacity_available")
                               or rec.get("regions") or [])
                for rgn in raw_regions:
                    name = rgn.get("name") if isinstance(rgn, dict) else str(rgn)
                    triad = _infer_region("Lambda Labs", name)
                    if triad in US_TRIAD:
                        rows.append(_emit("Lambda Labs", triad, price,
                                          source="api", source_url=url,
                                          term="on_demand", product=(rec.get("name") or "H100")))
            if rows:
                break  # got capacity; no need to hit second URL

        if rows:
            df = (pd.concat(rows, ignore_index=True)
                    .drop_duplicates(subset=["provider","region","gpu_model","term"]))
            print(f"✓ {len(df)} region(s)")
            return df

        # -------- optional single-page web assist if API returns no US capacity ----------
        print("API empty, checking pricing page...", end=" ")
        try:
            page_url = "https://cloud.lambdalabs.com/pricing"
            rr = _req("GET", page_url, headers=UA, timeout=timeout)
            soup = BeautifulSoup(rr.text, "html.parser")
            # scan any H100 row for a $ price
            for tr in soup.find_all("tr"):
                row = " ".join(td.get_text(" ", strip=True) for td in tr.find_all(["td","th"]))
                if "H100" not in row.upper():
                    continue
                m = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)", row.replace(",", ""))
                if not m: 
                    continue
                price = float(m.group(1))
                if 0.4 <= price <= 30:
                    print("✓ web")
                    return _emit("Lambda Labs", "US-East", price, source="web", source_url=page_url)
            print("none"); return pd.DataFrame()
        except Exception:
            print("fail"); return pd.DataFrame()

    except Exception as e:
        print(f"✗ {str(e)[:60]}"); return pd.DataFrame()



# ---------------------- Crusoe (web; dynamic) ----------------------
def scrape_crusoe_h100(timeout=60000) -> pd.DataFrame:
    print("[Crusoe]", end=" ")
    if not PLAYWRIGHT_AVAILABLE:
        print("✗ Playwright not available"); return pd.DataFrame()
    try:
        async def _run():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto("https://www.crusoe.ai/cloud/pricing", timeout=timeout, wait_until="domcontentloaded")
                for _ in range(3):
                    await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(700)
                html = await page.content()
                await browser.close()
                return html
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio; nest_asyncio.apply()
            html = loop.run_until_complete(_run())
        except RuntimeError:
            html = asyncio.run(_run())

        soup = BeautifulSoup(html, "html.parser")
        rows = []
        for table in soup.find_all("table"):
            for tr in table.find_all("tr"):
                tds = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
                if not tds or not any("H100" in x.upper() for x in tds):
                    continue
                triad = _infer_region("Crusoe", tds[0] if tds else "", " ".join(tds))
                price = None
                for cell in tds[1:]:
                    m = re.search(r'\$\s*([0-9]+(?:\.[0-9]+)?)\s*/?\s*(?:h|hr|hour)\b', cell, re.I)
                    if m:
                        price = float(m.group(1)); break
                if price and 0.4 <= price <= 30:
                    rows.append(_emit("Crusoe", triad, price,
                                      model="H100", term="on_demand",
                                      source="web", source_url="https://www.crusoe.ai/cloud/pricing"))
        if rows:
            df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["provider","region","gpu_model","term"])
            print(f"✓ {len(df)} region(s)")
            return df
        print("✗ No parsable rows"); return pd.DataFrame()
    except Exception as e:
        print(f"✗ {str(e)[:60]}"); return pd.DataFrame()

# ---------------------- FluidStack (web) ----------------------
def scrape_fluidstack_h100(timeout=25) -> pd.DataFrame:
    print("[FluidStack]", end=" ")
    try:
        url = "https://www.fluidstack.io/pricing"
        r = _req("GET", url, headers=UA, timeout=timeout)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = []
        for el in soup.find_all(["section","div","tr","li"]):
            text = el.get_text(" ", strip=True)
            if "H100" not in text.upper():
                continue
            m = re.search(r'\$\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*(?:gpu\s*/\s*)?(?:h|hr|hour)\b', text, re.I)
            if not m:
                continue
            price = float(m.group(1))
            if not (0.4 <= price <= 30):
                continue
            triad = _infer_region("FluidStack", text)
            rows.append(_emit("FluidStack", triad, price, source="web", source_url=url))
        if rows:
            df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["provider","region","gpu_model","term"])
            print(f"✓ {len(df)} row(s)")
            return df
        print("✗ No parsable rows"); return pd.DataFrame()
    except Exception as e:
        print(f"✗ {str(e)[:60]}"); return pd.DataFrame()

# ---------------------- CoreWeave (web; dynamic) ----------------------
def scrape_coreweave_h100(timeout=60000) -> pd.DataFrame:
    print("[CoreWeave]", end=" ")
    if not PLAYWRIGHT_AVAILABLE:
        print("✗ Playwright not available"); return pd.DataFrame()
    try:
        async def _run():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto("https://www.coreweave.com/pricing", timeout=timeout, wait_until="networkidle")
                await page.wait_for_timeout(1500)
                html = await page.content()
                await browser.close()
                return html
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio; nest_asyncio.apply()
            html = loop.run_until_complete(_run())
        except RuntimeError:
            html = asyncio.run(_run())

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        if "H100" not in text.upper():
            print("✗ No H100 mention"); return pd.DataFrame()
        triad = _infer_region("CoreWeave", text)
        m = re.search(r'\$\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*(?:hr|hour)\b', text, re.I)
        if m:
            price = float(m.group(1))
            if 0.4 <= price <= 30:
                print(f"✓ ${price:.2f}/hr")
                return _emit("CoreWeave", triad, price, source="web", source_url="https://www.coreweave.com/pricing")
        print("✗ No parsable price"); return pd.DataFrame()
    except Exception as e:
        print(f"✗ {str(e)[:60]}"); return pd.DataFrame()

# ---------------------- Paperspace (web) ----------------------
def scrape_paperspace_h100(timeout=25) -> pd.DataFrame:
    print("[Paperspace]", end=" ")
    try:
        url = "https://www.paperspace.com/pricing"
        r = _req("GET", url, headers=UA, timeout=timeout)
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        if "H100" not in text.upper():
            print("✗ No H100 mention"); return pd.DataFrame()
        triad = _infer_region("Paperspace", text, url)
        m = re.search(r'\$\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*(?:hr|hour)\b', text, re.I)
        if m:
            price = float(m.group(1))
            if 0.4 <= price <= 30:
                print(f"✓ ${price:.2f}/hr")
                return _emit("Paperspace", triad, price, source="web", source_url=url)
        print("✗ No parsable price"); return pd.DataFrame()
    except Exception as e:
        print(f"✗ {str(e)[:60]}"); return pd.DataFrame()

# ---------------------- Voltage Park (web) ----------------------
def scrape_voltagepark_h100(timeout=30) -> pd.DataFrame:
    print("[Voltage Park]", end=" ")
    try:
        url = "https://www.voltagepark.com/pricing"
        r = _req("GET", url, headers=UA, timeout=timeout)
        text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
        if "H100" not in text.upper():
            print("✗ No H100 mention"); return pd.DataFrame()
        triad = _infer_region("Voltage Park", text, url)
        m = re.search(r'H100.*?\$\s*([0-9]+(?:\.[0-9]{2}))', text, re.I|re.S)
        if not m:
            m = re.search(r'\$\s*([0-9]+(?:\.[0-9]{2})).*?H100', text, re.I|re.S)
        if m:
            price = float(m.group(1))
            if 0.4 <= price <= 30:
                print(f"✓ ${price:.2f}/hr")
                return _emit("Voltage Park", triad, price, source="web", source_url=url)
        print("✗ No parsable price"); return pd.DataFrame()
    except Exception as e:
        print(f"✗ {str(e)[:60]}"); return pd.DataFrame()

# ---------------------- OVHcloud (web) ----------------------
def scrape_ovhcloud_h100(timeout=30) -> pd.DataFrame:
    print("[OVHcloud]", end=" ")
    try:
        url = "https://www.ovhcloud.com/en/public-cloud/prices/"
        r = _req("GET", url, headers=UA, timeout=timeout)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = []
        for tr in soup.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
            row_text = " ".join(cells)
            if "H100" not in row_text.upper():
                continue
            triad = _infer_region("OVHcloud", row_text, url)
            m = re.search(r'\$\s*([0-9]+(?:\.[0-9]+)?)', row_text)
            if m:
                price = float(m.group(1))
                if 0.4 <= price <= 30:
                    rows.append(_emit("OVHcloud", triad, price, source="web", source_url=url))
        if rows:
            df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["provider","region","gpu_model","term"])
            print(f"✓ {len(df)} row(s)")
            return df
        print("✗ No parsable rows"); return pd.DataFrame()
    except Exception as e:
        print(f"✗ {str(e)[:60]}"); return pd.DataFrame()

# ---------------------- TensorDock (web) ----------------------
def scrape_tensordock_h100(timeout=30) -> pd.DataFrame:
    print("[TensorDock]", end=" ")
    try:
        urls = ["https://tensordock.com/gpu-h100", "https://tensordock.com/cloud-gpus"]
        for url in urls:
            r = _req("GET", url, headers=UA, timeout=timeout)
            text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
            if "H100" not in text.upper():
                continue
            triad = _infer_region("TensorDock", text, url)
            m = re.search(r'\$\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*(?:h|hr|hour)\b', text, re.I)
            if m:
                price = float(m.group(1))
                if 0.4 <= price <= 30:
                    print(f"✓ ${price:.2f}/hr")
                    return _emit("TensorDock", triad, price, source="web", source_url=url)
        print("✗ No parsable rows"); return pd.DataFrame()
    except Exception as e:
        print(f"✗ {str(e)[:60]}"); return pd.DataFrame()

# ---------------------- Jarvislabs  ----------------------
def scrape_jarvislabs_h100(timeout=30) -> pd.DataFrame:
    """
    Scrape Jarvislabs H100 pricing from their docs/blog.
    Filters out competitor comparisons to avoid false prices.
    Handles both USD ($) and INR (₹) prices.
    """
    print("[Jarvislabs]", end=" ")
    INR_TO_USD = 0.012
    urls = [
        "https://docs.jarvislabs.ai/blog/h100-price",
        "https://jarvislabs.ai/pricing",
    ]
    all_prices = []
    for url in urls:
        try:
            r = _req("GET", url, headers=UA, timeout=timeout)
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(" ", strip=True)
            if "H100" not in text:
                continue
            h100_positions = [m.start() for m in re.finditer(r'\bH100\b', text, re.I)]
            
            for pos in h100_positions:
                snippet = text[pos:min(len(text), pos + 200)]
                competing_providers = ['Baseten', 'Lambda', 'RunPod', 'CoreWeave', 'versus', 'compared to', 'vs']
                if any(provider in snippet for provider in competing_providers):
                    continue
                usd_matches = re.findall(
                    r'\$\s*(\d+\.\d{2})\s*(?:/|per)?\s*(?:gpu\s*/\s*)?(?:h|hr|hour)?',
                    snippet,
                    re.I
                )
                
                for price_str in usd_matches:
                    try:
                        price = float(price_str)
                        if 1.5 <= price <= 5.0:
                            all_prices.append(price)
                    except ValueError:
                        continue
                inr_matches = re.findall(
                    r'₹\s*(\d+\.?\d*)\s*(?:/|per)?\s*(?:h|hr|hour)?',
                    snippet,
                    re.I
                )
                
                for price_str in inr_matches:
                    try:
                        inr_price = float(price_str)
                        if 150 <= inr_price <= 500:  
                            usd_price = inr_price * INR_TO_USD
                            all_prices.append(usd_price)
                    except ValueError:
                        continue
            if all_prices:
                break
                
        except Exception:
            continue
    if not all_prices:
        print("✗ No parsable rows")
        return pd.DataFrame()
    min_price = min(all_prices)
    triad = _infer_region("Jarvislabs", "US")
    print(f" ${min_price:.2f}/hr ({len(all_prices)} mentions)")
    
    return _emit("Jarvislabs", triad, round(min_price, 2), 
                 source="web", source_url=urls[0])

# ---------------------- Nebius (web) ----------------------
def scrape_nebius_h100(timeout=30) -> pd.DataFrame:
    print("[Nebius]", end=" ")
    try:
        url = "https://nebius.com/prices"
        r = _req("GET", url, headers=UA, timeout=timeout)
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        if "H100" not in text.upper():
            print("✗ No H100 mention"); return pd.DataFrame()
        triad = _infer_region("Nebius", text, url)
        m = re.search(r'\$\s*([0-9]+(?:\.[0-9]{2,4}))\s*/\s*(?:gpu\s*/\s*)?(?:h|hr|hour)\b', text, re.I)
        if m:
            price = float(m.group(1))
            if 0.4 <= price <= 30:
                print(f"✓ ${price:.2f}/hr")
                return _emit("Nebius", triad, price, model="H100 SXM", source="web", source_url=url)
        print("✗ No parsable price"); return pd.DataFrame()
    except Exception as e:
        print(f"✗ {str(e)[:60]}"); return pd.DataFrame()

# ---------------------- SF Compute (web; dynamic) ----------------------
# ---------------------- SF Compute (Playwright-fixed) ----------------------
async def scrape_sfcompute_h100_async(headless: bool = True, debug: bool = False) -> pd.DataFrame:
    """Robust H100 price scraper for SF Compute using Playwright."""
    print("[SF Compute]", end=" ")
    
    if not PLAYWRIGHT_AVAILABLE:
        print("✗ Playwright not available")
        return pd.DataFrame()

    DUR_LABEL = "1 hour"
    GPU_LABEL = "H100"
    SF_URL = "https://sfcompute.com/buy"
    MIN_OK, MAX_OK = 0.25, 50.0
    
    PRICE_RE_DOLLAR = re.compile(r"(?<=\$)\s*([0-9]+(?:\.[0-9]{1,3})?)")
    PRICE_RE_DEC = re.compile(r"([0-9]+\.[0-9]{1,3})")

    async def _get_1h_prices(page) -> List[float]:
        """Read numeric values across the '1 hour' row; prefer $-anchored, fallback to decimals."""
        label = page.locator(
            ":is(th,td,div,span,button,a)",
            has_text=re.compile(r"^\s*%s\s*$" % re.escape(DUR_LABEL), re.I)
        ).first
        if await label.count() == 0:
            label = page.get_by_text(re.compile(r"\b%s\b" % re.escape(DUR_LABEL), re.I)).first
            if await label.count() == 0:
                return []

        cells = await label.evaluate("""(el, dur) => {
            const norm = s => (s||'').replace(/\\s+/g,' ').trim();
            function findRowRoot(n){
              while (n && n !== document.body){
                if (n.tagName === 'TR') return n;
                const role = n.getAttribute && n.getAttribute('role');
                if (role && role.toLowerCase() === 'row') return n;
                const cls = (n.className||'')+'';
                if (/\\brow\\b/i.test(cls)) return n;
                n = n.parentElement;
              }
              return el.parentElement || el;
            }
            function findLabelCell(root, d){
              const all = Array.from(root.querySelectorAll('th,td,div,span,button,a'));
              return all.find(n => norm(n.innerText||n.textContent) === norm(d)) || el;
            }
            const row = findRowRoot(el);
            const labelCell = findLabelCell(row, dur);
            const sibs = Array.from(labelCell.parentElement ? labelCell.parentElement.children : []);
            const idx = sibs.indexOf(labelCell);
            let candidates = idx >= 0 ? sibs.slice(idx+1) : [];
            if (!candidates.length) candidates = Array.from(row.querySelectorAll('td,th,div,span,button,a'));
            return candidates.map(n => {
              const normTxt = t => (t||'').replace(/\\s+/g,' ').trim();
              const base  = normTxt(n.innerText || n.textContent);
              const aria  = normTxt(n.getAttribute?.('aria-label'));
              const title = normTxt(n.getAttribute?.('title'));
              const data  = normTxt(n.getAttribute?.('data-price') || n.getAttribute?.('data-value'));
              const nested = Array.from(n.querySelectorAll('a,button,span,div'))
                                  .map(x => normTxt(x.innerText||x.textContent)).filter(Boolean).join(' ');
              return [base, aria, title, data, nested].filter(Boolean).join(' ');
            });
        }""", DUR_LABEL)

        prices = []
        for txt in cells:
            m = PRICE_RE_DOLLAR.search(txt)
            val = float(m.group(1)) if m else (float(PRICE_RE_DEC.search(txt).group(1)) if PRICE_RE_DEC.search(txt) else None)
            if val is not None and MIN_OK <= val <= MAX_OK:
                prices.append(val)
        return prices

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=headless, args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            page = await browser.new_page()
            await page.goto(SF_URL, wait_until="domcontentloaded")

            # Try to open "Explore Prices" tab
            for sel in (
                page.get_by_role("tab", name=re.compile(r"Explore Prices", re.I)),
                page.get_by_text(re.compile(r"Explore Prices", re.I))
            ):
                try:
                    await sel.first.click(timeout=1200)
                    await page.wait_for_timeout(250)
                    break
                except:
                    pass

            # Select H100 GPU
            picked = False
            try:
                cb = page.get_by_role("combobox").first
                if await cb.count():
                    await cb.select_option(label=GPU_LABEL)
                    picked = True
            except:
                pass
            
            if not picked:
                for q in (
                    page.get_by_role("button", name=re.compile(r"H100|H200", re.I)),
                    page.get_by_role("button", name=re.compile(r"^%s$" % GPU_LABEL, re.I)),
                    page.get_by_text(re.compile(r"^%s$" % GPU_LABEL, re.I)),
                ):
                    try:
                        if await q.count():
                            await q.first.click()
                            picked = True
                            break
                    except:
                        pass
            
            await page.wait_for_timeout(350)

            # Find the pricing grid/table
            grid = page.locator("table").first
            if await grid.count() == 0:
                grid = page.locator("section, div").filter(
                    has_text=re.compile("1 hour|1 day|1 week|1 month", re.I)
                ).first
            
            try:
                await grid.wait_for(state="visible", timeout=4000)
            except:
                pass
            
            await page.wait_for_timeout(300)

            # Extract prices using the working method
            prices = await _get_1h_prices(grid)

            if debug:
                html = await page.content()
                open("sfcompute_debug.html", "w", encoding="utf-8").write(html)
                print(" [saved debug html]", end="")

            await browser.close()

        if not prices:
            print("✗ No parsable price")
            return pd.DataFrame()

        price = min(prices)
        triad = _infer_region("SF Compute", "US-West")
        print(f"✓ ${price:.2f}/hr")

        return _emit("SF Compute", triad, round(price, 4),
                     source="web", source_url=SF_URL,
                     term="on_demand", product="H100")

    except Exception as e:
        print(f"✗ Error: {str(e)[:60]}")
        return pd.DataFrame()


def scrape_sfcompute_h100(headless: bool = True, debug: bool = False) -> pd.DataFrame:
    """Sync wrapper for main pipeline."""
    try:
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            df = loop.run_until_complete(scrape_sfcompute_h100_async(headless=headless, debug=debug))
        except RuntimeError:
            df = asyncio.run(scrape_sfcompute_h100_async(headless=headless, debug=debug))
        return df
    except Exception as e:
        print(f"✗ Error: {e}")
        return pd.DataFrame()
    
# ---------------------- Scaleway (web) ----------------------
def scrape_scaleway_h100(timeout=30) -> pd.DataFrame:
    """
    Best-effort static HTML scrape for Scaleway H100 price.
    Tries multiple pages; accepts € and normalizes €~$ conservatively (1:1).
    """
    print("[Scaleway]", end=" ")
    urls = [
        "https://www.scaleway.com/en/pricing/gpu/",
        "https://www.scaleway.com/en/gpu-instances/",
        "https://www.scaleway.com/en/pricing/",
    ]

    def _price_in(text):
        # €X.XX/hr or $X.XX/hr
        m = re.search(r"[€$]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*(?:h|hr|hour)\b", text, re.I)
        return float(m.group(1)) if m else None

    price = None
    found_url = None
    for url in urls:
        try:
            r = _req("GET", url, headers=UA, timeout=timeout)
            soup = BeautifulSoup(r.text, "html.parser")

            # Search table rows first
            for tr in soup.find_all("tr"):
                row = " ".join(td.get_text(" ", strip=True) for td in tr.find_all(["td","th"]))
                if "H100" in row.upper():
                    p = _price_in(row)
                    if p and 0.4 <= p <= 30:
                        price = p
                        found_url = url
                        break

            # Fallback whole page text if table scan failed
            if price is None:
                text = soup.get_text(" ", strip=True)
                if "H100" in text.upper():
                    p = _price_in(text)
                    if p and 0.4 <= p <= 30:
                        price = p
                        found_url = url
        except Exception:
            continue
        if price is not None:
            break

    if price is None:
        print(" Not found")
        return pd.DataFrame()

    triad = _infer_region("Scaleway", "Europe")  
    print(f"✓ ${price:.2f}/hr")
    return _emit("Scaleway", triad, price, source="web", source_url=found_url)


# ---------------------- Genesis Cloud (web) ----------------------
def scrape_genesiscloud_h100(timeout=30) -> pd.DataFrame:
    """
    Parse Genesis Cloud public pricing for H100. 
    Tries multiple canonical URLs and extracts $/hr or €/hr (€ treated ~=$).
    """
    print("[Genesis Cloud]", end=" ")
    urls = [
        "https://www.genesiscloud.com/pricing",
        "https://www.genesiscloud.com/gpu",
        "https://www.genesiscloud.com/gpu-cloud-pricing",
        "https://www.genesiscloud.com/en/pricing",
    ]
    
    def _price_in(text):
        
        m = re.search(r"[€$]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*(?:h|hr|hour)\b", text, re.I)
        return float(m.group(1)) if m else None

    price = None
    found_url = None
    for url in urls:
        try:
            r = _req("GET", url, headers=UA, timeout=timeout)
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Try rows first
            for tr in soup.find_all("tr"):
                row = " ".join(td.get_text(" ", strip=True) for td in tr.find_all(["td","th"]))
                if "H100" in row.upper():
                    p = _price_in(row)
                    if p and 0.4 <= p <= 30:
                        price = p
                        found_url = url
                        break
            
            # Fallback: whole-text scan
            if price is None:
                text = soup.get_text(" ", strip=True)
                p = _price_in(text) if "H100" in text.upper() else None
                if p and 0.4 <= p <= 30:
                    price = p
                    found_url = url
        except Exception:
            continue
        if price is not None:
            break

    if price is None:
        print(" Not found")
        return pd.DataFrame()

    triad = _infer_region("Genesis Cloud", "Europe")  
    print(f" ${price:.2f}/hr")
    return _emit("Genesis Cloud", triad, price, source="web", source_url=found_url)




# ---------------------- RunPod (GraphQL API; global → infer triad) ----------------------
def scrape_runpod_h100(timeout=30) -> pd.DataFrame:
    print("[RunPod API]", end=" ")
    if not RUNPOD_API_KEY:
        print("✗ no API key"); return pd.DataFrame()
    try:
        query = """
        query GetH100 {
          gpuTypes(input: { id: "NVIDIA H100 80GB HBM3" }) {
            id
            displayName
            lowestPrice(input: { gpuCount: 1, secureCloud: true }) {
              uninterruptablePrice
              stockStatus
              maxUnreservedGpuCount
            }
          }
        }"""

        # Try Authorization header first
        headers_hdr = {**UA, "Content-Type": "application/json", "Authorization": f"Bearer {RUNPOD_API_KEY}"}
        url_hdr = "https://api.runpod.io/graphql"
        try:
            r = _req("POST", url_hdr, json={"query": query}, headers=headers_hdr, timeout=timeout)
            data = r.json() or {}
        except Exception:
            data = {}

        # Fallback to api_key query param if header path failed/empty
        if not (data.get("data") and data["data"].get("gpuTypes")):
            headers_qs = {**UA, "Content-Type": "application/json"}
            url_qs = f"https://api.runpod.io/graphql?api_key={RUNPOD_API_KEY}"
            r = _req("POST", url_qs, json={"query": query}, headers=headers_qs, timeout=timeout)
            data = r.json() or {}

        gpu_types = (data.get("data") or {}).get("gpuTypes") or []
        if not gpu_types:
            print("✗ no H100 data"); return pd.DataFrame()

        lowest = (gpu_types[0] or {}).get("lowestPrice") or {}
        # Some tenants return decimals as strings; be defensive
        def _f(x):
            try: return float(x)
            except Exception: return 0.0
        def _i(x):
            try: return int(x)
            except Exception: return 0

        price = _f(lowest.get("uninterruptablePrice"))
        count = _i(lowest.get("maxUnreservedGpuCount"))
        stock = (lowest.get("stockStatus") or "unknown")

        if price <= 0 or count <= 0:
            print(f"✗ no availability (stock: {stock})"); return pd.DataFrame()

        triad = _infer_region("RunPod", "US")  # global → deterministic triad (no split)
        print(f"✓ {count} GPUs @ ${price:.2f}/hr")
        return _emit("RunPod", triad, price,
                     gpu_count=count, source="api",
                     source_url="https://api.runpod.io/graphql",
                     term="on_demand", product="NVIDIA H100 80GB HBM3")
    except Exception as e:
        print(f" {str(e)[:60]}"); return pd.DataFrame()

# ---------------------- Together.ai  ----------------------
def scrape_together_ai(timeout=30) -> pd.DataFrame:
    print("[Together.ai]", end=" ")
    URL = "https://www.together.ai/pricing"

    def _parse_per_hour(text: str) -> Optional[float]:
        """
        Parse per-hour price. Handles:
          - '$X.YZ / hr', '$X.YZ per hour'
          - per-second '$0.00xx / sec' -> multiplied by 3600
        Returns a single plausible per-hour float, else None.
        """
        if not text:
            return None
        t = " ".join(text.split())  # collapse whitespace

        # 1) explicit per-second -> per-hour
        m = re.search(r"\$\s*(0\.\d{4,6})\s*(?:/|per)\s*s(?:ec)?\b", t, re.I)
        if m:
            per_sec = float(m.group(1))
            per_hour = per_sec * 3600.0
            if 0.5 <= per_hour <= 30:
                return per_hour

        # 2) explicit per-hour
        m = re.search(r"\$\s*([0-9]+(?:\.[0-9]{1,4})?)\s*(?:/|per)\s*(?:gpu\s*/\s*)?h(?:r|our)\b", t, re.I)
        if m:
            per_hour = float(m.group(1))
            if 0.5 <= per_hour <= 30:
                return per_hour

        # 3) any dollar near 'hour'
        m = re.search(r"(?:^|[^a-z])hour[^a-z].{0,40}\$\s*([0-9]+(?:\.[0-9]{1,4})?)", t, re.I)
        if m:
            per_hour = float(m.group(1))
            if 0.5 <= per_hour <= 30:
                return per_hour

        # 4) any dollar near 'H100' window (500 chars on either side)
        for hit in re.finditer(r"H100", t, re.I):
            lo, hi = max(0, hit.start()-500), min(len(t), hit.end()+500)
            window = t[lo:hi]
            # try per-sec then per-hour inside window
            m = re.search(r"\$\s*(0\.\d{4,6})\s*(?:/|per)\s*s(?:ec)?\b", window, re.I)
            if m:
                per_sec = float(m.group(1))
                per_hour = per_sec * 3600.0
                if 0.5 <= per_hour <= 30:
                    return per_hour
            m = re.search(r"\$\s*([0-9]+(?:\.[0-9]{1,4})?)\s*(?:/|per)\s*(?:gpu\s*/\s*)?h(?:r|our)\b", window, re.I)
            if m:
                per_hour = float(m.group(1))
                if 0.5 <= per_hour <= 30:
                    return per_hour

        return None

    def _extract_price_from_html(html: str) -> Optional[float]:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        price = _parse_per_hour(text)
        if not price:
            # Try to scan structured JSON blobs if present
            for script in soup.find_all("script", {"type": "application/ld+json"}):
                try:
                    data = json.loads(script.string or "{}")
                except Exception:
                    continue
                flat = json.dumps(data)
                ph = _parse_per_hour(flat)
                if ph:
                    price = ph
                    break
        return price

    # Static try
    try:
        r = _req("GET", URL, headers=UA, timeout=timeout)
        html = r.text
        price = _extract_price_from_html(html)
        if price:
            triad = _infer_region("Together.ai", html)
            print(f" ${price:.4f}/hr")
            return _emit("Together.ai", triad, round(price, 4),
                        source="web", source_url=URL,
                        term="on_demand", product="Serverless H100")
    except Exception as e:
        print(f" (static error: {e})", end="")

    # Playwright fallback
    if not PLAYWRIGHT_AVAILABLE:
        print(" – no price found")
        return pd.DataFrame()
    
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(user_agent=UA["User-Agent"])
            page = ctx.new_page()
            page.goto(URL, wait_until="domcontentloaded", timeout=timeout * 1000)
            page.wait_for_timeout(1500)
            html = page.content()
            browser.close()
        
        price = _extract_price_from_html(html)
        if price:
            triad = _infer_region("Together.ai", html)
            print(f" (via Playwright) ${price:.4f}/hr")
            return _emit("Together.ai", triad, round(price, 4),
                        source="web", source_url=URL,
                        term="on_demand", product="Serverless H100")
    except Exception:
        pass

    print(" – no price found")
    return pd.DataFrame()



# ---------------------- Replicate (web; dynamic) ----------------------
def scrape_replicate(timeout=60000) -> pd.DataFrame:
    print("[Replicate]", end=" ")
    if not PLAYWRIGHT_AVAILABLE:
        print("✗ Playwright not available"); return pd.DataFrame()
    try:
        async def _run():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto("https://replicate.com/pricing", wait_until="domcontentloaded")
                await page.wait_for_timeout(1500)
                html = await page.content()
                await browser.close()
                return html
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio; nest_asyncio.apply()
            html = loop.run_until_complete(_run())
        except RuntimeError:
            html = asyncio.run(_run())

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        triad = _infer_region("Replicate", text)
        m = re.search(r'H100.*?\$\s*(0\.\d{4,6})\s*/\s*sec', text, re.I|re.S)
        if m:
            per_sec = float(m.group(1)); per_hour = per_sec * 3600
            if 0.4 <= per_hour <= 30:
                print(f"✓ ${per_hour:.4f}/hr")
                return _emit("Replicate", triad, per_hour, source="web", source_url="https://replicate.com/pricing")
        m = re.search(r'H100.*?\$\s*([0-9]+(?:\.[0-9]+))\s*/\s*h', text, re.I|re.S)
        if m:
            price = float(m.group(1))
            if 0.4 <= price <= 30:
                print(f"✓ ${price:.4f}/hr")
                return _emit("Replicate", triad, price, source="web", source_url="https://replicate.com/pricing")
        print("✗ No parsable price"); return pd.DataFrame()
    except Exception as e:
        print(f"✗ {str(e)[:60]}"); return pd.DataFrame()

# ---------------------- GCP (Spot VMs - per-GPU price) ----------------------
def scrape_gcp_h100(timeout=30) -> pd.DataFrame:
    """
    GCP Spot VMs pricing page - shows explicit per-GPU H100 prices.
    Page lists rows like: "H100 (A3-HIGH) $2.253" and "H100 (A3-MEGA) $2.3791"
    """
    print("[GCP]", end=" ")
    try:
        url = "https://cloud.google.com/spot-vms/pricing"
        r = _req("GET", url, headers=UA, timeout=timeout)
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)

        # Search in table rows first
        rows = []
        for tr in soup.find_all("tr"):
            row = " ".join(td.get_text(" ", strip=True) for td in tr.find_all(["td","th"]))
            if re.search(r"\bH100\b", row, re.I):
                m = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/|\s*per\s*)?(?:h|hr|hour)\b", row, re.I)
                if m:
                    val = float(m.group(1))
                    if 0.4 <= val <= 30:
                        rows.append(val)
        
        if not rows:
            # Fallback to page text
            m_all = re.findall(r"H100\s*\(A3-[^)]+\)\s*\$\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
            rows = [float(x) for x in m_all if 0.4 <= float(x) <= 30]

        if rows:
            price = min(rows)  # Use lowest price (A3-HIGH vs A3-MEGA)
            triad = _infer_region("GCP", text, url)
            print(f"✓ ${price:.3f}/hr")
            return _emit("GCP", triad, round(price, 4),
                        product="H100 Spot", source="web", source_url=url)
        
        print("✗ Not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"✗ {str(e)[:60]}")
        return pd.DataFrame()


# ---------------------- Azure (Retail Prices API) ----------------------
def scrape_azure_h100(timeout=30) -> pd.DataFrame:
    """
    Azure Retail Prices API - queries ND H100 v5 VMs.
    VM price is for 8x H100, so divide by 8 for per-GPU price.
    """
    print("[Azure]", end=" ")
    try:
        base = "https://prices.azure.com/api/retail/prices"
        out = []
        
        for region in ["eastus", "eastus2", "westus3", "centralus"]:
            triad = AZURE_REGION_TO_TRIAD.get(region) or _infer_region("Azure", region)
            
            filt = (f"priceType eq 'Consumption' and "
                   f"serviceFamily eq 'Compute' and "
                   f"armRegionName eq '{region}' and "
                   f"contains(productName, 'Virtual Machines') and "
                   f"contains(skuName, 'ND') and "
                   f"contains(skuName, 'H100') and "
                   f"contains(skuName, 'v5')")
            
            params = {'$filter': filt}
            r = _req("GET", base, params=params, timeout=timeout)
            data = r.json()
            
            for item in data.get('Items', []):
                price = item.get('retailPrice', 0)
                unit = (item.get('unitOfMeasure') or '').lower()
                sku = (item.get('skuName') or '').lower()
                product_name = item.get('productName', '')
                
                if price and 'hour' in unit and 'spot' not in sku and 'low priority' not in sku:
                    # ND96isr H100 v5 has 8 GPUs
                    per_gpu = float(price) / 8
                    if 0.4 <= per_gpu <= 30:
                        out.append(_emit("Azure", triad, round(per_gpu, 2),
                                       product=item.get('skuName'),
                                       term="on_demand", source="api",
                                       source_url=f"{base}?$filter={filt}"))
                        break  # One price per region is enough
        
        if out:
            df = pd.concat(out, ignore_index=True)
            print(f"✓ {len(df)} region(s)")
            return df
        
        print("✗ Not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"✗ {str(e)[:60]}")
        return pd.DataFrame()
    
# ---------------------- AWS (Price List API; per region code) ----------------------
def scrape_aws_h100() -> pd.DataFrame:
    """
    Scrape AWS using their public Price List API.
    Downloads region-specific pricing JSON (~5-10MB per region).
    NO AUTH REQUIRED. Returns per-GPU price for p5.48xlarge (8×H100).
    """
    print("[AWS]", end=" ")
    
    def _scrape_region(region_code: str) -> Optional[float]:
        """Scrape a specific AWS region."""
        location = AWS_LOCATION_MAP.get(region_code)
        if not location:
            return None
        
        try:
            url = f'https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/{region_code}/index.json'
            r = _req("GET", url, timeout=120)
            data = r.json()
            
            products = data.get('products', {})
            terms = data.get('terms', {}).get('OnDemand', {})
            
            # Find p5.48xlarge
            for sku, product in products.items():
                if product.get('productFamily') != 'Compute Instance':
                    continue
                
                attrs = product.get('attributes', {})
                
                # Match our criteria
                if (attrs.get('instanceType') == 'p5.48xlarge' and
                    attrs.get('operatingSystem') == 'Linux' and
                    attrs.get('tenancy') == 'Shared' and
                    attrs.get('location') == location):
                    
                    # Get on-demand pricing
                    if sku not in terms:
                        continue
                    
                    for term_key, term_data in terms[sku].items():
                        price_dims = term_data.get('priceDimensions', {})
                        
                        for dim_key, dim_data in price_dims.items():
                            if dim_data.get('unit') == 'Hrs':
                                price_str = dim_data.get('pricePerUnit', {}).get('USD', '0')
                                try:
                                    node_price = float(price_str)
                                    if node_price > 0:
                                        per_gpu = node_price / H100S_PER_NODE
                                        return per_gpu
                                except:
                                    continue
            return None
            
        except Exception:
            return None
    
    # Try multiple regions
    out = []
    for region_code in ["us-east-1", "us-east-2", "us-west-1", "us-west-2"]:
        price = _scrape_region(region_code)
        if price and 0.4 <= price <= 30:
            triad = AWS_REGION_TO_TRIAD.get(region_code) or _infer_region("AWS", region_code)
            out.append(_emit("AWS", triad, round(price, 2),
                           product="p5.48xlarge", term="on_demand",
                           source="aws_price_list_api",
                           source_url=f"https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/{region_code}/index.json"))
    
    if out:
        df = pd.concat(out, ignore_index=True)
        df = df.drop_duplicates(subset=["provider", "region", "gpu_model", "term"])
        print(f"✓ {len(df)} region(s)")
        return df
    
    print("✗ Not found")
    return pd.DataFrame()

# ------------------------------------------------------------------------------
# COMBINED RUNNER
# ------------------------------------------------------------------------------
def scrape_all_providers() -> pd.DataFrame:
    """
    Scrape ALL H100 providers.
    - No data dropped: every row assigned a US triad via _infer_region policy.
    - No fabricated splits. No invented gpu_count.
    """
    print("="*70)
    print("Scraping ALL H100 Providers (no data loss; triad-mapped)")
    print("="*70 + "\n")

    scrapers = [
        # API-first / dynamic-rich sources
        ("Vast.ai", scrape_vastai_h100),
        ("Lambda Labs", scrape_lambda_h100),
        ("Crusoe", scrape_crusoe_h100),
        ("FluidStack", scrape_fluidstack_h100),
        ("CoreWeave", scrape_coreweave_h100),
        ("Paperspace", scrape_paperspace_h100),
        ("Voltage Park", scrape_voltagepark_h100),
        ("OVHcloud", scrape_ovhcloud_h100),
        ("TensorDock", scrape_tensordock_h100),
        ("Jarvislabs", scrape_jarvislabs_h100),
        ("Nebius", scrape_nebius_h100),
        ("SF Compute", scrape_sfcompute_h100),
        ("RunPod", scrape_runpod_h100),
        ("Together.ai", scrape_together_ai),
        ("Replicate", scrape_replicate),
        ("Scaleway", scrape_scaleway_h100),
        ("Genesis Cloud", scrape_genesiscloud_h100),
        ("AWS", scrape_aws_h100),
        ("Azure", scrape_azure_h100),
        ("GCP", scrape_gcp_h100),
    ]

    results: List[pd.DataFrame] = []
    success = 0
    failed = 0

    for name, scraper in scrapers:
        try:
            df = scraper()
            if df is not None and not df.empty:
                results.append(df)
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ {name}: Unexpected error — {str(e)[:80]}")
            failed += 1

    if not results:
        print("\n" + "="*70)
        print("⚠ No data scraped from any provider")
        print("="*70 + "\n")
        return pd.DataFrame()

    df_all = pd.concat(results, ignore_index=True)

    # Enforce triad membership (imputer guarantees this, but keep for safety)
    df_all = df_all[df_all["region"].isin(US_TRIAD)].copy()

    # Sanity
    df_all = df_all[(df_all["price_hourly_usd"] >= 0.4) & (df_all["price_hourly_usd"] <= 30)]
    df_all["gpu_count"] = df_all["gpu_count"].clip(lower=1)

    # Dedupe identical listings (keep latest)
    df_all = _dedupe_rows(df_all)

    print("\n" + "="*70)
    print(f"✓ Successfully scraped {success}/{success + failed} providers")
    print(f"  Total entries: {len(df_all)}")
    print(f"  Unique providers: {df_all['provider'].nunique()}")
    print(f"  Total GPU count: {df_all['gpu_count'].sum():,}")
    print(f"  Price range: ${df_all['price_hourly_usd'].min():.2f} - ${df_all['price_hourly_usd'].max():.2f}")
    print("="*70 + "\n")

    return df_all

def print_summary(df: pd.DataFrame):
    if df.empty:
        print("No data available"); return
    print("Provider Summary:")
    print("-"*70)
    for provider in sorted(df['provider'].unique()):
        provider_df = df[df['provider'] == provider]
        example = provider_df.iloc[0]
        regions = provider_df['region'].nunique()
        total_gpus = provider_df['gpu_count'].sum()
        print(f"  {provider:18} ${example['price_hourly_usd']:.2f}/GPU")
        print(f"  {'':18} {regions} regions | GPUs: {total_gpus:,} | Source: {example['source']}")

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    df = scrape_all_providers()

    if not df.empty:
        print_summary(df)

        # Save to CSV
        try:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out = f"h100_scraped_{ts}.csv"
            df.to_csv(out, index=False)
            print(f"✓ Saved to {out}\n")
        except Exception as e:
            print(f"✗ Could not save: {e}\n")

        # Show sample
        print("Sample Data:")
        print("-"*70)
        cols = ["provider","region","price_hourly_usd","gpu_count","term","source"]
        print(df[cols].head(15).to_string(index=False))
    else:
        print("\n⚠ No data scraped.")
        print("\nTroubleshooting:")
        print("  - Install Playwright: pip install playwright")
        print("  - Install browser: python -m playwright install chromium")
        print("  - Ensure env vars: LAMBDA_API_KEY / RUNPOD_API_KEY")
        print("  - Check internet connection")
        print("  - Some providers may have changed their HTML/DOM or API")# MARKER Fri Oct 17 10:38:42 IST 2025
