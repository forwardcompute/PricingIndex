#!/usr/bin/env python3
"""
ORNN US H100 Compute Price Index (HCPI) Calculator
==================================================

Replicates the exact methodology from: https://ornn.trade/docs/ornn_index.pdf

Calculates indices for:
- Overall US
- Each region (US-West, US-Central, US-East)
- Provider categories (Big 3, Major GPU Clouds, Serverless, Specialized)

Methodology:
1. Regional Order Books: Aggregate listings by price within each region
2. Regional Median: Calculate liquidity-weighted median price m_r
3. Exponential Weights: φ = exp(-λ * (price - m_r) / m_r), λ = 3
4. Regional Index: I_r = Σ(price × gpu_count × φ) / Σ(gpu_count × φ)
5. Regional Liquidity: G_r = Σ(gpu_count × φ)
6. US Index: I_US = Σ(I_r × G_r) / Σ(G_r)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import json

# Lambda parameter for exponential downweighting (from ORNN paper)
LAMBDA = 3.0

# Expected US regions
US_REGIONS = ['US-West', 'US-Central', 'US-East']

# Provider categories
PROVIDER_CATEGORIES = {
    'Big 3 Hyperscalers': [
        'AWS',
        'GCP',
        'Azure',
        'Google Cloud',
        'Microsoft Azure'
    ],
    'Major GPU Clouds': [
        'CoreWeave',
        'Lambda Labs',
        'Crusoe',
        'RunPod',
        'Paperspace',
        'FluidStack',
        'Nebius'
    ],
    'Serverless': [
        'Together.ai',
        'Replicate'
    ],
    'Specialized': [
        'Jarvislabs',
        'Vast.ai',
        'OVHcloud',
        'Hyperstack',
        'TensorDock',
        'Voltage Park',
        'Scaleway',
        'Genesis Cloud',
        'SF Compute'
    ]
}


def categorize_provider(provider_name: str) -> str:
    """
    Categorize a provider into one of the four categories.
    
    Args:
        provider_name: Name of the provider
    
    Returns:
        Category name or 'Other'
    """
    for category, providers in PROVIDER_CATEGORIES.items():
        for p in providers:
            if p.lower() in provider_name.lower():
                return category
    return 'Other'


def construct_regional_order_books(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Step 1: Construct regional order books.
    
    For each region, aggregate all listings by price and sum GPU counts.
    This creates the order book structure: price -> total_gpu_count
    
    Args:
        df: DataFrame with columns [provider, region, price_hourly_usd, gpu_count]
    
    Returns:
        Dict mapping region -> order_book DataFrame
    """
    order_books = {}
    
    for region in US_REGIONS:
        region_data = df[df['region'] == region].copy()
        
        if region_data.empty:
            continue
        
        # Aggregate by price, summing GPU counts
        order_book = (region_data
                     .groupby('price_hourly_usd')
                     .agg({
                         'gpu_count': 'sum',  # Sum all GPUs at this price
                         'provider': 'count'  # Count providers (for reference)
                     })
                     .reset_index()
                     .rename(columns={
                         'price_hourly_usd': 'price',
                         'gpu_count': 'q',  # Using ORNN notation
                         'provider': 'num_providers'
                     }))
        
        order_books[region] = order_book
    
    return order_books


def calculate_liquidity_weighted_median(prices: np.ndarray, quantities: np.ndarray) -> float:
    """
    Liquidity-weighted median: first price where cumulative qty >= 50% of total.
    Assumes quantities >= 0 and sum > 0.
    """
    if prices.size == 0:
        raise ValueError("Empty price array")
    if quantities.size != prices.size:
        raise ValueError("prices and quantities must be same length")

    # Sort by price
    sort_idx = np.argsort(prices, kind="mergesort")
    p = prices[sort_idx]
    q = quantities[sort_idx].astype(float)

    total = q.sum()
    if total <= 0:
        # fallback to simple median of prices if no liquidity
        return float(np.median(p))

    csum = np.cumsum(q)
    # first index where cumulative >= 50%
    cutoff = 0.5 * total
    idx = np.searchsorted(csum, cutoff, side="left")
    if idx >= p.size:
        idx = p.size - 1
    return float(p[idx])



def compute_regional_index(order_book: pd.DataFrame, lam: float = LAMBDA) -> Tuple[Optional[float], float]:
    """
    Returns (I_r, G_r). If no liquidity after weighting, returns (None, 0.0).
    """
    if order_book.empty:
        return None, 0.0

    # Ensure numeric
    ob = order_book.copy()
    ob["price"] = pd.to_numeric(ob["price"], errors="coerce")
    ob["q"] = pd.to_numeric(ob["q"], errors="coerce").fillna(0)

    ob = ob.dropna(subset=["price"])
    ob = ob[ob["q"] > 0]
    if ob.empty:
        return None, 0.0

    prices = ob["price"].to_numpy(dtype=float)
    quantities = ob["q"].to_numpy(dtype=float)

    # Step 2: liquidity-weighted median
    m_r = calculate_liquidity_weighted_median(prices, quantities)
    if m_r <= 0:
        return None, 0.0

    # Step 3: exponential weights
    phi = np.exp(-lam * (prices - m_r) / m_r)

    # Steps 4–5
    wq = quantities * phi
    denom = wq.sum()
    if denom <= 0:
        return None, 0.0
    Ir = float((prices * wq).sum() / denom)
    Gr = float(denom)
    return Ir, Gr



def compute_us_index(regional_indices: Dict[str, float], 
                     regional_liquidities: Dict[str, float]) -> float:
    """
    Step 6: Aggregate regional indices into US index.
    
    I_US = Σ(I_r × G_r) / Σ(G_r)
    
    Args:
        regional_indices: Dict mapping region -> index value
        regional_liquidities: Dict mapping region -> liquidity value
    
    Returns:
        US aggregate index
    """
    if not regional_indices:
        return None
    
    # Filter out None values
    valid_regions = [(region, idx, regional_liquidities[region]) 
                     for region, idx in regional_indices.items() 
                     if idx is not None]
    
    if not valid_regions:
        return None
    
    # Weighted average by liquidity
    numerator = sum(idx * liq for _, idx, liq in valid_regions)
    denominator = sum(liq for _, _, liq in valid_regions)
    
    if denominator == 0:
        return None
    
    return numerator / denominator


def calculate_category_index(df: pd.DataFrame, category: str, lam: float = LAMBDA) -> Dict:
    """
    Calculate HCPI for a specific provider category.
    
    Args:
        df: Full DataFrame with all providers
        category: Category name (e.g., 'Big 3 Hyperscalers')
        lam: Lambda parameter
    
    Returns:
        Dict with category index results
    """
    # Add category column
    df_copy = df.copy()
    df_copy['category'] = df_copy['provider'].apply(categorize_provider)
    
    # Filter to this category
    df_category = df_copy[df_copy['category'] == category].copy()
    
    if df_category.empty:
        return {
            'category_index': None,
            'regional_indices': {},
            'regional_liquidities': {},
            'providers': [],
            'total_listings': 0
        }
    
    # Calculate using same methodology
    order_books = construct_regional_order_books(df_category)
    
    regional_indices = {}
    regional_liquidities = {}
    
    for region, order_book in order_books.items():
        index, liquidity = compute_regional_index(order_book, lam)
        regional_indices[region] = index
        regional_liquidities[region] = liquidity
    
    category_index = compute_us_index(regional_indices, regional_liquidities)
    
    return {
        'category_index': round(category_index, 4) if category_index else None,
        'regional_indices': {k: round(v, 4) for k, v in regional_indices.items() if v is not None},
        'regional_liquidities': {k: round(v, 2) for k, v in regional_liquidities.items()},
        'providers': sorted(df_category['provider'].unique().tolist()),
        'total_listings': len(df_category),
        'total_gpus': int(df_category['gpu_count'].sum())
    }


def calculate_hcpi(df: pd.DataFrame, lam: float = LAMBDA, verbose: bool = True) -> Dict:
    """
    Calculate the complete HCPI following ORNN methodology.
    Includes overall US, regional, and category-based indices.
    
    Args:
        df: DataFrame with columns [provider, region, price_hourly_usd, gpu_count]
        lam: Lambda parameter (default: 3.0 from ORNN paper)
        verbose: Print calculation steps
    
    Returns:
        Dict with complete HCPI results including category indices
    """
    if verbose:
        print("="*70)
        print("CALCULATING ORNN US H100 COMPUTE PRICE INDEX (HCPI)")
        print("="*70)
        print(f"Lambda parameter: {lam}")
        print(f"Total listings: {len(df)}")
        print(f"Unique providers: {df['provider'].nunique()}")
        print(f"Total GPU count: {df['gpu_count'].sum():,.0f}")
        print()
    
    # Validate and clean data
    df_clean = df[df['region'].isin(US_REGIONS)].copy()
    df_clean = df_clean.dropna(subset=['price_hourly_usd', 'gpu_count'])
    df_clean = df_clean[df_clean['price_hourly_usd'] > 0]
    df_clean = df_clean[df_clean['gpu_count'] > 0]
    
    if df_clean.empty:
        return {
            'us_index': None,
            'error': 'No valid data after cleaning',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    # ===== OVERALL US INDEX =====
    
    # Step 1: Construct regional order books
    order_books = construct_regional_order_books(df_clean)
    
    if verbose:
        print("Regional Order Books:")
        for region, ob in order_books.items():
            print(f"  {region}: {len(ob)} unique prices, {ob['q'].sum():,.0f} total GPUs")
        print()
    
    # Steps 2-5: Calculate regional indices and liquidities
    regional_indices = {}
    regional_liquidities = {}
    regional_details = {}
    
    for region, order_book in order_books.items():
        index, liquidity = compute_regional_index(order_book, lam)
        regional_indices[region] = index
        regional_liquidities[region] = liquidity
        
        # Store details for reporting
        if index is not None:
            regional_details[region] = {
                'index': round(index, 4),
                'liquidity': round(liquidity, 2),
                'num_prices': len(order_book),
                'total_gpus': int(order_book['q'].sum()),
                'num_providers': int(order_book['num_providers'].sum()),
                'price_range': {
                    'min': round(order_book['price'].min(), 2),
                    'max': round(order_book['price'].max(), 2),
                    'median': round(calculate_liquidity_weighted_median(
                        order_book['price'].values, 
                        order_book['q'].values
                    ), 2)
                }
            }
    
    if verbose:
        print("Regional Indices:")
        for region, details in regional_details.items():
            print(f"  {region}: ${details['index']:.4f}/GPU/hr")
            print(f"    Liquidity: {details['liquidity']:.2f}")
            print(f"    Price range: ${details['price_range']['min']:.2f} - ${details['price_range']['max']:.2f}")
            print(f"    Providers: {details['num_providers']}, GPUs: {details['total_gpus']:,}")
        print()
    
    # Step 6: Calculate US aggregate index
    us_index = compute_us_index(regional_indices, regional_liquidities)
    
    if verbose:
        if us_index is not None:
            print(f"US AGGREGATE INDEX: ${us_index:.4f}/GPU/hr")
        else:
            print("US AGGREGATE INDEX: Unable to calculate")
    
    # ===== CATEGORY INDICES =====
    
    if verbose:
        print("\n" + "="*70)
        print("CALCULATING CATEGORY INDICES")
        print("="*70 + "\n")
    
    category_indices = {}
    
    for category in PROVIDER_CATEGORIES.keys():
        category_result = calculate_category_index(df_clean, category, lam)
        category_indices[category] = category_result
        
        if verbose and category_result['category_index'] is not None:
            print(f"{category}:")
            print(f"  Index: ${category_result['category_index']:.4f}/GPU/hr")
            print(f"  Providers: {', '.join(category_result['providers'])}")
            print(f"  Total GPUs: {category_result['total_gpus']:,}")
            if category_result['regional_indices']:
                print(f"  Regional breakdown:")
                for reg, idx in category_result['regional_indices'].items():
                    print(f"    {reg}: ${idx:.4f}/hr")
            print()
    
    if verbose:
        print("="*70)
        print()
    
    # Compile results
    return {
        'us_index': round(us_index, 4) if us_index is not None else None,
        'regional_indices': {k: round(v, 4) for k, v in regional_indices.items() if v is not None},
        'regional_liquidities': {k: round(v, 2) for k, v in regional_liquidities.items()},
        'regional_details': regional_details,
        'category_indices': category_indices,
        'metadata': {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'lambda': lam,
            'methodology': 'ORNN',
            'total_listings': len(df_clean),
            'unique_providers': int(df_clean['provider'].nunique()),
            'total_gpus': int(df_clean['gpu_count'].sum()),
            'regions_covered': list(order_books.keys()),
            'provider_list': sorted(df_clean['provider'].unique().tolist())
        }
    }


def format_hcpi_report(result: Dict) -> str:
    """
    Pretty text report of HCPI results.
    """
    if result.get('error'):
        return f"ERROR: {result['error']}"

    report = []
    report.append("╔══════════════════════════════════════════════════════════════╗")
    report.append("║     ORNN US H100 COMPUTE PRICE INDEX (HCPI)                 ║")
    report.append("╚══════════════════════════════════════════════════════════════╝")
    report.append("")

    # Main index
    us_index = result.get('us_index')
    report.append(f"US INDEX:  ${us_index:.4f} per GPU per hour" if us_index is not None
                  else "US INDEX:  Unable to calculate")
    report.append("")

    # Regional breakdown
    report.append("REGIONAL BREAKDOWN:")
    report.append("-" * 64)
    details = result.get('regional_details', {})
    for region in ['US-West', 'US-Central', 'US-East']:
        d = details.get(region)
        if not d:
            continue
        report.append(f"{region:12} ${d['index']:.4f}/hr")
        report.append(f"{'':12} Liquidity: {d['liquidity']:.2f} | "
                      f"Providers: {d['num_providers']} | GPUs: {d['total_gpus']:,}")
        pr = d['price_range']
        report.append(f"{'':12} Price range: ${pr['min']:.2f} – ${pr['max']:.2f} (median: ${pr['median']:.2f})")
        report.append("")

    # Category indices (optional)
    cats = result.get('category_indices', {})
    if cats:
        report.append("CATEGORY INDICES:")
        report.append("-" * 64)
        for category, c in cats.items():
            ci = c.get('category_index')
            if ci is None:
                continue
            report.append(f"{category}")
            report.append(f"  Index: ${ci:.4f}/hr")
            provs = c.get('providers', [])
            if provs:
                report.append(f"  Providers: {', '.join(provs)}")
            tg = c.get('total_gpus')
            if tg is not None:
                report.append(f"  GPUs: {tg:,}")
            reg_i = c.get('regional_indices', {})
            if reg_i:
                report.append("  Regional breakdown:")
                for reg, val in reg_i.items():
                    report.append(f"    {reg}: ${val:.4f}/hr")
            report.append("")

    # Metadata
    meta = result.get('metadata', {})
    report.append("METADATA:")
    report.append("-" * 64)
    report.append(f"Timestamp:        {meta.get('timestamp', 'N/A')}")
    report.append(f"Total Providers:  {meta.get('unique_providers', 'N/A')}")
    report.append(f"Total Listings:   {meta.get('total_listings', 'N/A')}")
    tg = meta.get('total_gpus')
    report.append(f"Total GPU Count:  {tg:,}" if isinstance(tg, int) else f"Total GPU Count:  {tg}")
    report.append(f"Lambda Parameter: {meta.get('lambda', 'N/A')}")
    providers = meta.get('provider_list', [])
    if providers:
        report.append("")
        report.append("PROVIDERS:")
        report.append("-" * 64)
        for i in range(0, len(providers), 3):
            row = providers[i:i+3]
            report.append("  " + " | ".join(f"{p:20}" for p in row))
    return "\n".join(report)

    
    report = []
    report.append("╔══════════════════════════════════════════════════════════════╗")
    report.append("║     ORNN US H100 COMPUTE PRICE INDEX (HCPI)                ║")
    report.append("╚══════════════════════════════════════════════════════════════╝")
    report.append("")
    
    # Main index
    us_index = result.get('us_index')
    if us_index:
        report.append(f"US INDEX:  ${us_index:.4f} per GPU per hour")
    else:
        report.append("US INDEX:  Unable to calculate")
    report.append("")
    
    # Regional breakdown
    report.append("REGIONAL BREAKDOWN:")
    report.append("-" * 64)
    
    for region in ['US-West', 'US-Central', 'US-East']:
        details = result.get('regional_details', {}).get(region)
        if details:
            report.append(f"{region:12} ${details['index']:.4f}/hr")
            report.append(f"{'':12} Liquidity: {details['liquidity']:.2f} | "
                        f"Providers: {details['num_providers']} | "
                        f"GPUs: {details['total_gpus']:,}")
            report.append(f"{'':12} Price range: ${details['price_range']['min']:.2f} - "
                        f"${details['price_range']['max']:.2f}")
            report.append("")
    
    # Metadata
    meta = result.get('metadata', {})
    report.append("METADATA:")
    report.append("-" * 64)
    report.append(f"Timestamp:        {meta.get('timestamp', 'N/A')}")
    report.append(f"Total Providers:  {meta.get('unique_providers', 'N/A')}")
    report.append(f"Total Listings:   {meta.get('total_listings', 'N/A')}")
    report.append(f"Total GPU Count:  {meta.get('total_gpus', 'N/A'):,}")
    report.append(f"Lambda Parameter: {meta.get('lambda', 'N/A')}")
    report.append("")
    
    # Provider list
    providers = meta.get('provider_list', [])
    if providers:
        report.append("PROVIDERS:")
        report.append("-" * 64)
        for i in range(0, len(providers), 3):
            row = providers[i:i+3]
            report.append("  " + " | ".join(f"{p:20}" for p in row))
    
    return "\n".join(report)


def save_hcpi_results(result: Dict, prefix: str = "hcpi"):
    """
    Save HCPI results to JSON files.
    
    Creates two files:
    - {prefix}_full.json: Complete results with all details
    - {prefix}_summary.json: Summary with just key metrics
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    # Save full results
    full_filename = f"{prefix}_full_{timestamp}.json"
    with open(full_filename, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"✓ Saved full results to {full_filename}")
    
    # Save summary
    summary = {
        'us_index': result.get('us_index'),
        'timestamp': result.get('metadata', {}).get('timestamp'),
        'regional_indices': result.get('regional_indices', {}),
        'total_providers': result.get('metadata', {}).get('unique_providers'),
        'total_gpus': result.get('metadata', {}).get('total_gpus')
    }
    
    summary_filename = f"{prefix}_summary_{timestamp}.json"
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_filename}")
    
    return full_filename, summary_filename


# Example usage
if __name__ == "__main__":
    # Example data structure
    sample_data = pd.DataFrame([
        {'provider': 'AWS', 'region': 'US-East', 'price_hourly_usd': 10.5, 'gpu_count': 1000},
        {'provider': 'GCP', 'region': 'US-West', 'price_hourly_usd': 11.2, 'gpu_count': 1000},
        {'provider': 'Azure', 'region': 'US-Central', 'price_hourly_usd': 10.8, 'gpu_count': 1000},
    ])
    
    # Calculate HCPI
    result = calculate_hcpi(sample_data, verbose=True)
    
    # Print report
    print(format_hcpi_report(result))
    
    # Save results
    save_hcpi_results(result)