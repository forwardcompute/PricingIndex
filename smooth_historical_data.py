#!/usr/bin/env python3
"""
Apply smoothing to existing hcpi_history.json

This retroactively smooths your historical index data without re-scraping.
Run this once to smooth all past data, then the smoothing will continue
automatically going forward in master_runner.py
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def smooth_historical_index(
    input_json: str = "hcpi/hcpi_history.json",
    output_json: str = "hcpi/hcpi_history_smoothed.json",
    window_hours: int = 24,
    backup: bool = True
):
    """
    Apply rolling average smoothing to historical index data.
    
    Args:
        input_json: Path to existing hcpi_history.json
        output_json: Path to write smoothed version
        window_hours: Hours for rolling average window
        backup: Create backup of original file
    """
    
    input_path = Path(input_json)
    output_path = Path(output_json)
    
    if not input_path.exists():
        print(f"ERROR: {input_json} not found!")
        return False
    
    # Backup original
    if backup:
        backup_path = input_path.with_suffix('.json.backup')
        backup_path.write_text(input_path.read_text())
        print(f"✓ Created backup: {backup_path}")
    
    # Load existing data
    try:
        data = json.loads(input_path.read_text())
        if not isinstance(data, list):
            print("ERROR: Expected JSON array")
            return False
    except Exception as e:
        print(f"ERROR reading JSON: {e}")
        return False
    
    print(f"Loaded {len(data)} historical data points")
    
    if len(data) == 0:
        print("No data to smooth")
        return False
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Handle multiple timestamp formats flexibly
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce', format='mixed')
    df = df.dropna(subset=['timestamp'])  # Remove any unparseable timestamps
    df = df.sort_values('timestamp')
    
    if df.empty:
        print("ERROR: All timestamps failed to parse")
        return False
    
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Identify numeric columns to smooth (indices)
    numeric_cols = []
    for col in df.columns:
        if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    
    print(f"Smoothing columns: {', '.join(numeric_cols)}")
    
    # Apply rolling average to each numeric column
    for col in numeric_cols:
        if df[col].notna().sum() > 0:  # Only if we have data
            # Save raw values
            raw_col = f"{col}_raw"
            df[raw_col] = df[col].copy()
            
            # Apply smoothing
            df[col] = df[col].rolling(
                window=window_hours,
                min_periods=1,
                center=True
            ).mean()
            
            # Stats
            raw_std = df[raw_col].std()
            smooth_std = df[col].std()
            reduction = ((raw_std - smooth_std) / raw_std * 100) if raw_std > 0 else 0
            
            print(f"  {col}: std {raw_std:.4f} → {smooth_std:.4f} ({reduction:.1f}% reduction)")
    
    # Convert back to JSON format (drop raw columns)
    df_output = df[[col for col in df.columns if not col.endswith('_raw')]]
    
    # Convert to records
    smoothed_data = []
    for _, row in df_output.iterrows():
        record = {"timestamp": row['timestamp'].strftime("%Y-%m-%dT%H:%M:%SZ")}
        for col in numeric_cols:
            val = row[col]
            if pd.notna(val):
                record[col] = round(float(val), 4)
        smoothed_data.append(record)
    
    # Write smoothed version
    output_path.write_text(json.dumps(smoothed_data, indent=2))
    print(f"\n✓ Wrote smoothed data to: {output_path}")
    print(f"  {len(smoothed_data)} data points")
    
    # Show comparison
    if 'us_index' in df.columns:
        orig_range = df['us_index_raw'].max() - df['us_index_raw'].min()
        smooth_range = df['us_index'].max() - df['us_index'].min()
        print(f"\nUS Index range:")
        print(f"  Original: {df['us_index_raw'].min():.2f} - {df['us_index_raw'].max():.2f} (${orig_range:.2f})")
        print(f"  Smoothed: {df['us_index'].min():.2f} - {df['us_index'].max():.2f} (${smooth_range:.2f})")
    
    print(f"\nTo use smoothed data:")
    print(f"  mv {output_path} {input_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Smooth historical HCPI index data")
    parser.add_argument("--input", default="hcpi/hcpi_history.json", 
                       help="Input JSON file")
    parser.add_argument("--output", default="hcpi/hcpi_history_smoothed.json",
                       help="Output JSON file (default: creates new file)")
    parser.add_argument("--window", type=int, default=24,
                       help="Rolling window size in hours (default: 24)")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create backup of original")
    parser.add_argument("--in-place", action="store_true",
                       help="Overwrite original file (use with caution)")
    
    args = parser.parse_args()
    
    output = args.input if args.in_place else args.output
    
    success = smooth_historical_index(
        input_json=args.input,
        output_json=output,
        window_hours=args.window,
        backup=not args.no_backup
    )
    
    if success:
        print("\n✅ Smoothing complete!")
    else:
        print("\n❌ Smoothing failed")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
