#!/usr/bin/env python3
import json
from pathlib import Path
from database_integration import HCPIDatabase

def main():
    db = HCPIDatabase()  # uses sqlite:///hcpi.db in CI step
    latest = db.get_latest_hcpi()
    # Fallback if none in DB yet
    payload = {"latest_hcpi": latest or {}}
    out = Path("docs/hcpi_dashboard.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"✓ wrote {out}")

if __name__ == "__main__":
    main()
