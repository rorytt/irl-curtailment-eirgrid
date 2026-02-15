# src/curtailment/cli_allocate_wind.py
from __future__ import annotations

import argparse
from pathlib import Path

from .io import (
    load_availability_nodes_calibrated,
    load_availability_regions_calibrated,
    load_constraint_national_hourly,
    load_constraint_percent_region_monthly,
    load_recorded_availability_hourly_national,
)
from .wind_counterfactual import build_counterfactual_wind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", required=True, help="Comma list of weather years, e.g. 2021,2024,2025")
    ap.add_argument("--out_dir", default="data/processed/curtailment_counterfactual_wind")
    ap.add_argument("--alpha", type=float, default=2.0, help="Region-hour shaping exponent")
    ap.add_argument("--beta", type=float, default=1.0, help="Node-hour shaping exponent")
    ap.add_argument("--gate_q", type=float, default=0.05, help="Gating quantile for A_rh (0 disables if set <0)")
    ap.add_argument("--no_raking", action="store_true")
    args = ap.parse_args()

    years = [int(x) for x in args.years.split(",")]
    gate_q = None if args.gate_q is None or args.gate_q < 0 else args.gate_q

    # Availability (2025 fleet under each weather year)
    df_avail_regions = load_availability_regions_calibrated(years=years)
    df_avail_nodes = load_availability_nodes_calibrated(years=years)

    # Recorded series (same weather years)
    df_dd_nat = load_constraint_national_hourly(years=years, technology="wind")
    df_avail_rec_nat = load_recorded_availability_hourly_national(years=years, technology="wind")
    df_dd_pct = load_constraint_percent_region_monthly(years=years, technology="wind")

    # Run allocation
    df_region_hour, df_node_hour, df_meta_month = build_counterfactual_wind(
        df_avail_regions=df_avail_regions,
        df_avail_nodes=df_avail_nodes,
        df_dd_nat=df_dd_nat,
        df_avail_rec_nat=df_avail_rec_nat,
        df_dd_pct=df_dd_pct,
        alpha_region=args.alpha,
        beta_node=args.beta,
        gate_quantile=gate_q,
        use_raking=not args.no_raking,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_region_hour.to_parquet(out_dir / "wind_curtailment_region_hour.parquet", index=False)
    df_node_hour.to_parquet(out_dir / "wind_curtailment_node_hour.parquet", index=False)
    df_meta_month.to_parquet(out_dir / "wind_curtailment_meta_month.parquet", index=False)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
