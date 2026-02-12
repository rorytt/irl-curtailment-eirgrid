from __future__ import annotations

from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/constraint_national_halfhour_wind.xlsx")
OUT_PATH = Path("data/processed/constraint_national_hourly.csv")
REPORT_PATH = Path("reports/constraint_national_preprocess_report.txt")

DUPES_PATH = Path("reports/constraint_national_duplicates_halfhour.csv")
MISSING_PATH = Path("reports/constraint_national_missing_halfhour.csv")

TOL = 1e-6


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def _normalise_technology(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .replace({"pv": "solar", "photovoltaic": "solar"})
    )


def _parse_datetime(series: pd.Series) -> pd.Series:
    # Works for Excel datetimes and strings. Safe even if already datetime64.
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.notna().mean() < 0.9:
        # fallback for dd/mm strings if any sneak in
        dt2 = pd.to_datetime(series, errors="coerce", utc=True, dayfirst=True)
        if dt2.notna().mean() > dt.notna().mean():
            dt = dt2
    return dt


def _write_duplicates(df: pd.DataFrame, value_col: str) -> int:
    dupes = df[df.duplicated(["weather_year", "technology", "datetime"], keep=False)].copy()
    if len(dupes) > 0:
        DUPES_PATH.parent.mkdir(parents=True, exist_ok=True)
        dupes.sort_values(["weather_year", "technology", "datetime"]).to_csv(DUPES_PATH, index=False)
    return int(len(dupes))


def _build_expected_halfhour_index(weather_year: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{weather_year}-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp(f"{weather_year}-12-31 23:30:00", tz="UTC")
    return pd.date_range(start=start, end=end, freq="30min")


def _audit_missing_halfhours(df_collapsed: pd.DataFrame) -> pd.DataFrame:
    missing_blocks = []
    for (wy, tech), g in df_collapsed.groupby(["weather_year", "technology"], sort=True):
        expected = _build_expected_halfhour_index(int(wy))
        actual = pd.DatetimeIndex(g["datetime"]).sort_values().unique()
        missing = expected.difference(actual)
        if len(missing) > 0:
            missing_blocks.append(
                pd.DataFrame(
                    {"weather_year": int(wy), "technology": str(tech), "missing_datetime": missing}
                )
            )
    if missing_blocks:
        return pd.concat(missing_blocks, ignore_index=True)
    return pd.DataFrame(columns=["weather_year", "technology", "missing_datetime"])


def _fill_missing_halfhours_by_avg(df_collapsed: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, int, float]:
    """
    For each (weather_year, technology), reindex to full 30-min UTC year grid.
    Fill missing values with average(nearest prev, nearest next) using ffill/bfill.
    Returns filled_df, n_filled, energy_added_by_fill.
    """
    parts = []
    n_filled_total = 0

    collapsed_total = float(df_collapsed[value_col].sum())

    for (wy, tech), g in df_collapsed.groupby(["weather_year", "technology"], sort=True):
        wy = int(wy)
        tech = str(tech)
        expected = _build_expected_halfhour_index(wy)

        s = (
            g[["datetime", value_col]]
            .set_index("datetime")[value_col]
            .sort_index()
        )

        s_full = s.reindex(expected)
        missing_mask = s_full.isna()
        n_missing = int(missing_mask.sum())

        if n_missing > 0:
            prev = s_full.ffill()
            nxt = s_full.bfill()
            filled_vals = (prev + nxt) / 2.0
            s_full = s_full.where(~missing_mask, filled_vals)

            # If still NaN (entire series empty), leave as NaN
            n_filled = int(n_missing - s_full.isna().sum())
            n_filled_total += n_filled

        out = (
            s_full.rename(value_col)
            .reset_index()
            .rename(columns={"index": "datetime"})
        )
        out.insert(1, "weather_year", wy)
        out.insert(2, "technology", tech)
        parts.append(out)

    filled_df = pd.concat(parts, ignore_index=True)
    filled_total = float(filled_df[value_col].sum())
    energy_added_by_fill = filled_total - collapsed_total
    return filled_df, n_filled_total, energy_added_by_fill


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw file: {RAW_PATH.resolve()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(RAW_PATH, sheet_name=0)
    raw_rows = len(raw)

    df = _standardise_columns(raw)

    # Require: datetime, constraint_hh, technology. weather_year optional.
    required = {"datetime", "constraint_hh", "technology"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}. Found: {list(df.columns)}")

    keep_cols = ["datetime", "constraint_hh", "technology"] + (["weather_year"] if "weather_year" in df.columns else [])
    df = df[keep_cols].copy()

    # Parse
    df["technology"] = _normalise_technology(df["technology"])
    df["datetime"] = _parse_datetime(df["datetime"])
    df["constraint_hh"] = pd.to_numeric(df["constraint_hh"], errors="coerce")

    inferred_weather_year = False
    if "weather_year" not in df.columns:
        df["weather_year"] = df["datetime"].dt.year
        inferred_weather_year = True
    else:
        df["weather_year"] = pd.to_numeric(df["weather_year"], errors="coerce")

    # Bad rows counts (pre-drop)
    bad_tech = int(df["technology"].isna().sum() + (df["technology"].astype(str).str.len() == 0).sum())
    bad_dt = int(df["datetime"].isna().sum())
    bad_val = int(df["constraint_hh"].isna().sum())
    bad_wy = int(df["weather_year"].isna().sum())

    # Drop unusable rows
    df = df.dropna(subset=["technology", "datetime", "constraint_hh", "weather_year"]).copy()
    df = df[df["technology"].astype(str).str.len() > 0].copy()
    df["weather_year"] = df["weather_year"].astype(int)

    neg_count = int((df["constraint_hh"] < 0).sum())
    tech_values = sorted(df["technology"].unique().tolist())
    wy_values = sorted(df["weather_year"].unique().tolist())

    value_col = "constraint_hh"

    # RAW total including duplicates (after cleaning)
    raw_total_including_duplicates = float(df[value_col].sum())

    # Write duplicates BEFORE collapsing
    dupes_rows = _write_duplicates(df, value_col=value_col)

    # Collapse duplicates by SUM
    before = len(df)
    df_collapsed = (
        df.groupby(["weather_year", "technology", "datetime"], as_index=False)
          .agg(constraint_hh=("constraint_hh", "sum"))
    )
    after = len(df_collapsed)
    dupes_collapsed = before - after
    collapsed_total = float(df_collapsed[value_col].sum())

    # Check raw == collapsed (since we sum duplicates)
    if abs(collapsed_total - raw_total_including_duplicates) > TOL:
        raise ValueError(
            f"Energy mismatch after duplicate collapse: raw={raw_total_including_duplicates}, "
            f"collapsed={collapsed_total}"
        )

    # Missing audit (on collapsed data)
    missing_df = _audit_missing_halfhours(df_collapsed)
    if len(missing_df) > 0:
        MISSING_PATH.parent.mkdir(parents=True, exist_ok=True)
        missing_df.to_csv(MISSING_PATH, index=False)

    # Fill missing half-hours by avg(prev,next)/nearest
    df_filled, n_filled, energy_added_by_fill = _fill_missing_halfhours_by_avg(df_collapsed, value_col=value_col)

    # Resample to hourly per (weather_year, technology)
    df_idx = df_filled.sort_values(["weather_year", "technology", "datetime"]).set_index("datetime")

    hourly_list = []
    freq_guess_map = {}

    for (wy, tech), g in df_idx.groupby(["weather_year", "technology"], sort=True):
        idx = g.index
        freq_guess = pd.infer_freq(idx[:200]) if len(idx) >= 3 else None
        freq_guess_map[(int(wy), str(tech))] = freq_guess

        h = (
            g[[value_col]]
            .resample("1h")
            .sum(min_count=1)
            .rename(columns={value_col: "constraint_hourly_MWh"})
            .reset_index()
        )
        h.insert(1, "weather_year", int(wy))
        h.insert(2, "technology", str(tech))
        hourly_list.append(h)

    hourly = pd.concat(hourly_list, ignore_index=True)
    hourly_total = float(hourly["constraint_hourly_MWh"].sum())

    expected_hourly_total = raw_total_including_duplicates + energy_added_by_fill
    if abs(hourly_total - expected_hourly_total) > TOL:
        raise ValueError(
            "Final energy audit failed.\n"
            f"raw_total_including_duplicates={raw_total_including_duplicates}\n"
            f"energy_added_by_fill={energy_added_by_fill}\n"
            f"expected_hourly_total={expected_hourly_total}\n"
            f"hourly_total={hourly_total}\n"
        )

    # Output
    hourly = hourly[["datetime", "weather_year", "technology", "constraint_hourly_MWh"]].copy()
    hourly.to_csv(OUT_PATH, index=False)

    # Report
    report_lines = []
    report_lines.append("Constraint National Preprocess Report (Half-hour -> Hourly, by technology)")
    report_lines.append("=" * 90)
    report_lines.append(f"Raw file: {RAW_PATH}")
    report_lines.append(f"Output file: {OUT_PATH}")
    report_lines.append("")
    report_lines.append(f"Raw rows: {raw_rows}")
    report_lines.append(f"Dropped rows (bad technology): {bad_tech}")
    report_lines.append(f"Dropped rows (bad weather_year): {bad_wy}")
    report_lines.append(f"Dropped rows (bad datetime): {bad_dt}")
    report_lines.append(f"Dropped rows (bad constraint_hh): {bad_val}")
    report_lines.append(f"weather_year inferred from datetime: {inferred_weather_year}")
    report_lines.append(f"Negative constraint_hh rows (kept, flagged): {neg_count}")
    report_lines.append("")
    report_lines.append(f"Technology values seen: {tech_values}")
    report_lines.append(f"Weather years seen: {wy_values}")
    report_lines.append("")
    report_lines.append(f"Duplicate half-hour rows detected (written to {DUPES_PATH} if >0): {dupes_rows}")
    report_lines.append(f"Duplicate half-hour rows collapsed by sum: {dupes_collapsed}")
    report_lines.append("")
    report_lines.append(f"Missing half-hours detected (written to {MISSING_PATH} if >0): {len(missing_df)}")
    report_lines.append(f"Missing half-hours filled (avg prev/next / nearest): {n_filled}")
    report_lines.append(f"Energy added by fill: {energy_added_by_fill:.6f} MWh")
    report_lines.append("")
    report_lines.append("Energy audit:")
    report_lines.append(f"  Raw total incl. duplicates: {raw_total_including_duplicates:.6f} MWh")
    report_lines.append(f"  Collapsed total (sum duplicates): {collapsed_total:.6f} MWh")
    report_lines.append(f"  Hourly total: {hourly_total:.6f} MWh")
    report_lines.append(f"  Check hourly == raw + fill_added: {abs(hourly_total - expected_hourly_total) <= TOL}")
    report_lines.append("")
    report_lines.append("Frequency inference by (weather_year, technology) after fill (best effort):")
    for key in sorted(freq_guess_map.keys()):
        report_lines.append(f"  {key}: {freq_guess_map[key]}")
    report_lines.append("")
    report_lines.append("Hourly output head:")
    report_lines.append(hourly.head(10).to_string(index=False))
    report_lines.append("")
    report_lines.append("Hourly output tail:")
    report_lines.append(hourly.tail(10).to_string(index=False))

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"✅ Wrote: {OUT_PATH}")
    print(f"📝 Report: {REPORT_PATH}")
    if dupes_rows > 0:
        print(f"🧪 Duplicates: {DUPES_PATH}")
    if len(missing_df) > 0:
        print(f"⚠️ Missing half-hours: {MISSING_PATH}")


if __name__ == "__main__":
    main()
