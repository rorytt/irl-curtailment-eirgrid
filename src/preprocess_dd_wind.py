from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


# --------- CONFIG YOU MAY NEED TO EDIT (ONCE) ----------
RAW_PATH = Path("data/raw/dd_national_halfhour_wind.xlsx")
OUT_PATH = Path("data/processed/dd_national_hourly_wind.csv")
REPORT_PATH = Path("reports/dd_wind_preprocess_report.txt")

# If your columns aren't exactly these, we'll auto-detect, but you can override:
POSSIBLE_DATETIME_COLS = ["datetime", "date_time", "timestamp", "settlement_datetime", "time"]
POSSIBLE_DD_COLS = ["dd_mwh", "dispatchdown_mwh", "dispatched_down_mwh", "dispatch_down_mwh", "dd", "value"]


@dataclass
class ColumnMap:
    datetime_col: str
    dd_col: str


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


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _infer_columns(df: pd.DataFrame) -> ColumnMap:
    dt_col = _find_column(df, POSSIBLE_DATETIME_COLS)
    dd_col = _find_column(df, POSSIBLE_DD_COLS)

    if dt_col is None:
        # fallback: pick first column that looks like datetime by name
        maybe = [c for c in df.columns if "date" in c or "time" in c]
        dt_col = maybe[0] if maybe else None

    if dd_col is None:
        # fallback: pick first numeric-looking column (after coercion)
        numeric_scores = {}
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            numeric_scores[c] = s.notna().mean()
        dd_col = max(numeric_scores, key=numeric_scores.get) if numeric_scores else None

    if dt_col is None or dd_col is None:
        raise ValueError(
            f"Could not infer columns. Found columns: {list(df.columns)}. "
            f"Please set POSSIBLE_* lists or hardcode the names."
        )

    return ColumnMap(datetime_col=dt_col, dd_col=dd_col)


def _parse_datetime(series: pd.Series) -> pd.Series:
    """
    Parse datetime robustly.
    We try UTC parsing first; if the source is naive/local, you can adapt later.
    """
    dt = pd.to_datetime(series, errors="coerce", utc=True)

    # If everything failed, try without utc=True (sometimes strings are odd)
    if dt.notna().mean() < 0.5:
        dt = pd.to_datetime(series, errors="coerce")

    return dt

def read_raw(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=0)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw file: {RAW_PATH.resolve()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw = read_raw(RAW_PATH)
    raw_rows = len(raw)

    df = _standardise_columns(raw)
    colmap = _infer_columns(df)

    # Keep only needed columns (prevents weird extra columns causing issues)
    df = df[[colmap.datetime_col, colmap.dd_col]].copy()
    df.rename(columns={colmap.datetime_col: "datetime", colmap.dd_col: "dd_mwh"}, inplace=True)

    # Parse datetime + numeric
    df["datetime"] = _parse_datetime(df["datetime"])
    df["dd_mwh"] = pd.to_numeric(df["dd_mwh"], errors="coerce")

    # Drop bad rows
    bad_dt = df["datetime"].isna().sum()
    bad_dd = df["dd_mwh"].isna().sum()
    df = df.dropna(subset=["datetime", "dd_mwh"])

    # Optional: clamp tiny negative noise (if you want strictness, replace with an assert)
    neg_count = (df["dd_mwh"] < 0).sum()
    if neg_count > 0:
        # keep as-is but you’ll see it in report; you can decide later
        pass

    # Aggregate duplicate timestamps (common reason for "more rows than it should")
    before_dupes = len(df)
    df = df.groupby("datetime", as_index=False).agg(dd_mwh=("dd_mwh", "sum"))
    after_dupes = len(df)
    dupes_collapsed = before_dupes - after_dupes

    # Sort + set index
    df = df.sort_values("datetime").set_index("datetime")

    # Detect expected half-hour regularity (30 minutes)
    # (We don’t hard-fail here because DST / missing data exists.)
    freq_guess = pd.infer_freq(df.index[:200])  # best effort
    # Convert to hourly: sum two half-hours
    hourly = df.resample("1H").sum(min_count=1)

    # Basic sanity checks
    # - hourly total should equal half-hour total (unless NaNs dropped etc.)
    half_total = df["dd_mwh"].sum()
    hour_total = hourly["dd_mwh"].sum()

    # Check missing hourly timestamps inside the covered range
    full_hour_index = pd.date_range(
        start=hourly.index.min(),
        end=hourly.index.max(),
        freq="1H",
        tz=hourly.index.tz
    )
    missing_hours = len(full_hour_index.difference(hourly.index))

    # Write output
    hourly = hourly.reset_index()
    hourly.to_csv(OUT_PATH, index=False)

    # Write a text report so you can cite/debug later
    report_lines = []
    report_lines.append("DD Wind Preprocess Report")
    report_lines.append("=" * 30)
    report_lines.append(f"Raw file: {RAW_PATH}")
    report_lines.append(f"Output file: {OUT_PATH}")
    report_lines.append("")
    report_lines.append(f"Raw rows: {raw_rows}")
    report_lines.append(f"Dropped rows (bad datetime): {bad_dt}")
    report_lines.append(f"Dropped rows (bad dd_mwh): {bad_dd}")
    report_lines.append(f"Negative dd_mwh rows (kept, flagged): {neg_count}")
    report_lines.append(f"Duplicate timestamps collapsed by sum: {dupes_collapsed}")
    report_lines.append("")
    report_lines.append(f"Inferred half-hour frequency (best effort): {freq_guess}")
    report_lines.append(f"Half-hour total (after cleaning): {half_total:.6f} MWh")
    report_lines.append(f"Hourly total: {hour_total:.6f} MWh")
    report_lines.append(f"Total difference (hourly - halfhour): {(hour_total - half_total):.6f} MWh")
    report_lines.append(f"Missing hourly timestamps in range: {missing_hours}")
    report_lines.append("")
    report_lines.append("Hourly output head:")
    report_lines.append(hourly.head(10).to_string(index=False))
    report_lines.append("")
    report_lines.append("Hourly output tail:")
    report_lines.append(hourly.tail(10).to_string(index=False))

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"✅ Wrote: {OUT_PATH}")
    print(f"📝 Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
