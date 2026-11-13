"""Data ingestion, validation, and preprocessing for CareLoop appointments data."""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def cap_series_iqr(
    series: pd.Series,
    factor: float = 1.5,
    clip_min_zero: bool = True,
    decimals: Optional[int] = None,
) -> pd.Series:
    """Caps values in a pandas Series using IQR fences."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    uf = q3 + factor * iqr
    lf = max(0.0, q1 - factor * iqr) if clip_min_zero else (q1 - factor * iqr)
    clipped = np.clip(series, lf, uf)
    return clipped.round(decimals) if decimals is not None else clipped


def parse_raw_row(row: Dict[str, str]) -> Dict:
    """Type-casts raw CSV fields and strips string whitespace."""
    converted = dict(row)
    for key in [
        "appointment_id",
        "patient_id",
        "clinic_id",
        "gender",
        "booking_channel",
        "visit_type",
        "reminder_sent",
        "status",
    ]:
        if key in converted and converted[key] is not None:
            converted[key] = str(converted[key]).strip()

    converted["age"] = int(row["age"])
    converted["fee_inr"] = int(row["fee_inr"])
    converted["prior_no_shows"] = int(row["prior_no_shows"])
    converted["distance_km"] = float(row["distance_km"])
    return converted


def ingest_foundations_csv(filepath: str | Path) -> Tuple[List[Dict], List[Dict]]:
    """Parses appointments CSV, separating valid rows from rows failing type conversion."""
    accepted, rejected = [], []
    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                accepted.append(parse_raw_row(row))
            except (ValueError, KeyError):
                rejected.append(row)
    return accepted, rejected


def clean_appointments_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Deduplicates, validates dates/ages, imputes medians, and caps outliers via IQR."""
    df = df_raw.drop_duplicates().copy()
    df["booking_channel"] = df["booking_channel"].str.strip().str.title()

    df["appt_date"] = pd.to_datetime(df["appt_date"], errors="coerce")
    df["booked_on"] = pd.to_datetime(df["booked_on"], errors="coerce")
    df = df.dropna(subset=["appt_date", "booked_on", "status"]).copy()

    # Domain constraint: appointment date cannot precede booking date
    df = df[df["appt_date"] >= df["booked_on"]].copy()

    # Age validity check (1-100)
    df.loc[(df["age"] < 1) | (df["age"] > 100), "age"] = np.nan
    df["age"] = df["age"].fillna(df["age"].median()).round().astype(int)
    df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())

    # IQR outlier capping
    df["distance_km"] = cap_series_iqr(df["distance_km"], decimals=1)
    df["fee_inr"] = cap_series_iqr(df["fee_inr"]).round().astype(int)

    # Engineered fields
    df["lead_days"] = (df["appt_date"] - df["booked_on"]).dt.days
    df["no_show"] = (df["status"] == "No-Show").astype(int)

    return df.sort_values("appointment_id").reset_index(drop=True)
