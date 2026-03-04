import pandas as pd
import numpy as np
import re
from io import StringIO

# ─── OCCUPANCY PROCESSOR ────────────────────────────────────────────────────────
OCCUPANCY_COL_MAP = {
    # Beds24 column name variants → standard name
    "date": "date",
    "Date": "date",
    "arrivals": "arrivals",
    "Arrivals": "arrivals",
    "arriving guests": "arriving_guests",
    "Arriving Guests": "arriving_guests",
    "ArrivingGuests": "arriving_guests",
    "departures": "departures",
    "Departures": "departures",
    "departing guests": "departing_guests",
    "Departing Guests": "departing_guests",
    "DepartingGuests": "departing_guests",
    "stay through": "stay_through",
    "Stay Through": "stay_through",
    "StayThrough": "stay_through",
    "staying guests": "staying_guests",
    "Staying Guests": "staying_guests",
    "StayingGuests": "staying_guests",
    "booked": "booked",
    "Booked": "booked",
    "booked guests": "booked_guests",
    "Booked Guests": "booked_guests",
    "BookedGuests": "booked_guests",
    "available": "available",
    "Available": "available",
    "black": "black",
    "Black": "black",
    "occupancy total": "occupancy_total",
    "Occupancy Total": "occupancy_total",
    "OccupancyTotal": "occupancy_total",
    "occupancy": "occupancy_total",
    "Occupancy": "occupancy_total",
}

FINANCIAL_COL_MAP = {
    "date": "date",
    "Date": "date",
    "booked": "booked_flag",
    "Booked": "booked_flag",
    "available": "available_flag",
    "Available": "available_flag",
    "guests": "guests",
    "Guests": "guests",
    "occupancy": "occupancy_pct",
    "Occupancy": "occupancy_pct",
    "occupancy total": "occupancy_pct",
    "room revenue idr": "room_revenue",
    "Room Revenue IDR": "room_revenue",
    "RoomRevenueIDR": "room_revenue",
    "room revenue": "room_revenue",
    "daily revenue idr": "daily_revenue",
    "Daily Revenue IDR": "daily_revenue",
    "DailyRevenueIDR": "daily_revenue",
    "daily revenue": "daily_revenue",
    "average daily revenue idr": "avg_daily_revenue",
    "Average Daily Revenue IDR": "avg_daily_revenue",
    "AverageDailyRevenueIDR": "avg_daily_revenue",
    "average daily rate": "avg_daily_revenue",
    "adr": "avg_daily_revenue",
    "ADR": "avg_daily_revenue",
    "revenue per available room idr": "revpar",
    "Revenue per Available Room IDR": "revpar",
    "RevPAR": "revpar",
    "revpar": "revpar",
    "revenue per guest idr": "revenue_per_guest",
    "Revenue per Guest IDR": "revenue_per_guest",
    "revenue per guest": "revenue_per_guest",
}


def _parse_occupancy_pct(val) -> float:
    """Convert '75%', '100%', '150%', 75.0 → float, capped at 100."""
    if pd.isna(val):
        return 0.0
    s = str(val).replace("%", "").strip()
    try:
        v = float(s)
        return min(v, 100.0)  # clip to 100 (normalisasi anomali >100%)
    except ValueError:
        return 0.0


def _safe_float(val) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val) -> int:
    try:
        return int(float(str(val).replace(",", "").strip()))
    except (ValueError, TypeError):
        return 0


def _parse_date(val) -> pd.Timestamp | None:
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]:
        try:
            return pd.to_datetime(val, format=fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(val, infer_datetime_format=True)
    except Exception:
        return None


def process_occupancy_csv(uploaded_file, villa_code: str) -> dict:
    """
    Parse & clean a Beds24 occupancy CSV.
    Returns:
        {
            "records": list of tuples for DB insert,
            "df_preview": pd.DataFrame (first 20 rows),
            "stats": {total, imported, skipped, anomalies_clipped},
            "errors": list of str,
        }
    """
    errors = []
    try:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(content))
    except Exception as e:
        return {"records": [], "df_preview": pd.DataFrame(),
                "stats": {}, "errors": [f"Gagal membaca CSV: {e}"]}

    # Normalise column names
    df.columns = [OCCUPANCY_COL_MAP.get(c, c.lower().replace(" ", "_")) for c in df.columns]

    required = ["date", "occupancy_total"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"records": [], "df_preview": df.head(5),
                "stats": {}, "errors": [f"Kolom tidak ditemukan: {missing}"]}

    # Optional cols with defaults
    int_cols = ["arrivals", "arriving_guests", "departures", "departing_guests",
                "stay_through", "staying_guests", "booked", "booked_guests", "available", "black"]
    for col in int_cols:
        if col not in df.columns:
            df[col] = 0

    # Parse dates
    df["date_parsed"] = df["date"].apply(_parse_date)
    n_bad_dates = df["date_parsed"].isna().sum()
    if n_bad_dates:
        errors.append(f"⚠️ {n_bad_dates} baris dilewati karena format tanggal tidak valid.")
    df = df.dropna(subset=["date_parsed"])

    # Parse occupancy + clip anomalies
    df["occ_raw"] = df.apply(
    lambda r: 100.0 if _safe_int(r.get("booked", 0)) > 0 else 0.0,
    axis=1
)
    anomalies_clipped = int((df["occ_raw"] > 100).sum()) if "occupancy_total" in df.columns else 0
    # Already capped by _parse_occupancy_pct

    # Remove duplicates by date
    df = df.drop_duplicates(subset=["date_parsed"], keep="last")

    # Build records
    records = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            rec = (
                villa_code,
                row["date_parsed"].date(),
                _safe_int(row.get("arrivals", 0)),
                _safe_int(row.get("arriving_guests", 0)),
                _safe_int(row.get("departures", 0)),
                _safe_int(row.get("departing_guests", 0)),
                _safe_int(row.get("stay_through", 0)),
                _safe_int(row.get("staying_guests", 0)),
                _safe_int(row.get("booked", 0)),
                _safe_int(row.get("booked_guests", 0)),
                _safe_int(row.get("available", 0)),
                _safe_int(row.get("black", 0)),
                round(float(row["occ_raw"]), 2),
            )
            records.append(rec)
        except Exception as e:
            skipped += 1
            errors.append(f"Baris {_}: {e}")

    stats = {
        "total": len(df) + n_bad_dates,
        "imported": len(records),
        "skipped": skipped + n_bad_dates,
        "anomalies_clipped": anomalies_clipped,
    }

    # Preview DF
    preview_cols = ["date_parsed", "occ_raw"] + [c for c in int_cols if c in df.columns]
    df_preview = df[preview_cols].head(20).rename(columns={"date_parsed": "date", "occ_raw": "occupancy_%"})

    return {
        "records": records,
        "df_preview": df_preview,
        "stats": stats,
        "errors": errors,
    }


def process_financial_csv(uploaded_file, villa_code: str) -> dict:
    """
    Parse & clean a Beds24 financial CSV.

    Flagging strategy (NO data removed):
    - is_outlier_adr  : ADR positif melampaui batas IQR × 3.5 (flag only, data tetap disimpan)
    - is_empty_villa  : guests == 0 (vila kosong murni, zero revenue wajar)
    - is_adr_missing  : guests > 0 tapi ADR == 0 (vila terisi tanpa ADR tercatat)

    Hanya baris dengan vila terisi (guests > 0) DAN ADR > 0 yang masuk
    kolom `for_modeling = 1` — digunakan sebagai subset pemodelan SARIMA.
    """
    errors = []
    try:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(content))
    except Exception as e:
        return {"records": [], "df_preview": pd.DataFrame(),
                "stats": {}, "errors": [f"Gagal membaca CSV: {e}"]}

    df.columns = [FINANCIAL_COL_MAP.get(c, c.lower().replace(" ", "_")) for c in df.columns]

    required = ["date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"records": [], "df_preview": df.head(5),
                "stats": {}, "errors": [f"Kolom tidak ditemukan: {missing}"]}

    # Default missing cols
    for col in ["booked_flag", "available_flag", "guests", "occupancy_pct",
                "room_revenue", "daily_revenue", "avg_daily_revenue", "revpar", "revenue_per_guest"]:
        if col not in df.columns:
            df[col] = 0

    # Parse dates
    df["date_parsed"] = df["date"].apply(_parse_date)
    n_bad = df["date_parsed"].isna().sum()
    if n_bad:
        errors.append(f"⚠️ {n_bad} baris dilewati karena format tanggal tidak valid.")
    df = df.dropna(subset=["date_parsed"])

    # Parse occupancy
    if "occupancy_pct" in df.columns:
        df["occupancy_pct"] = df["occupancy_pct"].apply(_parse_occupancy_pct)

    # Parse numeric
    for col in ["room_revenue", "daily_revenue", "avg_daily_revenue", "revpar", "revenue_per_guest"]:
        df[col] = df[col].apply(_safe_float)

    for col in ["guests"]:
        df[col] = df[col].apply(_safe_int)

    # ── Duplicates (keep last, still count for stats) ──────────────────────────
    n_dup = df.duplicated(subset=["date_parsed"]).sum()
    df = df.drop_duplicates(subset=["date_parsed"], keep="last")

    # ── FLAG 1: is_empty_villa — vila kosong murni (guests == 0) ──────────────
    df["is_empty_villa"] = (df["guests"] == 0).astype(int)

    # ── FLAG 2: is_adr_missing — vila terisi tapi ADR tidak tercatat ──────────
    df["is_adr_missing"] = (
        (df["guests"] > 0) & (df["avg_daily_revenue"] == 0)
    ).astype(int)

    # ── FLAG 3: is_outlier_adr — IQR × 3.5, hanya pada ADR positif ───────────
    # Validasi silang: outlier hanya dianggap valid jika terjadi saat vila
    # memang terisi (guests > 0). ADR positif saat vila kosong tidak mungkin
    # terjadi, sehingga seluruh outlier yang terdeteksi adalah harga nyata.
    df["is_outlier_adr"] = 0
    adr_nonzero = df.loc[df["avg_daily_revenue"] > 0, "avg_daily_revenue"]
    outliers_flagged = 0
    iqr_upper = None
    if len(adr_nonzero) > 10:
        q1, q3 = adr_nonzero.quantile(0.25), adr_nonzero.quantile(0.75)
        iqr = q3 - q1
        iqr_upper = round(q3 + 3.5 * iqr, 0)
        iqr_lower = max(0, q1 - 3.5 * iqr)
        mask_outlier = (df["avg_daily_revenue"] > 0) & (
            (df["avg_daily_revenue"] < iqr_lower) | (df["avg_daily_revenue"] > iqr_upper)
        )
        df.loc[mask_outlier, "is_outlier_adr"] = 1
        outliers_flagged = int(mask_outlier.sum())
        if outliers_flagged:
            errors.append(
                f"ℹ️ {outliers_flagged} baris ADR melebihi batas IQR×3.5 "
                f"(upper ≈ {iqr_upper:,.0f} IDR) — di-flag sebagai outlier, data tetap disimpan."
            )

    # ── FLAG 4: for_modeling — subset bersih untuk SARIMA ─────────────────────
    # Hanya baris: vila terisi (guests>0) DAN ADR tercatat (>0)
    df["for_modeling"] = (
        (df["guests"] > 0) & (df["avg_daily_revenue"] > 0)
    ).astype(int)

    # ── Build records ──────────────────────────────────────────────────────────
    records = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            rec = (
                villa_code,
                row["date_parsed"].date(),
                1 if _safe_float(row.get("booked_flag", 0)) >= 1 else 0,
                1 if _safe_float(row.get("available_flag", 0)) >= 1 else 0,
                _safe_int(row.get("guests", 0)),
                round(float(row["occupancy_pct"]), 2),
                round(float(row["room_revenue"]), 2),
                round(float(row["daily_revenue"]), 2),
                round(float(row["avg_daily_revenue"]), 2),
                round(float(row["revpar"]), 2),
                round(float(row["revenue_per_guest"]), 2),
                int(row["is_empty_villa"]),
                int(row["is_adr_missing"]),
                int(row["is_outlier_adr"]),
                int(row["for_modeling"]),
            )
            records.append(rec)
        except Exception as e:
            skipped += 1

    zero_revenue_total = int(df["is_empty_villa"].sum() + df["is_adr_missing"].sum())

    stats = {
        "total": len(df) + n_bad + n_dup,
        "imported": len(records),
        "skipped": skipped + n_bad,
        "duplicates_removed": int(n_dup),
        "outliers_flagged": outliers_flagged,
        "iqr_upper": iqr_upper,
        "zero_revenue": zero_revenue_total,
        "empty_villa_days": int(df["is_empty_villa"].sum()),
        "adr_missing_days": int(df["is_adr_missing"].sum()),
        "for_modeling": int(df["for_modeling"].sum()),
    }

    preview_cols = ["date_parsed", "guests", "occupancy_pct", "avg_daily_revenue",
                    "daily_revenue", "is_empty_villa", "is_adr_missing",
                    "is_outlier_adr", "for_modeling"]
    preview_cols = [c for c in preview_cols if c in df.columns]
    df_preview = df[preview_cols].head(20).rename(columns={"date_parsed": "date"})

    return {
        "records": records,
        "df_preview": df_preview,
        "stats": stats,
        "errors": errors,
    }
