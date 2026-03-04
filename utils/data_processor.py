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


def _parse_dates_vectorized(series: pd.Series) -> pd.Series:
    """
    Vectorized date parsing — jauh lebih cepat dari apply(_parse_date) baris per baris.
    Coba beberapa format umum, fallback ke pandas inference.
    """
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
            if parsed.notna().sum() > len(series) * 0.8:
                return parsed
        except Exception:
            pass
    # Fallback: pandas auto-infer
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")


def _parse_numeric_col(series: pd.Series) -> pd.Series:
    """Vectorized numeric parsing — hapus koma, convert ke float."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    ).fillna(0.0)


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

    # ── FIX: Vectorized date parsing (lebih cepat dari apply row-by-row) ──────
    df["date_parsed"] = _parse_dates_vectorized(df["date"])
    n_bad_dates = int(df["date_parsed"].isna().sum())
    if n_bad_dates:
        errors.append(f"⚠️ {n_bad_dates} baris dilewati karena format tanggal tidak valid.")
    df = df.dropna(subset=["date_parsed"])

    # ── FIX: Parse occupancy dari kolom occupancy_total (bukan dari booked) ───
    df["occ_raw"] = df["occupancy_total"].apply(_parse_occupancy_pct)
    anomalies_clipped = int((df["occ_raw"] >= 100).sum())

    # ── FIX: Vectorized int parsing untuk semua kolom integer ─────────────────
    for col in int_cols:
        df[col] = _parse_numeric_col(df[col]).astype(int)

    # Remove duplicates by date
    df = df.drop_duplicates(subset=["date_parsed"], keep="last").reset_index(drop=True)

    # ── FIX: Build records dengan zip vectorized (bukan iterrows) ─────────────
    try:
        records = list(zip(
            [villa_code] * len(df),
            df["date_parsed"].dt.date,
            df["arrivals"].tolist(),
            df["arriving_guests"].tolist(),
            df["departures"].tolist(),
            df["departing_guests"].tolist(),
            df["stay_through"].tolist(),
            df["staying_guests"].tolist(),
            df["booked"].tolist(),
            df["booked_guests"].tolist(),
            df["available"].tolist(),
            df["black"].tolist(),
            df["occ_raw"].round(2).tolist(),
        ))
        skipped = 0
    except Exception as e:
        errors.append(f"Error saat build records: {e}")
        records = []
        skipped = len(df)

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

    # ── FIX: Vectorized date parsing ──────────────────────────────────────────
    df["date_parsed"] = _parse_dates_vectorized(df["date"])
    n_bad = int(df["date_parsed"].isna().sum())
    if n_bad:
        errors.append(f"⚠️ {n_bad} baris dilewati karena format tanggal tidak valid.")
    df = df.dropna(subset=["date_parsed"])

    # ── FIX: Vectorized occupancy parsing ─────────────────────────────────────
    if "occupancy_pct" in df.columns:
        df["occupancy_pct"] = df["occupancy_pct"].apply(_parse_occupancy_pct)

    # ── FIX: Vectorized numeric parsing untuk semua kolom revenue ─────────────
    for col in ["room_revenue", "daily_revenue", "avg_daily_revenue", "revpar", "revenue_per_guest"]:
        df[col] = _parse_numeric_col(df[col])

    df["guests"] = _parse_numeric_col(df["guests"]).astype(int)

    # ── FIX: Vectorized booked/available flag parsing ─────────────────────────
    df["booked_flag"]    = (_parse_numeric_col(df["booked_flag"]) >= 1).astype(int)
    df["available_flag"] = (_parse_numeric_col(df["available_flag"]) >= 1).astype(int)

    # ── Duplicates (keep last, still count for stats) ──────────────────────────
    n_dup = int(df.duplicated(subset=["date_parsed"]).sum())
    df = df.drop_duplicates(subset=["date_parsed"], keep="last").reset_index(drop=True)

    # ── FLAG 1: is_empty_villa — vila kosong murni (guests == 0) ──────────────
    df["is_empty_villa"] = (df["guests"] == 0).astype(int)

    # ── FLAG 2: is_adr_missing — vila terisi tapi ADR tidak tercatat ──────────
    df["is_adr_missing"] = (
        (df["guests"] > 0) & (df["avg_daily_revenue"] == 0)
    ).astype(int)

    # ── FLAG 3: is_outlier_adr — IQR × 3.5, hanya pada ADR positif ───────────
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
    df["for_modeling"] = (
        (df["guests"] > 0) & (df["avg_daily_revenue"] > 0)
    ).astype(int)

    # ── FIX: Build records dengan zip vectorized (bukan iterrows) ─────────────
    try:
        records = list(zip(
            [villa_code] * len(df),
            df["date_parsed"].dt.date,
            df["booked_flag"].tolist(),
            df["available_flag"].tolist(),
            df["guests"].tolist(),
            df["occupancy_pct"].round(2).tolist(),
            df["room_revenue"].round(2).tolist(),
            df["daily_revenue"].round(2).tolist(),
            df["avg_daily_revenue"].round(2).tolist(),
            df["revpar"].round(2).tolist(),
            df["revenue_per_guest"].round(2).tolist(),
            df["is_empty_villa"].tolist(),
            df["is_adr_missing"].tolist(),
            df["is_outlier_adr"].tolist(),
            df["for_modeling"].tolist(),
        ))
        skipped = 0
    except Exception as e:
        errors.append(f"Error saat build records: {e}")
        records = []
        skipped = len(df)

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