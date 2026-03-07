import mysql.connector
from mysql.connector import pooling
import pandas as pd
import streamlit as st
from datetime import datetime
import hashlib
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ─── CONNECTION CONFIG ───────────────────────────────────────────────────────
def get_db_config():
    try:
        return {
            "host":            st.secrets["DB_HOST"],
            "port":            int(st.secrets["DB_PORT"]),
            "user":            st.secrets["DB_USER"],
            "password":        st.secrets["DB_PASSWORD"],
            "database":        st.secrets["DB_NAME"],
            "autocommit":      True,
            "charset":         "utf8mb4",
            "use_unicode":     True,
            "connect_timeout": 10,
        }
    except Exception:
        return {
            "host":            os.getenv("DB_HOST", "localhost"),
            "port":            int(os.getenv("DB_PORT", "3306")),
            "user":            os.getenv("DB_USER", "root"),
            "password":        os.getenv("DB_PASSWORD", ""),
            "database":        os.getenv("DB_NAME", "villas_analytics"),
            "autocommit":      True,
            "charset":         "utf8mb4",
            "use_unicode":     True,
            "connect_timeout": 10,
        }

try:
    DB_CONFIG = get_db_config()
except Exception:
    import traceback
    DB_CONFIG = {}
    print("DB_CONFIG error:", traceback.format_exc())


# ─── CONNECTION POOL ─────────────────────────────────────────────────────────
@st.cache_resource
def get_connection_pool():
    """Buat connection pool MySQL — hanya dibuat sekali selama app hidup."""
    try:
        pool = pooling.MySQLConnectionPool(
            pool_name="villa_pool",
            pool_size=5,
            **DB_CONFIG,
        )
        return pool
    except mysql.connector.Error as e:
        st.error(f"❌ Gagal membuat connection pool: {e}")
        return None


def get_conn():
    pool = get_connection_pool()
    if pool:
        try:
            return pool.get_connection()
        except mysql.connector.Error as e:
            st.error(f"❌ Gagal mengambil koneksi: {e}")
    return None


def test_connection() -> dict:
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()[0]
        cursor.execute("SELECT DATABASE()")
        db_name = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return {"ok": True, "message": f"✅ Terkoneksi ke MySQL {version}",
                "version": version, "db_name": db_name}
    except mysql.connector.Error as e:
        return {"ok": False, "message": f"❌ Koneksi gagal: {e}",
                "version": None, "db_name": None}


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def _convert_decimals(df: pd.DataFrame) -> pd.DataFrame:
    """Konversi Decimal ke float secara vectorized — lebih cepat dari .apply()"""
    import decimal
    for col in df.columns:
        if not df.empty and df[col].dtype == object:
            non_null = df[col].dropna()
            if not non_null.empty and isinstance(non_null.iloc[0], decimal.Decimal):
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _to_python(val):
    import numpy as np
    if isinstance(val, np.integer):  return int(val)
    if isinstance(val, np.floating): return float(val)
    if isinstance(val, np.bool_):    return bool(val)
    if isinstance(val, np.ndarray):  return val.tolist()
    return val


def _clean_params(params):
    if params is None:
        return None
    return tuple(_to_python(p) for p in params)


# ─── QUERY RUNNER ────────────────────────────────────────────────────────────
def run_query(sql: str, params=None, fetch=True):
    conn = get_conn()
    if not conn:
        return None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, _clean_params(params) or ())
        if fetch:
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return _convert_decimals(df)
        else:
            conn.commit()
            cursor.close()
            conn.close()
            return True
    except mysql.connector.Error as e:
        st.error(f"DB Error: {e}")
        if conn:
            try: conn.close()
            except: pass
        return None


def run_many(sql: str, data: list):
    conn = get_conn()
    if not conn:
        return False
    try:
        clean_data = [_clean_params(row) for row in data]
        cursor = conn.cursor()
        cursor.executemany(sql, clean_data)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as e:
        st.error(f"Bulk insert error: {e}")
        if conn:
            try: conn.close()
            except: pass
        return False


# ─── SCHEMA DDL ──────────────────────────────────────────────────────────────
DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        username    VARCHAR(50)  UNIQUE NOT NULL,
        password    VARCHAR(255) NOT NULL,
        full_name   VARCHAR(100) NOT NULL,
        email       VARCHAR(100) UNIQUE,
        role        ENUM('admin','manager','viewer') DEFAULT 'viewer',
        is_active   TINYINT(1)   DEFAULT 1,
        last_login  DATETIME,
        created_at  DATETIME     DEFAULT CURRENT_TIMESTAMP,
        created_by  INT,
        FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS villas (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        villa_code  VARCHAR(20)  UNIQUE NOT NULL,
        villa_name  VARCHAR(100) NOT NULL,
        area        VARCHAR(50)  NOT NULL,
        color_hex   VARCHAR(10)  DEFAULT '#38bdf8',
        description VARCHAR(255) DEFAULT '',
        is_active   TINYINT(1)   DEFAULT 1,
        created_at  DATETIME     DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS upload_logs (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        filename        VARCHAR(255) NOT NULL,
        file_type       ENUM('occupancy','financial') NOT NULL,
        villa_code      VARCHAR(20),
        rows_total      INT     DEFAULT 0,
        rows_imported   INT     DEFAULT 0,
        rows_skipped    INT     DEFAULT 0,
        status          ENUM('success','partial','failed') DEFAULT 'success',
        uploaded_by     INT,
        uploaded_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
        notes           TEXT,
        FOREIGN KEY (uploaded_by) REFERENCES users(id) ON DELETE SET NULL,
        FOREIGN KEY (villa_code)  REFERENCES villas(villa_code) ON DELETE SET NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS occupancy_data (
        id               BIGINT AUTO_INCREMENT PRIMARY KEY,
        villa_code       VARCHAR(20) NOT NULL,
        date             DATE        NOT NULL,
        arrivals         INT         DEFAULT 0,
        arriving_guests  INT         DEFAULT 0,
        departures       INT         DEFAULT 0,
        departing_guests INT         DEFAULT 0,
        stay_through     INT         DEFAULT 0,
        staying_guests   INT         DEFAULT 0,
        booked           INT         DEFAULT 0,
        booked_guests    INT         DEFAULT 0,
        available        INT         DEFAULT 0,
        black            INT         DEFAULT 0,
        occupancy_pct    FLOAT       DEFAULT 0,
        created_at       DATETIME    DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_villa_date (villa_code, date),
        INDEX idx_occ_date       (date),
        INDEX idx_occ_villa_date (villa_code, date),
        FOREIGN KEY (villa_code) REFERENCES villas(villa_code) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS financial_data (
        id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
        villa_code          VARCHAR(20)   NOT NULL,
        date                DATE          NOT NULL,
        booked_flag         TINYINT(1)    DEFAULT 0,
        available_flag      TINYINT(1)    DEFAULT 0,
        guests              INT           DEFAULT 0,
        occupancy_pct       FLOAT         DEFAULT 0,
        room_revenue        DECIMAL(15,2) DEFAULT 0,
        daily_revenue       DECIMAL(15,2) DEFAULT 0,
        avg_daily_revenue   DECIMAL(15,2) DEFAULT 0,
        revpar              DECIMAL(15,2) DEFAULT 0,
        revenue_per_guest   DECIMAL(15,2) DEFAULT 0,
        is_empty_villa      TINYINT(1)    DEFAULT 0,
        is_adr_missing      TINYINT(1)    DEFAULT 0,
        is_outlier_adr      TINYINT(1)    DEFAULT 0,
        for_modeling        TINYINT(1)    DEFAULT 0,
        created_at          DATETIME      DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_villa_date (villa_code, date),
        INDEX idx_fin_date       (date),
        INDEX idx_fin_villa_date (villa_code, date),
        INDEX idx_fin_modeling   (villa_code, for_modeling),
        FOREIGN KEY (villa_code) REFERENCES villas(villa_code) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS sarima_models (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        villa_name  VARCHAR(100) UNIQUE NOT NULL,
        model_blob  LONGBLOB     NOT NULL,
        meta_blob   LONGBLOB     NOT NULL,
        mape        FLOAT,
        rmse        FLOAT,
        aic         FLOAT,
        trained_at  DATETIME     DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_villa_name (villa_name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]

DDL_MIGRATIONS = [
    # Kolom tambahan financial_data
    "ALTER TABLE financial_data ADD COLUMN IF NOT EXISTS is_empty_villa TINYINT(1) DEFAULT 0",
    "ALTER TABLE financial_data ADD COLUMN IF NOT EXISTS is_adr_missing TINYINT(1) DEFAULT 0",
    "ALTER TABLE financial_data ADD COLUMN IF NOT EXISTS is_outlier_adr TINYINT(1) DEFAULT 0",
    "ALTER TABLE financial_data ADD COLUMN IF NOT EXISTS for_modeling   TINYINT(1) DEFAULT 0",
    # Kolom tambahan villas
    "ALTER TABLE villas ADD COLUMN IF NOT EXISTS description VARCHAR(255) DEFAULT ''",
    # Kolom tambahan sarima_models
    "ALTER TABLE sarima_models ADD COLUMN IF NOT EXISTS mape       FLOAT",
    "ALTER TABLE sarima_models ADD COLUMN IF NOT EXISTS rmse       FLOAT",
    "ALTER TABLE sarima_models ADD COLUMN IF NOT EXISTS aic        FLOAT",
    "ALTER TABLE sarima_models ADD COLUMN IF NOT EXISTS trained_at DATETIME DEFAULT CURRENT_TIMESTAMP",
    # Index optimasi — IF NOT EXISTS agar aman dijalankan berulang
    "CREATE INDEX IF NOT EXISTS idx_occ_date       ON occupancy_data (date)",
    "CREATE INDEX IF NOT EXISTS idx_occ_villa_date ON occupancy_data (villa_code, date)",
    "CREATE INDEX IF NOT EXISTS idx_fin_date       ON financial_data (date)",
    "CREATE INDEX IF NOT EXISTS idx_fin_villa_date ON financial_data (villa_code, date)",
    "CREATE INDEX IF NOT EXISTS idx_fin_modeling   ON financial_data (villa_code, for_modeling)",
]

SEED_VILLAS = """
    INSERT IGNORE INTO villas (villa_code, villa_name, area, color_hex) VALUES
    ('briana',   'Briana Villas',   'Canggu',   '#3D6BE8'),
    ('castello', 'Castello Villas', 'Canggu',   '#7c3aed'),
    ('elina',    'Elina Villas',    'Canggu',   '#059669'),
    ('isola',    'Isola Villas',    'Canggu',   '#db2777'),
    ('eindra',   'Eindra Villas',   'Seminyak', '#d97706'),
    ('esha',     'Esha Villas',     'Seminyak', '#b45309'),
    ('ozamiz',   'Ozamiz Villas',   'Seminyak', '#9333ea')
"""

SEED_ADMIN = """
    INSERT IGNORE INTO users (username, password, full_name, email, role)
    VALUES ('admin', %s, 'Administrator', 'admin@villasrus.com', 'admin')
"""


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def init_db() -> dict:
    """Buat semua tabel + seed data + jalankan migration."""
    conn = get_conn()
    if not conn:
        return {"ok": False, "message": "Tidak bisa konek ke database.", "tables_created": []}

    tables_created = []
    try:
        cursor = conn.cursor()

        # Cek apakah DB sudah pernah diinit — skip DDL berat kalau sudah ada
        cursor.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = 'users'
        """)
        already_exists = cursor.fetchone()[0] > 0

        if not already_exists:
            # Pertama kali — buat semua tabel
            for ddl in DDL_STATEMENTS:
                cursor.execute(ddl)
                words = ddl.split()
                for i, w in enumerate(words):
                    if w.upper() == "EXISTS" and i + 1 < len(words):
                        tables_created.append(words[i + 1].strip())
                        break

        # Selalu jalankan migration — aman karena pakai IF NOT EXISTS
        for migration in DDL_MIGRATIONS:
            try:
                cursor.execute(migration)
            except Exception:
                pass

        # Seed data
        cursor.execute(SEED_VILLAS)
        cursor.execute(SEED_ADMIN, (hash_password("admin123"),))

        conn.commit()
        cursor.close()
        conn.close()

        msg = "Database berhasil diinisialisasi." if not already_exists else "Migration selesai."
        return {"ok": True, "message": msg, "tables_created": tables_created}

    except mysql.connector.Error as e:
        return {"ok": False, "message": f"Init DB error: {e}", "tables_created": tables_created}


@st.cache_resource
def init_db_once() -> dict:
    """Hanya dieksekusi satu kali selama app process hidup."""
    return init_db()


# ─── AUTH ────────────────────────────────────────────────────────────────────
def get_user_by_credentials(username: str, password: str):
    hashed = hash_password(password)
    df = run_query(
        "SELECT id, username, full_name, role, is_active "
        "FROM users WHERE username=%s AND password=%s AND is_active=1",
        (username, hashed)
    )
    if df is not None and not df.empty:
        run_query(
            "UPDATE users SET last_login=%s WHERE username=%s",
            (datetime.now(), username), fetch=False
        )
        return df.iloc[0].to_dict()
    return None


def get_all_users():
    return run_query(
        "SELECT id, username, full_name, email, role, is_active, last_login, created_at "
        "FROM users ORDER BY created_at DESC"
    )


def create_user(username, password, full_name, email, role, created_by):
    return run_query(
        "INSERT INTO users (username, password, full_name, email, role, created_by) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        (username, hash_password(password), full_name, email, role, created_by),
        fetch=False
    )


def update_user(user_id, full_name, email, role, is_active):
    return run_query(
        "UPDATE users SET full_name=%s, email=%s, role=%s, is_active=%s WHERE id=%s",
        (full_name, email, role, is_active, user_id), fetch=False
    )


def reset_user_password(user_id, new_password):
    return run_query(
        "UPDATE users SET password=%s WHERE id=%s",
        (hash_password(new_password), user_id), fetch=False
    )


def delete_user(user_id):
    return run_query(
        "DELETE FROM users WHERE id=%s AND role!='admin'",
        (user_id,), fetch=False
    )


# ─── VILLAS ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_villas():
    return run_query(
        "SELECT id, villa_code, villa_name, area, color_hex, description, is_active "
        "FROM villas WHERE is_active=1 ORDER BY area, villa_name"
    )


# ─── OCCUPANCY DATA ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_occupancy_data(villa_codes=None, date_from=None, date_to=None):
    sql = """
        SELECT
            o.villa_code,
            o.date,
            o.occupancy_pct,
            o.booked,
            o.staying_guests,
            o.available,
            v.villa_name,
            v.area,
            v.color_hex
        FROM occupancy_data o
        INNER JOIN villas v ON o.villa_code = v.villa_code
        WHERE v.is_active = 1
    """
    params = []
    if villa_codes:
        sql += f" AND o.villa_code IN ({','.join(['%s']*len(villa_codes))})"
        params.extend(villa_codes)
    if date_from:
        sql += " AND o.date >= %s"
        params.append(date_from)
    if date_to:
        sql += " AND o.date <= %s"
        params.append(date_to)
    sql += " ORDER BY o.villa_code, o.date"
    return run_query(sql, params or None)


@st.cache_data(ttl=3600)
def get_occupancy_data_full(villa_codes=None, date_from=None, date_to=None):
    """Versi lengkap semua kolom — pakai hanya jika butuh kolom detail."""
    sql = """
        SELECT o.*, v.villa_name, v.area, v.color_hex
        FROM occupancy_data o
        INNER JOIN villas v ON o.villa_code = v.villa_code
        WHERE v.is_active = 1
    """
    params = []
    if villa_codes:
        sql += f" AND o.villa_code IN ({','.join(['%s']*len(villa_codes))})"
        params.extend(villa_codes)
    if date_from:
        sql += " AND o.date >= %s"
        params.append(date_from)
    if date_to:
        sql += " AND o.date <= %s"
        params.append(date_to)
    sql += " ORDER BY o.villa_code, o.date"
    return run_query(sql, params or None)


# ─── FINANCIAL DATA ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_financial_data(villa_codes=None, date_from=None, date_to=None):
    sql = """
        SELECT
            f.villa_code,
            f.date,
            f.guests,
            f.occupancy_pct,
            f.avg_daily_revenue,
            f.daily_revenue,
            f.room_revenue,
            f.revpar,
            f.is_empty_villa,
            f.is_adr_missing,
            f.is_outlier_adr,
            f.for_modeling,
            v.villa_name,
            v.area,
            v.color_hex
        FROM financial_data f
        INNER JOIN villas v ON f.villa_code = v.villa_code
        WHERE v.is_active = 1
    """
    params = []
    if villa_codes:
        sql += f" AND f.villa_code IN ({','.join(['%s']*len(villa_codes))})"
        params.extend(villa_codes)
    if date_from:
        sql += " AND f.date >= %s"
        params.append(date_from)
    if date_to:
        sql += " AND f.date <= %s"
        params.append(date_to)
    sql += " ORDER BY f.villa_code, f.date"
    return run_query(sql, params or None)


@st.cache_data(ttl=3600)
def get_financial_data_for_modeling(villa_codes=None, date_from=None, date_to=None):
    sql = """
        SELECT
            f.villa_code,
            f.date,
            f.guests,
            f.occupancy_pct,
            f.avg_daily_revenue,
            f.daily_revenue,
            f.for_modeling,
            v.villa_name,
            v.area,
            v.color_hex
        FROM financial_data f
        INNER JOIN villas v ON f.villa_code = v.villa_code
        WHERE f.for_modeling = 1
          AND v.is_active = 1
    """
    params = []
    if villa_codes:
        sql += f" AND f.villa_code IN ({','.join(['%s']*len(villa_codes))})"
        params.extend(villa_codes)
    if date_from:
        sql += " AND f.date >= %s"
        params.append(date_from)
    if date_to:
        sql += " AND f.date <= %s"
        params.append(date_to)
    sql += " ORDER BY f.villa_code, f.date"
    return run_query(sql, params or None)


# ─── SARIMA MODELS ───────────────────────────────────────────────────────────
def get_sarima_model(villa_name: str):
    return run_query(
        "SELECT * FROM sarima_models WHERE villa_name=%s", (villa_name,)
    )

def get_all_sarima_models():
    return run_query(
        "SELECT villa_name, arima_order, seasonal_order, m_used, "
        "n_train, n_cycles, mape, rmse, aic, trained_at "
        "FROM sarima_models ORDER BY villa_name"
    )

def delete_sarima_model(villa_name: str):
    return run_query(
        "DELETE FROM sarima_models WHERE villa_name=%s",
        (villa_name,), fetch=False
    )


# ─── INSERT BULK ─────────────────────────────────────────────────────────────
def insert_occupancy_bulk(records: list):
    sql = """
        INSERT INTO occupancy_data
            (villa_code, date, arrivals, arriving_guests, departures,
             departing_guests, stay_through, staying_guests, booked,
             booked_guests, available, black, occupancy_pct)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            occupancy_pct  = VALUES(occupancy_pct),
            booked         = VALUES(booked),
            staying_guests = VALUES(staying_guests)
    """
    return run_many(sql, records)


def insert_financial_bulk(records: list):
    sql = """
        INSERT INTO financial_data
            (villa_code, date, booked_flag, available_flag, guests,
             occupancy_pct, room_revenue, daily_revenue,
             avg_daily_revenue, revpar, revenue_per_guest,
             is_empty_villa, is_adr_missing, is_outlier_adr, for_modeling)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            avg_daily_revenue = VALUES(avg_daily_revenue),
            daily_revenue     = VALUES(daily_revenue),
            occupancy_pct     = VALUES(occupancy_pct),
            guests            = VALUES(guests),
            is_empty_villa    = VALUES(is_empty_villa),
            is_adr_missing    = VALUES(is_adr_missing),
            is_outlier_adr    = VALUES(is_outlier_adr),
            for_modeling      = VALUES(for_modeling)
    """
    return run_many(sql, records)


# ─── UPLOAD LOGS ─────────────────────────────────────────────────────────────
def log_upload(filename, file_type, villa_code, rows_total,
               rows_imported, rows_skipped, status, user_id, notes=""):
    return run_query(
        """INSERT INTO upload_logs
           (filename, file_type, villa_code, rows_total, rows_imported,
            rows_skipped, status, uploaded_by, notes)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (filename, file_type, villa_code, rows_total,
         rows_imported, rows_skipped, status, user_id, notes),
        fetch=False
    )


def get_upload_logs():
    return run_query("""
        SELECT
            l.id, l.filename, l.file_type, l.villa_code,
            l.rows_total, l.rows_imported, l.rows_skipped,
            l.status, l.uploaded_at, l.notes,
            u.username, u.full_name
        FROM upload_logs l
        LEFT JOIN users u ON l.uploaded_by = u.id
        ORDER BY l.uploaded_at DESC
        LIMIT 100
    """)


# ─── DATA SUMMARY ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_data_summary():
    """Subquery terpisah per tabel — lebih cepat dari COUNT(DISTINCT) double JOIN."""
    return run_query("""
        SELECT
            v.villa_code,
            v.villa_name,
            v.area,
            COALESCE(occ.occ_rows, 0)  AS occ_rows,
            COALESCE(occ.occ_from, '') AS occ_from,
            COALESCE(occ.occ_to,   '') AS occ_to,
            COALESCE(fin.fin_rows, 0)  AS fin_rows
        FROM villas v
        LEFT JOIN (
            SELECT
                villa_code,
                COUNT(date) AS occ_rows,
                MIN(date)   AS occ_from,
                MAX(date)   AS occ_to
            FROM occupancy_data
            GROUP BY villa_code
        ) occ ON v.villa_code = occ.villa_code
        LEFT JOIN (
            SELECT
                villa_code,
                COUNT(date) AS fin_rows
            FROM financial_data
            GROUP BY villa_code
        ) fin ON v.villa_code = fin.villa_code
        WHERE v.is_active = 1
        ORDER BY v.area, v.villa_name
    """)