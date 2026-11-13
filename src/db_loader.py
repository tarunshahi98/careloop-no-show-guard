"""Database loader utility for importing CareLoop clinic and appointment records into SQLite."""

from pathlib import Path
import re
import sqlite3
from typing import List, Optional, Union
import pandas as pd

SQL_QUERIES_FILE = Path(__file__).resolve().parent.parent / "02_queries.sql"


def extract_schema_ddl(sql_file: Optional[Union[Path, str]] = None) -> List[str]:
    """Extracts table DDL statements from the SQL schema file."""
    path = Path(sql_file) if sql_file else SQL_QUERIES_FILE
    if not path.exists():
        for candidate in [
            Path("02_queries.sql"),
            Path("../02_queries.sql"),
            Path("sql/02_queries.sql"),
            Path("../sql/02_queries.sql"),
        ]:
            if candidate.exists():
                path = candidate
                break

    sql_text = path.read_text(encoding="utf-8")
    return re.findall(r"(CREATE TABLE IF NOT EXISTS [^;]+;)", sql_text, flags=re.DOTALL)


_ddl_statements = extract_schema_ddl()
CLINICS_TABLE_DDL = _ddl_statements[0] if len(_ddl_statements) > 0 else ""
APPOINTMENTS_TABLE_DDL = _ddl_statements[1] if len(_ddl_statements) > 1 else ""


def init_database(
    conn: sqlite3.Connection, sql_file: Optional[Union[Path, str]] = None
) -> None:
    """Initializes schema tables and enables SQLite foreign keys."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    for statement in extract_schema_ddl(sql_file):
        cursor.execute(statement)
    conn.commit()


def load_clinics_parameterized(
    conn: sqlite3.Connection, clinics_df: pd.DataFrame
) -> int:
    """Bulk inserts clinic records using parameterized SQL."""
    insert_sql = "INSERT OR REPLACE INTO clinics VALUES (?, ?, ?, ?, ?, ?, ?);"
    records = [
        (
            row.clinic_id,
            row.clinic_name,
            row.city,
            row.tier,
            float(row.latitude),
            float(row.longitude),
            None if pd.isna(row.temp_c) else float(row.temp_c),
        )
        for row in clinics_df.itertuples(index=False)
    ]
    cursor = conn.cursor()
    cursor.executemany(insert_sql, records)
    conn.commit()
    return len(records)


def load_appointments_parameterized(
    conn: sqlite3.Connection, appointments_df: pd.DataFrame
) -> int:
    """Bulk inserts appointment records with transaction rollback on error."""
    insert_sql = """
    INSERT INTO appointments (
        appointment_id, patient_id, clinic_id, booked_on, appt_date,
        age, gender, distance_km, fee_inr, booking_channel,
        visit_type, reminder_sent, prior_no_shows, status,
        lead_days, no_show
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    records = [
        (
            str(row.appointment_id),
            str(row.patient_id),
            str(row.clinic_id),
            str(row.booked_on),
            str(row.appt_date),
            int(row.age),
            str(row.gender),
            float(row.distance_km),
            int(row.fee_inr),
            str(row.booking_channel),
            str(row.visit_type),
            str(row.reminder_sent),
            int(row.prior_no_shows),
            str(row.status),
            int(row.lead_days),
            int(row.no_show),
        )
        for row in appointments_df.itertuples(index=False)
    ]
    cursor = conn.cursor()
    try:
        cursor.executemany(insert_sql, records)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(records)
