"""
crud.py — Micro QA
============================================================
Modular CRUD helper functions for every database table.
All functions open and close their own connection using a
context manager (safe for single-user Streamlit usage).
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional

from db_setup import DB_PATH, get_connection


# ===========================================================================
# UNITS
# ===========================================================================

def get_units() -> pd.DataFrame:
    """Return all units as a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM units ORDER BY name;", conn)


def get_unit_by_id(unit_id: int) -> Optional[dict]:
    """Return a single unit as a dict, or None if not found."""
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM units WHERE id=?;", (unit_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))


def get_unit_by_name(name: str) -> Optional[dict]:
    """Return a single unit by its name as a dict, or None if not found."""
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM units WHERE name=?;", (name,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))



def add_unit(
    name: str,
    modality: str,
    manufacturer: str = "",
    model: str = "",
    serial_no: str = "",
    location: str = "",
    notes: str = "",
) -> int:
    """
    Insert a new unit and return its new primary-key id.
    Raises sqlite3.IntegrityError if the name already exists.
    """
    sql = """
        INSERT INTO units (name, modality, manufacturer, model, serial_no, location, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    with get_connection() as conn:
        cur = conn.execute(sql, (name, modality, manufacturer, model, serial_no, location, notes))
        conn.commit()
        return cur.lastrowid


# ===========================================================================
# TESTS
# ===========================================================================

def get_tests() -> pd.DataFrame:
    """Return all test definitions as a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM tests ORDER BY name;", conn)


def get_tests_for_unit(unit_id: int) -> pd.DataFrame:
    """Return only the tests assigned to a specific unit."""
    sql = """
        SELECT t.*
        FROM tests t
        JOIN unit_tests ut ON ut.test_id = t.id
        WHERE ut.unit_id = ?
        ORDER BY t.name;
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn, params=(unit_id,))


def get_test_by_id(test_id: int) -> Optional[dict]:
    """Return a single test definition as a dict, or None."""
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM tests WHERE id=?;", (test_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], row))






def get_test_by_name(name: str) -> Optional[dict]:
    """Return a single test definition by name as a dict, or None."""
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM tests WHERE name=?;", (name,))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], row))






def add_test(
    name: str,
    data_type: str,
    description: str = "",
    unit_of_measure: str = "",
    reference_value: Optional[float] = None,
    tolerance_low: Optional[float] = None,
    tolerance_high: Optional[float] = None,
    action_low: Optional[float] = None,
    action_high: Optional[float] = None,
    choices: Optional[str] = None,
    frequency: str = "Daily",
) -> int:
    """
    Insert a new test definition.
    Returns the new test id.
    """
    sql = """
        INSERT INTO tests
            (name, description, data_type, unit_of_measure, reference_value,
             tolerance_low, tolerance_high, action_low, action_high, choices, frequency)
        VALUES (?,?,?,?,?,?,?,?,?,?,?);
    """
    with get_connection() as conn:
        cur = conn.execute(
            sql,
            (name, description, data_type, unit_of_measure, reference_value,
             tolerance_low, tolerance_high, action_low, action_high, choices, frequency),
        )
        conn.commit()
        return cur.lastrowid


def update_unit(
    unit_id: int,
    name: str,
    modality: str,
    manufacturer: str = "",
    model: str = "",
    serial_no: str = "",
    location: str = "",
    notes: str = "",
) -> None:
    """
    Update all editable fields of an existing unit.
    Raises sqlite3.IntegrityError if the new name collides with another unit.
    """
    sql = """
        UPDATE units
        SET name=?, modality=?, manufacturer=?, model=?,
            serial_no=?, location=?, notes=?
        WHERE id=?;
    """
    with get_connection() as conn:
        conn.execute(sql, (name, modality, manufacturer, model, serial_no, location, notes, unit_id))
        conn.commit()


def assign_test_to_unit(unit_id: int, test_id: int) -> None:
    """Create a link between a unit and a test (unit_tests junction table)."""
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO unit_tests (unit_id, test_id) VALUES (?,?);",
            (unit_id, test_id),
        )
        conn.commit()


# ===========================================================================
# TEST RESULTS
# ===========================================================================

def compute_status(
    value: float,
    action_low: Optional[float],
    action_high: Optional[float],
    tolerance_low: Optional[float],
    tolerance_high: Optional[float],
) -> str:
    """
    Determine if a measured value is 'OK', 'Tolerance', or 'Action'
    based on the test's defined limits.

    Priority: Action > Tolerance > OK
    action_low / action_high are the widest limits (worst breach first).
    """
    # ── Action breach ──────────────────────────────────────────────────────
    if action_low is not None and value < action_low:
        return "Action"
    if action_high is not None and value > action_high:
        return "Action"
    # ── Tolerance breach ───────────────────────────────────────────────────
    if tolerance_low is not None and value < tolerance_low:
        return "Tolerance"
    if tolerance_high is not None and value > tolerance_high:
        return "Tolerance"
    return "OK"


def log_result(
    unit_id: int,
    test_id: int,
    value_float: Optional[float] = None,
    value_boolean: Optional[bool] = None,
    value_choice: Optional[str] = None,
    performed_by: str = "",
    comment: str = "",
    performed_at: Optional[str] = None,
) -> str:
    """
    Insert a new test result row and return its computed status string.

    For float tests, status is computed from the test's tolerance limits.
    For boolean / choice tests, status is always 'OK' (pass/fail is implicit).
    """
    if performed_at is None:
        performed_at = datetime.now().isoformat(timespec="seconds")

    # Determine status for float values
    status = "OK"
    if value_float is not None:
        test = get_test_by_id(test_id)
        if test:
            status = compute_status(
                value_float,
                test.get("action_low"),
                test.get("action_high"),
                test.get("tolerance_low"),
                test.get("tolerance_high"),
            )

    sql = """
        INSERT INTO test_results
            (unit_id, test_id, value_float, value_boolean, value_choice,
             status, performed_by, comment, performed_at)
        VALUES (?,?,?,?,?,?,?,?,?);
    """
    bool_int = None if value_boolean is None else int(value_boolean)
    with get_connection() as conn:
        conn.execute(
            sql,
            (unit_id, test_id, value_float, bool_int, value_choice,
             status, performed_by, comment, performed_at),
        )
        conn.commit()

    return status


def get_historical_data(unit_id: int, test_id: int) -> pd.DataFrame:
    """
    Return all recorded results for a unit+test pair, ordered by date asc.
    Columns: id, value_float, value_boolean, value_choice, status,
             performed_by, comment, performed_at.
    """
    sql = """
        SELECT
            tr.id,
            tr.value_float,
            tr.value_boolean,
            tr.value_choice,
            tr.status,
            tr.performed_by,
            tr.comment,
            tr.performed_at
        FROM test_results tr
        WHERE tr.unit_id = ?
          AND tr.test_id = ?
        ORDER BY tr.performed_at ASC;
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=(unit_id, test_id))

    # Coerce performed_at to datetime for clean plotting
    df["performed_at"] = pd.to_datetime(df["performed_at"], errors="coerce")
    return df


def get_recent_results(unit_id: int, limit: int = 50) -> pd.DataFrame:
    """
    Return the most recent `limit` results for a given unit (all tests combined).
    Joins to test name for a friendly display.
    """
    sql = """
        SELECT
            t.name  AS test_name,
            tr.value_float,
            tr.value_boolean,
            tr.value_choice,
            tr.status,
            tr.performed_by,
            tr.performed_at
        FROM test_results tr
        JOIN tests t ON t.id = tr.test_id
        WHERE tr.unit_id = ?
        ORDER BY tr.performed_at DESC
        LIMIT ?;
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn, params=(unit_id, limit))


# ===========================================================================
# SERVICE LOGS
# ===========================================================================

def add_service_log(
    unit_id: int,
    event_type: str,
    start_datetime: str,
    end_datetime: Optional[str] = None,
    performed_by: str = "",
    description: str = "",
) -> int:
    """Insert a new service log entry and return its id."""
    sql = """
        INSERT INTO service_logs
            (unit_id, event_type, start_datetime, end_datetime, performed_by, description)
        VALUES (?,?,?,?,?,?);
    """
    with get_connection() as conn:
        cur = conn.execute(
            sql,
            (unit_id, event_type, start_datetime, end_datetime, performed_by, description),
        )
        conn.commit()
        return cur.lastrowid


def get_service_logs(unit_id: Optional[int] = None) -> pd.DataFrame:
    """
    Return service log entries.
    If unit_id is provided, filters for that unit only.
    Joins unit name for a friendly display.
    """
    base_sql = """
        SELECT
            sl.id,
            u.name  AS unit_name,
            sl.event_type,
            sl.start_datetime,
            sl.end_datetime,
            sl.performed_by,
            sl.description,
            sl.created_at
        FROM service_logs sl
        JOIN units u ON u.id = sl.unit_id
        {where}
        ORDER BY sl.start_datetime DESC;
    """
    if unit_id is not None:
        sql = base_sql.format(where="WHERE sl.unit_id = ?")
        with get_connection() as conn:
            return pd.read_sql(sql, conn, params=(unit_id,))
    else:
        sql = base_sql.format(where="")
        with get_connection() as conn:
            return pd.read_sql(sql, conn)


# ===========================================================================
# SUMMARY / DASHBOARD HELPERS
# ===========================================================================

def get_status_summary() -> pd.DataFrame:
    """
    Return aggregate counts of OK / Tolerance / Action statuses
    per unit for the current calendar day.
    Useful for a dashboard overview widget.
    """
    sql = """
        SELECT
            u.name          AS unit_name,
            tr.status,
            COUNT(*)        AS count
        FROM test_results tr
        JOIN units u ON u.id = tr.unit_id
        WHERE DATE(tr.performed_at) = DATE('now')
        GROUP BY u.name, tr.status
        ORDER BY u.name, tr.status;
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


# ===========================================================================
# CALIBRATION LOGS (TRS-398)
# ===========================================================================

def log_calibration(
    beam_type: str,
    unit_name: str = "",
    institution: str = "",
    physicist: str = "",
    chamber: str = "",
    energy: str = "",
    setup: str = "",
    n_dw: Optional[float] = None,
    kq: Optional[float] = None,
    k_pol: Optional[float] = None,
    k_s: Optional[float] = None,
    k_tp: Optional[float] = None,
    k_vol_fff: Optional[float] = None,
    m_corrected: Optional[float] = None,
    dose_mu_zref: Optional[float] = None,
    dose_mu_zmax: Optional[float] = None,
    notes: str = "",
    performed_at: Optional[str] = None,
) -> int:
    """Insert a TRS-398 calibration result and return its id."""
    if performed_at is None:
        performed_at = datetime.now().isoformat(timespec="seconds")

    sql = """
        INSERT INTO calibration_logs
            (beam_type, unit_name, institution, physicist, chamber, energy,
             setup, n_dw, kq, k_pol, k_s, k_tp, k_vol_fff,
             m_corrected, dose_mu_zref, dose_mu_zmax, notes, performed_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
    """
    with get_connection() as conn:
        cur = conn.execute(
            sql,
            (beam_type, unit_name, institution, physicist, chamber, energy,
             setup, n_dw, kq, k_pol, k_s, k_tp, k_vol_fff,
             m_corrected, dose_mu_zref, dose_mu_zmax, notes, performed_at),
        )
        conn.commit()
        return cur.lastrowid


def get_calibration_logs() -> pd.DataFrame:
    """Return all calibration log entries, most recent first."""
    sql = """
        SELECT *
        FROM calibration_logs
        ORDER BY performed_at DESC;
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


# ===========================================================================
# BEAM PROFILE QA LOGS
# ===========================================================================

def save_profile_qa_log(
    unit_name: str,
    physicist: str,
    beam_energy: str = "6 MV",
    beam_type: str = "Flattened",
    protocol: str = "IEC 60976",
    data_source: str = "Synthetic Reference",
    axis: str = "Crossplane (X)",
    unit_id: Optional[int] = None,
    institution: str = "",
    field_size_nominal: str = "10x10 cm",
    field_width_x_mm: Optional[float] = None,
    field_width_y_mm: Optional[float] = None,
    cax_offset_x_mm: Optional[float] = None,
    cax_offset_y_mm: Optional[float] = None,
    penumbra_left_x_mm: Optional[float] = None,
    penumbra_right_x_mm: Optional[float] = None,
    penumbra_top_y_mm: Optional[float] = None,
    penumbra_bottom_y_mm: Optional[float] = None,
    flatness_x: Optional[float] = None,
    flatness_y: Optional[float] = None,
    symmetry_x: Optional[float] = None,
    symmetry_y: Optional[float] = None,
    symmetry_area_x: Optional[float] = None,
    symmetry_area_y: Optional[float] = None,
    left_slope_x: Optional[float] = None,
    right_slope_x: Optional[float] = None,
    status: str = "OK",
    notes: str = "",
    raw_summary_json: str = "",
    performed_at: Optional[str] = None,
) -> int:
    """Insert a Beam Profile QA result and return its primary key id."""
    if performed_at is None:
        performed_at = datetime.now().isoformat(timespec="seconds")

    # If unit_id is None, attempt resolution from unit_name
    if unit_id is None and unit_name:
        with get_connection() as conn:
            cur = conn.execute("SELECT id FROM units WHERE name = ?;", (unit_name,))
            row = cur.fetchone()
            if row:
                unit_id = row[0]

    sql = """
        INSERT INTO profile_qa_logs (
            unit_id, unit_name, institution, physicist,
            beam_energy, beam_type, protocol, data_source, axis, field_size_nominal,
            field_width_x_mm, field_width_y_mm, cax_offset_x_mm, cax_offset_y_mm,
            penumbra_left_x_mm, penumbra_right_x_mm, penumbra_top_y_mm, penumbra_bottom_y_mm,
            flatness_x, flatness_y, symmetry_x, symmetry_y,
            symmetry_area_x, symmetry_area_y,
            left_slope_x, right_slope_x,
            status, notes, raw_summary_json, performed_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
    """
    params = (
        unit_id, unit_name, institution, physicist,
        beam_energy, beam_type, protocol, data_source, axis, field_size_nominal,
        field_width_x_mm, field_width_y_mm, cax_offset_x_mm, cax_offset_y_mm,
        penumbra_left_x_mm, penumbra_right_x_mm, penumbra_top_y_mm, penumbra_bottom_y_mm,
        flatness_x, flatness_y, symmetry_x, symmetry_y,
        symmetry_area_x, symmetry_area_y,
        left_slope_x, right_slope_x,
        status, notes, raw_summary_json, performed_at
    )
    with get_connection() as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid


# Alias for compatibility
log_profile_qa = save_profile_qa_log


def get_profile_qa_logs(unit_id: Optional[int] = None, limit: int = 50) -> pd.DataFrame:
    """Return all profile QA log entries, ordered by performed_at descending."""
    base_sql = """
        SELECT *
        FROM profile_qa_logs
        {where}
        ORDER BY performed_at DESC
        LIMIT ?;
    """
    with get_connection() as conn:
        if unit_id is not None:
            sql = base_sql.format(where="WHERE unit_id = ?")
            return pd.read_sql(sql, conn, params=(unit_id, limit))
        else:
            sql = base_sql.format(where="")
            return pd.read_sql(sql, conn, params=(limit,))


def get_profile_qa_by_id(log_id: int) -> Optional[dict]:
    """Return a single profile QA record by id as a dict, or None."""
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM profile_qa_logs WHERE id = ?;", (log_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], row))






# ---------------------------------------------------------------------------
# Reference Beam Data CRUD (Golden Profiles & PDD Baselines)
# ---------------------------------------------------------------------------

def save_reference_beam_data(
    unit_name: str,
    energy: str,
    beam_type: str,
    curve_type: str,
    field_size_nominal: str,
    curve_data_json: str,
    unit_id: Optional[int] = None,
    depth_mm: float = 100.0,
    ssd_cm: float = 100.0,
    tpr20_10: Optional[float] = None,
    pdd_10: Optional[float] = None,
    pdd_20: Optional[float] = None,
    d_max_mm: Optional[float] = None,
    field_width_mm: Optional[float] = None,
    cax_offset_mm: Optional[float] = None,
    penumbra_left_mm: Optional[float] = None,
    penumbra_right_mm: Optional[float] = None,
    flatness_pct: Optional[float] = None,
    symmetry_pct: Optional[float] = None,
    source_filename: str = "",
    uploaded_by: str = "Medical Physicist",
    notes: str = "",
) -> int:
    """Save or update a reference beam data baseline (profile or PDD curve)."""
    if unit_id is None:
        unit = get_unit_by_name(unit_name)
        if unit:
            unit_id = unit["id"]

    sql = """
        INSERT INTO reference_beam_data (
            unit_id, unit_name, energy, beam_type, curve_type, field_size_nominal,
            depth_mm, ssd_cm, tpr20_10, pdd_10, pdd_20, d_max_mm,
            field_width_mm, cax_offset_mm, penumbra_left_mm, penumbra_right_mm,
            flatness_pct, symmetry_pct, curve_data_json, source_filename,
            uploaded_by, notes, uploaded_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, datetime('now')
        );
    """
    params = (
        unit_id, unit_name, energy, beam_type, curve_type, field_size_nominal,
        depth_mm, ssd_cm, tpr20_10, pdd_10, pdd_20, d_max_mm,
        field_width_mm, cax_offset_mm, penumbra_left_mm, penumbra_right_mm,
        flatness_pct, symmetry_pct, curve_data_json, source_filename,
        uploaded_by, notes
    )
    with get_connection() as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid


def get_reference_beam_data(
    unit_name: Optional[str] = None,
    energy: Optional[str] = None,
    curve_type: Optional[str] = None,
    field_size_nominal: Optional[str] = None,
) -> pd.DataFrame:
    """Query reference beam curves matching optional filters."""
    conditions = []
    params = []
    if unit_name:
        conditions.append("unit_name = ?")
        params.append(unit_name)
    if energy:
        conditions.append("energy = ?")
        params.append(energy)
    if curve_type:
        conditions.append("curve_type = ?")
        params.append(curve_type)
    if field_size_nominal:
        conditions.append("field_size_nominal = ?")
        params.append(field_size_nominal)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"""
        SELECT id, unit_name, energy, beam_type, curve_type, field_size_nominal,
               depth_mm, tpr20_10, pdd_10, pdd_20, d_max_mm, field_width_mm,
               penumbra_left_mm, penumbra_right_mm, flatness_pct, symmetry_pct,
               source_filename, uploaded_by, uploaded_at, notes
        FROM reference_beam_data
        {where_clause}
        ORDER BY unit_name ASC, energy ASC, curve_type ASC;
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn, params=params)


def get_reference_beam_by_id(record_id: int) -> Optional[dict]:
    """Retrieve full reference curve data by record id including curve_data_json."""
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM reference_beam_data WHERE id = ?;", (record_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], row))






def delete_reference_beam_data(record_id: int) -> bool:
    """Delete a reference beam curve from the database."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM reference_beam_data WHERE id = ?;", (record_id,))
        conn.commit()
        return cur.rowcount > 0


def get_reference_pdd_for_unit_energy(unit_name: str, energy: str) -> Optional[dict]:
    """Find the most recent reference PDD curve for a given unit and energy."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT * FROM reference_beam_data
            WHERE unit_name = ? AND energy = ? AND curve_type LIKE '%PDD%'
            ORDER BY uploaded_at DESC LIMIT 1;
            """,
            (unit_name, energy),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], row))





# ---------------------------------------------------------------------------
# Ion Chambers Registry & Calibration Certificates
# ---------------------------------------------------------------------------

def add_ion_chamber(
    model: str,
    serial_no: str,
    ndw_cgy: float,
    user_label: str = "",
    cal_date: str = "",
    cal_lab: str = "",
    cert_id: str = "",
    ref_quality: str = "Co-60",
    notes: str = "",
) -> int:
    """Register or update an ionization chamber with its certified N_D,w calibration factor."""
    ndw_gy = ndw_cgy * 0.01
    sql = """
        INSERT INTO ion_chambers (
            model, serial_no, ndw_cgy, ndw_gy, user_label,
            cal_date, cal_lab, cert_id, ref_quality, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(serial_no) DO UPDATE SET
            model = excluded.model,
            ndw_cgy = excluded.ndw_cgy,
            ndw_gy = excluded.ndw_gy,
            user_label = excluded.user_label,
            cal_date = excluded.cal_date,
            cal_lab = excluded.cal_lab,
            cert_id = excluded.cert_id,
            notes = excluded.notes;
    """
    with get_connection() as conn:
        cur = conn.execute(sql, (
            model, serial_no, ndw_cgy, ndw_gy, user_label,
            cal_date, cal_lab, cert_id, ref_quality, notes
        ))
        conn.commit()
        return cur.lastrowid


def get_ion_chambers(active_only: bool = True) -> pd.DataFrame:
    """Return all registered ionization chambers."""
    sql = "SELECT * FROM ion_chambers WHERE is_active = 1 ORDER BY model, serial_no;" if active_only else "SELECT * FROM ion_chambers ORDER BY model, serial_no;"
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


def get_ion_chamber_by_id(chamber_id: int) -> Optional[dict]:
    """Retrieve an ionization chamber record by id."""
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM ion_chambers WHERE id = ?;", (chamber_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], row))


def delete_ion_chamber(chamber_id: int) -> bool:
    """Delete an ionization chamber record from the database."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM ion_chambers WHERE id = ?;", (chamber_id,))
        conn.commit()
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Unit Energies (Linac Energy Configuration Registry)
# ---------------------------------------------------------------------------

def add_unit_energy(
    unit_id: int,
    energy_name: str,
    beam_type: str = "Photon Flattened",
    nominal_mv: float = 6.0,
    d_max_mm: Optional[float] = None,
    tpr20_10: Optional[float] = None,
    pdd_10: Optional[float] = None,
) -> int:
    """Add or update a configured beam energy for a specific LINAC unit."""
    sql = """
        INSERT INTO unit_energies (
            unit_id, energy_name, beam_type, nominal_mv, d_max_mm, tpr20_10, pdd_10
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(unit_id, energy_name) DO UPDATE SET
            beam_type = excluded.beam_type,
            nominal_mv = excluded.nominal_mv,
            d_max_mm = COALESCE(excluded.d_max_mm, unit_energies.d_max_mm),
            tpr20_10 = COALESCE(excluded.tpr20_10, unit_energies.tpr20_10),
            pdd_10 = COALESCE(excluded.pdd_10, unit_energies.pdd_10);
    """
    with get_connection() as conn:
        cur = conn.execute(sql, (
            unit_id, energy_name, beam_type, nominal_mv, d_max_mm, tpr20_10, pdd_10
        ))
        conn.commit()
        return cur.lastrowid


def get_unit_energies(unit_id: Optional[int] = None, unit_name: Optional[str] = None) -> pd.DataFrame:
    """Query configured beam energies for a specific linac unit."""
    if unit_id is None and unit_name:
        u = get_unit_by_name(unit_name)
        if u:
            unit_id = u["id"]

    with get_connection() as conn:
        if unit_id is not None:
            return pd.read_sql(
                "SELECT * FROM unit_energies WHERE unit_id = ? ORDER BY nominal_mv ASC, energy_name ASC;",
                conn,
                params=(unit_id,)
            )
        else:
            return pd.read_sql("SELECT * FROM unit_energies ORDER BY unit_id ASC, nominal_mv ASC;", conn)


def delete_unit_energy(energy_id: int) -> bool:
    """Delete a configured beam energy."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM unit_energies WHERE id = ?;", (energy_id,))
        conn.commit()
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Database Export & Backup
# ---------------------------------------------------------------------------

def export_database_sql() -> str:
    """Exports the entire SQLite database as an ASCII SQL dump string."""
    with get_connection() as conn:
        return "\n".join(conn.iterdump())


def dump_database_to_file(filepath: str = "qa_data_dump.sql") -> str:
    """Writes the full database SQL dump to disk."""
    dump_text = export_database_sql()
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(dump_text)
    return filepath



