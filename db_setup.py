"""
db_setup.py — Micro QA
============================================================
Handles SQLite database initialisation and seeding.

Tables:
  - units        : Medical devices / treatment machines
  - tests        : QA test definitions with tolerances
  - test_results : Measured QA values linked to unit + test
  - service_logs : Downtime and maintenance records
"""

import sqlite3
import os

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "qa_data.db")


def get_connection() -> sqlite3.Connection:
    """Return a sqlite3 connection with foreign-key enforcement enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ---------------------------------------------------------------------------
# Schema Creation
# ---------------------------------------------------------------------------
def init_db() -> None:
    """
    Create all required tables if they do not already exist.
    Safe to call every time the app starts (idempotent).
    """
    ddl_statements = [
        # ── Units (medical devices) ──────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS units (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL UNIQUE,
            modality    TEXT    NOT NULL,           -- e.g. LINAC, CT, MRI
            manufacturer TEXT,
            model       TEXT,
            serial_no   TEXT,
            location    TEXT,
            notes       TEXT,
            created_at  TEXT    DEFAULT (datetime('now'))
        );
        """,

        # ── Tests (QA test definitions) ──────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS tests (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT    NOT NULL,
            description     TEXT,
            data_type       TEXT    NOT NULL CHECK(data_type IN ('float','boolean','choice')),
            unit_of_measure TEXT,                  -- e.g. %, cGy, mm
            reference_value REAL,                  -- expected / baseline
            tolerance_low   REAL,                  -- lower warn limit  (value < tol_low → warning)
            tolerance_high  REAL,                  -- upper warn limit  (value > tol_high → warning)
            action_low      REAL,                  -- lower action limit
            action_high     REAL,                  -- upper action limit
            choices         TEXT,                  -- CSV list for 'choice' type
            frequency       TEXT    DEFAULT 'Daily',  -- Daily / Weekly / Monthly
            created_at      TEXT    DEFAULT (datetime('now'))
        );
        """,

        # ── Unit–Test Link: which tests belong to which unit ─────────────
        """
        CREATE TABLE IF NOT EXISTS unit_tests (
            unit_id  INTEGER NOT NULL REFERENCES units(id)  ON DELETE CASCADE,
            test_id  INTEGER NOT NULL REFERENCES tests(id)  ON DELETE CASCADE,
            PRIMARY KEY (unit_id, test_id)
        );
        """,

        # ── Test Results (measured values) ───────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS test_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_id         INTEGER NOT NULL REFERENCES units(id)  ON DELETE CASCADE,
            test_id         INTEGER NOT NULL REFERENCES tests(id)  ON DELETE CASCADE,
            value_float     REAL,                  -- used when data_type = 'float'
            value_boolean   INTEGER,               -- 0/1, used when data_type = 'boolean'
            value_choice    TEXT,                  -- used when data_type = 'choice'
            status          TEXT,                  -- 'OK', 'Tolerance', 'Action'
            performed_by    TEXT,
            comment         TEXT,
            performed_at    TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        """,

        # ── Service Logs ─────────────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS service_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_id         INTEGER NOT NULL REFERENCES units(id)  ON DELETE CASCADE,
            event_type      TEXT    NOT NULL,      -- 'Downtime', 'Maintenance', 'Repair', 'Note'
            start_datetime  TEXT    NOT NULL,
            end_datetime    TEXT,                  -- NULL if ongoing
            performed_by    TEXT,
            description     TEXT,
            created_at      TEXT    DEFAULT (datetime('now'))
        );
        """,

        # ── Calibration Logs (TRS-398 results) ──────────────────────────
        """
        CREATE TABLE IF NOT EXISTS calibration_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            beam_type       TEXT    NOT NULL,       -- 'photon' | 'electron'
            unit_name       TEXT,
            institution     TEXT,
            physicist       TEXT,
            chamber         TEXT,
            energy          TEXT,
            setup           TEXT,
            n_dw            REAL,
            kq              REAL,
            k_pol           REAL,
            k_s             REAL,
            k_tp            REAL,
            k_vol_fff       REAL,                  -- FFF volume correction (Farmer)
            m_corrected     REAL,
            dose_mu_zref    REAL,
            dose_mu_zmax    REAL,
            notes           TEXT,
            performed_at    TEXT    DEFAULT (datetime('now'))
        );
        """,

        # ── Profile QA Logs (Beam Profile results) ────────────────────────
        """
        CREATE TABLE IF NOT EXISTS profile_qa_logs (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_id                 INTEGER REFERENCES units(id) ON DELETE SET NULL,
            unit_name               TEXT    NOT NULL,
            institution             TEXT,
            physicist               TEXT    NOT NULL,
            beam_energy             TEXT    NOT NULL,       -- e.g. '6 MV', '6 MV FFF', '10 MV'
            beam_type               TEXT    NOT NULL,       -- 'Flattened' | 'FFF'
            protocol                TEXT    NOT NULL,       -- 'Varian' | 'IEC 60976' | 'Siemens' | 'Elekta'
            data_source             TEXT    NOT NULL,       -- 'Water Tank (CSV/ASCII)', 'DICOM EPID', 'Synthetic'
            axis                    TEXT    NOT NULL,       -- 'Crossplane (X)', 'Inplane (Y)', 'Both'
            field_size_nominal      TEXT,                   -- e.g. '10x10 cm'

            -- Measured Geometry
            field_width_x_mm        REAL,
            field_width_y_mm        REAL,
            cax_offset_x_mm         REAL,
            cax_offset_y_mm         REAL,

            -- Penumbra Measurements (80-20%)
            penumbra_left_x_mm      REAL,
            penumbra_right_x_mm     REAL,
            penumbra_top_y_mm       REAL,
            penumbra_bottom_y_mm    REAL,

            -- Flatness & Symmetry
            flatness_x              REAL,
            flatness_y              REAL,
            symmetry_x              REAL,
            symmetry_y              REAL,
            symmetry_area_x         REAL,
            symmetry_area_y         REAL,

            -- FFF Specific
            left_slope_x            REAL,                   -- %/mm
            right_slope_x           REAL,

            -- Status & Notes
            status                  TEXT    NOT NULL,       -- 'OK' | 'Tolerance' | 'Action'
            notes                   TEXT,
            raw_summary_json        TEXT,                   -- JSON string of sampled points for redisplay
            performed_at            TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS reference_beam_data (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_id                 INTEGER REFERENCES units(id) ON DELETE SET NULL,
            unit_name               TEXT    NOT NULL,
            energy                  TEXT    NOT NULL,       -- e.g. '6 MV', '6 MV FFF', '10 MV', '10 MV FFF'
            beam_type               TEXT    NOT NULL,       -- 'Flattened' | 'FFF'
            curve_type              TEXT    NOT NULL,       -- 'Inplane Profile', 'Crossplane Profile', 'PDD'
            field_size_nominal      TEXT    NOT NULL,       -- e.g. '10x10 cm', '40x40 cm', '28x28 cm'
            depth_mm                REAL    DEFAULT 100.0,
            ssd_cm                  REAL    DEFAULT 100.0,

            -- PDD & TRS-398 Metrics
            tpr20_10                REAL,
            pdd_10                  REAL,
            pdd_20                  REAL,
            d_max_mm                REAL,

            -- Profile Metrics
            field_width_mm          REAL,
            cax_offset_mm           REAL,
            penumbra_left_mm        REAL,
            penumbra_right_mm       REAL,
            flatness_pct            REAL,
            symmetry_pct            REAL,

            -- Complete Curve JSON for plotting and baseline comparison
            curve_data_json         TEXT    NOT NULL,       -- JSON string with {"x_mm": [...], "values": [...]}
            source_filename         TEXT,
            uploaded_by             TEXT    DEFAULT 'Medical Physicist',
            notes                   TEXT,
            uploaded_at             TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS ion_chambers (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            model               TEXT    NOT NULL,       -- e.g. 'PTW 30013 Farmer (0.60 cm³, waterproof)'
            serial_no           TEXT    NOT NULL UNIQUE, -- e.g. '010482', '007905'
            ndw_cgy             REAL    NOT NULL,       -- cGy/nC (e.g. 5.385)
            ndw_gy              REAL    NOT NULL,       -- Gy/nC (e.g. 0.05385)
            user_label          TEXT,                   -- e.g. 'Farmer 1 (Clinical Standard)'
            cal_date            TEXT,                   -- e.g. '2024-05-15'
            cal_lab             TEXT,                   -- e.g. 'PTW-Freiburg SSDL'
            cert_id             TEXT,                   -- e.g. 'CAL-2024-PTW-88910'
            ref_quality         TEXT    DEFAULT 'Co-60',
            ref_temp_c          REAL    DEFAULT 20.0,
            ref_press_kpa       REAL    DEFAULT 101.325,
            is_active           INTEGER DEFAULT 1,
            notes               TEXT,
            created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS unit_energies (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_id             INTEGER NOT NULL REFERENCES units(id) ON DELETE CASCADE,
            energy_name         TEXT    NOT NULL,       -- e.g. '6 MV', '6 MV FFF', '10 MV', '10 MV FFF'
            beam_type           TEXT    NOT NULL,       -- 'Photon Flattened', 'Photon FFF', 'Electron'
            nominal_mv          REAL    NOT NULL,       -- 6.0, 10.0, etc.
            d_max_mm            REAL,                   -- e.g. 14.0 mm
            tpr20_10            REAL,                   -- Reference BQI
            pdd_10              REAL,                   -- Reference PDD at depth 10 cm (%)
            created_at          TEXT    DEFAULT (datetime('now')),
            UNIQUE(unit_id, energy_name)
        );
        """,
    ]

    with get_connection() as conn:
        for stmt in ddl_statements:
            conn.execute(stmt)
        conn.commit()

    print(f"[db_setup] Database ready at: {DB_PATH}")


# ---------------------------------------------------------------------------
# Seed Data (runs only if tables are empty)
# ---------------------------------------------------------------------------
def seed_db() -> None:
    """
    Populate the database with standard clinical QA test templates.
    Leaves units, chambers, energies, baselines, and test logs completely clean.
    """
    with get_connection() as conn:
        cur = conn.cursor()

        # Seed standard clinical QA tests if table is empty
        cur.execute("SELECT COUNT(*) FROM tests;")
        if cur.fetchone()[0] == 0:
            tests = [
                # (name, description, data_type, unit_of_measure, ref, tol_lo, tol_hi, act_lo, act_hi, choices, freq)
                (
                    "Daily Output Constancy",
                    "Measures the daily beam output relative to a calibrated reference value.",
                    "float", "%", 100.0, 97.0, 103.0, 95.0, 105.0, None, "Daily",
                ),
                (
                    "Flatness – 6 MV",
                    "Beam flatness measurement at 6 MV photon energy.",
                    "float", "%", 0.0, -1.5, 1.5, -2.5, 2.5, None, "Daily",
                ),
                (
                    "Symmetry – 6 MV",
                    "Beam symmetry measurement at 6 MV photon energy.",
                    "float", "%", 0.0, -1.5, 1.5, -2.5, 2.5, None, "Daily",
                ),
                (
                    "Laser Alignment Check",
                    "Verify room lasers align with the radiation isocenter.",
                    "float", "mm", 0.0, -1.0, 1.0, -2.0, 2.0, None, "Daily",
                ),
                (
                    "Door Interlock",
                    "Pass/Fail check for treatment room door interlock.",
                    "boolean", None, None, None, None, None, None, None, "Daily",
                ),
                (
                    "Trajectory Log – Overall MLC RMS",
                    "Overall MLC leaf positioning RMS error from Varian Trajectory Logs (.bin).",
                    "float", "mm", 0.0, 0.0, 0.5, 0.0, 1.0, None, "Daily",
                ),
                (
                    "Picket Fence – Max MLC Error",
                    "Maximum MLC leaf positioning error measured via EPID Picket Fence.",
                    "float", "mm", 0.0, 0.0, 0.25, 0.0, 0.5, None, "Weekly",
                ),
                (
                    "Winston-Lutz – 3D Isocenter Diameter",
                    "3D radiation isocenter sphere diameter from multi-angle EPID Winston-Lutz.",
                    "float", "mm", 0.0, 0.0, 0.75, 0.0, 1.0, None, "Monthly",
                ),
                (
                    "CBCT CatPhan – Uniformity Index",
                    "CBCT image uniformity index from CatPhan phantom analysis.",
                    "float", "%", 0.0, -2.0, 2.0, -3.0, 3.0, None, "Monthly",
                ),
                (
                    "CBCT CatPhan – MTF 50%",
                    "High-contrast spatial resolution MTF 50% (lp/mm) from CatPhan analysis.",
                    "float", "lp/mm", 0.5, 0.35, 2.0, 0.25, 2.5, None, "Monthly",
                ),
                (
                    "Beam Profile Flatness",
                    "Beam profile flatness evaluation across inner 80% field width.",
                    "float", "%", 0.0, -1.5, 1.5, -2.5, 2.5, None, "Daily",
                ),
                (
                    "Beam Profile Symmetry",
                    "Beam profile point-difference symmetry across inner 80% field width.",
                    "float", "%", 0.0, -1.5, 1.5, -2.5, 2.5, None, "Daily",
                ),
                (
                    "Beam Profile Penumbra",
                    "80%-20% beam profile penumbra width in mm.",
                    "float", "mm", 6.0, 5.0, 7.0, 4.5, 7.5, None, "Monthly",
                ),
            ]
            cur.executemany(
                """INSERT INTO tests
                   (name, description, data_type, unit_of_measure, reference_value,
                    tolerance_low, tolerance_high, action_low, action_high, choices, frequency)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?);""",
                tests,
            )

        conn.commit()

    print("[db_setup] Seed data applied.")


# ---------------------------------------------------------------------------
# Entry point (run directly to initialise the DB)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    seed_db()
    print("[db_setup] Done.")
