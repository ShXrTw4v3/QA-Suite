"""
Automated Pytest Validation Suite: SQLite Database Schema & CRUD Persistence

Covers:
- Database schema initialization (init_db) and table idempotency.
- Seeding verification (units, tests, unit_tests link).
- New profile_qa_logs schema and CRUD operations (save_profile_qa_log / log_profile_qa).
- Multi-table CRUD helpers:
  * Units (add_unit, get_units, get_unit_by_id, delete_unit)
  * Tests & Unit-Test assignments (add_test, assign_test_to_unit, get_tests_for_unit)
  * Test Results & Automatic Status Classification (OK, Tolerance, Action)
  * Calibration Logs (TRS-398 photon & electron logging)
  * Service & Downtime Logs (active and resolved)
- Foreign Key cascades and referential integrity (PRAGMA foreign_keys = ON).
"""

import sqlite3
import pytest
import pandas as pd
from datetime import datetime

import db_setup
import crud


# ==============================================================================
# Tier 1: Category-Partition & Schema Initialisation
# ==============================================================================

class TestDatabaseSchemaAndSeeding:
    """Verify table creation, idempotency, and seeding across all QA domains."""

    def test_schema_creation_and_all_tables_present(self, temp_db):
        """Verify that init_db creates all 7 tables including profile_qa_logs."""
        expected_tables = {
            "units",
            "tests",
            "unit_tests",
            "test_results",
            "service_logs",
            "calibration_logs",
            "profile_qa_logs",
        }
        with db_setup.get_connection() as conn:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            actual_tables = {row[0] for row in cur.fetchall()}

        missing = expected_tables - actual_tables
        assert not missing, f"Missing tables in database: {missing}"

    def test_schema_init_idempotency(self, temp_db):
        """Calling init_db multiple times must be completely safe and idempotent."""
        db_setup.init_db()
        db_setup.init_db()

    def test_seed_db_idempotency(self, seeded_db):
        """Calling seed_db multiple times must not create duplicate baseline rows."""
        with db_setup.get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM units;")
            initial_units = cur.fetchone()[0]
            cur = conn.execute("SELECT COUNT(*) FROM tests;")
            initial_tests = cur.fetchone()[0]

        # Call seed_db a second time
        db_setup.seed_db()

        with db_setup.get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM units;")
            assert cur.fetchone()[0] == initial_units
            cur = conn.execute("SELECT COUNT(*) FROM tests;")
            assert cur.fetchone()[0] == initial_tests


# ==============================================================================
# Tier 2: Boundary Value Analysis (BVA) & Foreign Key Enforcement
# ==============================================================================

class TestDatabaseBoundariesAndIntegrity:
    """Verify status calculation limits, foreign key cascading, and unique constraints."""

    def test_foreign_key_cascade_deletion(self, temp_db):
        """
        Verify that deleting a unit cascades cleanly and removes all related
        test_results and unit_tests rows without leaving orphans.
        """
        unit_id = crud.add_unit(name="Cascade LINAC", modality="LINAC")
        test_id = crud.add_test(
            name="Output Constancy",
            description="Daily output",
            data_type="float",
            unit_of_measure="%",
            reference_value=100.0,
            tolerance_low=98.0,
            tolerance_high=102.0,
            action_low=95.0,
            action_high=105.0,
        )
        crud.assign_test_to_unit(unit_id, test_id)

        # Log a result
        crud.log_result(unit_id=unit_id, test_id=test_id, value_float=100.2, performed_by="Physicist A")

        with db_setup.get_connection() as conn:
            # Confirm rows exist
            assert conn.execute("SELECT COUNT(*) FROM unit_tests WHERE unit_id=?;", (unit_id,)).fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM test_results WHERE unit_id=?;", (unit_id,)).fetchone()[0] == 1

            # Delete the unit
            conn.execute("DELETE FROM units WHERE id=?;", (unit_id,))
            conn.commit()

            # Confirm cascading deletion
            assert conn.execute("SELECT COUNT(*) FROM unit_tests WHERE unit_id=?;", (unit_id,)).fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM test_results WHERE unit_id=?;", (unit_id,)).fetchone()[0] == 0

    @pytest.mark.parametrize(
        "value,expected_status",
        [
            (100.0, "OK"),           # Exactly nominal
            (99.0, "OK"),            # Inside +/- 2%
            (101.5, "OK"),           # Inside +/- 2%
            (102.5, "Tolerance"),    # Between 102% and 105% -> Tolerance
            (97.5, "Tolerance"),     # Between 95% and 98% -> Tolerance
            (105.5, "Action"),       # > 105% -> Action
            (94.0, "Action"),        # < 95% -> Action
        ],
    )
    def test_automatic_status_classification(self, temp_db, value, expected_status):
        """
        Verify crud.log_result automatically classifies status based on
        tolerance_low/high and action_low/high thresholds.
        """
        unit_id = crud.add_unit(name=f"Unit_{value}", modality="LINAC")
        test_id = crud.add_test(
            name=f"Test_{value}",
            data_type="float",
            reference_value=100.0,
            tolerance_low=98.0,
            tolerance_high=102.0,
            action_low=95.0,
            action_high=105.0,
        )
        actual_status = crud.log_result(unit_id=unit_id, test_id=test_id, value_float=value)
        assert actual_status == expected_status

        # Also verify in database row
        with db_setup.get_connection() as conn:
            cur = conn.execute("SELECT status FROM test_results WHERE unit_id=? AND test_id=?;", (unit_id, test_id))
            assert cur.fetchone()[0] == expected_status

    def test_unique_unit_name_constraint(self, temp_db):
        """Inserting a duplicate unit name must raise sqlite3.IntegrityError."""
        crud.add_unit(name="Unique TrueBeam", modality="LINAC")
        with pytest.raises(sqlite3.IntegrityError):
            crud.add_unit(name="Unique TrueBeam", modality="LINAC")


# ==============================================================================
# Tier 3: Combinatorial & Multi-Table CRUD Operations
# ==============================================================================

class TestMultiTableCRUD:
    """Verify record insertion and queries across all QA tables."""

    def test_calibration_log_crud(self, temp_db):
        """Verify TRS-398 reference dosimetry result logging and retrieval."""
        cal_id = crud.log_calibration(
            beam_type="photon",
            unit_name="Varian TrueBeam #1",
            institution="Medical Center",
            physicist="Medical Physicist",
            chamber="PTW 30013 Farmer",
            energy="6 MV",
            setup="SSD 100 cm, zref 10 cm",
            n_dw=0.05385,
            kq=0.9920,
            k_pol=0.9998,
            k_s=1.0098,
            k_tp=1.0104,
            k_vol_fff=1.0000,
            m_corrected=18.420,
            dose_mu_zref=0.995,
            dose_mu_zmax=1.485,
            notes="Annual output calibration",
        )
        assert cal_id > 0

        df_logs = crud.get_calibration_logs()
        assert len(df_logs) >= 1
        assert "Varian TrueBeam #1" in df_logs["unit_name"].values
        assert df_logs.iloc[0]["chamber"] == "PTW 30013 Farmer"
        assert df_logs.iloc[0]["kq"] == pytest.approx(0.9920, abs=1e-4)

    def test_service_log_crud(self, temp_db):
        """Verify machine service, maintenance, and downtime logging."""
        unit_id = crud.add_unit(name="Service LINAC", modality="LINAC")

        # Add service log
        log_id = crud.add_service_log(
            unit_id=unit_id,
            event_type="Maintenance",
            start_datetime="2026-10-28 08:00:00",
            end_datetime="2026-10-28 10:30:00",
            performed_by="Engineer X",
            description="MLC Routine Calibration",
        )
        assert log_id > 0

        # Retrieve service logs
        df_service = crud.get_service_logs(unit_id=unit_id)
        assert len(df_service) == 1
        assert df_service.iloc[0]["event_type"] == "Maintenance"
        assert df_service.iloc[0]["description"] == "MLC Routine Calibration"


# ==============================================================================
# Tier 4: Clinical Real-World Profile QA Logging Lifecycle
# ==============================================================================

class TestProfileQALogPersistence:
    """Verify full end-to-end lifecycle for profile_qa_logs table."""

    def test_profile_qa_save_and_retrieve_flattened(self, temp_db):
        """
        Clinical Scenario: Save a 6 MV Flattened water tank profile QA result
        and verify all geometric, penumbral, flatness, and symmetry fields.
        """
        unit_id = crud.add_unit(name="TrueBeam TB-1", modality="LINAC")

        log_id = crud.save_profile_qa_log(
            unit_name="TrueBeam TB-1",
            unit_id=unit_id,
            institution="Department of Medical Physics",
            physicist="Medical Physicist",
            beam_energy="6 MV",
            beam_type="Flattened",
            protocol="IEC 60976",
            data_source="Water Tank (CSV/ASCII)",
            axis="Crossplane (X)",
            field_size_nominal="10x10 cm",
            field_width_x_mm=100.08,
            cax_offset_x_mm=0.04,
            penumbra_left_x_mm=5.93,
            penumbra_right_x_mm=5.95,
            flatness_x=100.00,
            symmetry_x=1.000,
            symmetry_area_x=0.00,
            status="OK",
            notes="Commissioning benchmark profile verification.",
            raw_summary_json='{"status": "OK", "protocol": "IEC"}',
        )
        assert log_id > 0

        # Retrieve by log ID
        record = crud.get_profile_qa_by_id(log_id)
        assert record is not None
        assert record["unit_name"] == "TrueBeam TB-1"
        assert record["beam_type"] == "Flattened"
        assert record["field_width_x_mm"] == pytest.approx(100.08, abs=1e-2)
        assert record["penumbra_left_x_mm"] == pytest.approx(5.93, abs=1e-2)
        assert record["status"] == "OK"

        # Retrieve by unit DataFrame query
        df = crud.get_profile_qa_logs(unit_id=unit_id)
        assert len(df) == 1
        assert df.iloc[0]["physicist"] == "Medical Physicist"

    def test_profile_qa_save_and_retrieve_fff(self, temp_db):
        """
        Clinical Scenario: Save a 10 MV FFF unflattened beam profile QA run.
        """
        log_id = crud.save_profile_qa_log(
            unit_name="TrueBeam TB-2",
            physicist="Chief Physicist",
            beam_energy="10 MV FFF",
            beam_type="FFF",
            protocol="FFF Consensus",
            data_source="DICOM EPID",
            axis="Crossplane (X)",
            field_size_nominal="10x10 cm",
            field_width_x_mm=99.85,
            penumbra_left_x_mm=5.52,
            penumbra_right_x_mm=5.48,
            symmetry_x=0.05,
            left_slope_x=0.648,
            right_slope_x=-0.651,
            status="OK",
        )
        assert log_id > 0

        record = crud.get_profile_qa_by_id(log_id)
        assert record["beam_energy"] == "10 MV FFF"
        assert record["beam_type"] == "FFF"
        assert record["left_slope_x"] == pytest.approx(0.648, abs=1e-3)
        assert record["right_slope_x"] == pytest.approx(-0.651, abs=1e-3)

    def test_reference_beam_data_crud(self, temp_db):
        """Verify save, query, get_by_id, and delete for machine reference beam baselines."""
        import json
        unit_id = crud.add_unit(name="Edge-Research", modality="LINAC")
        dummy_curve = json.dumps({"x_mm": [-50.0, 0.0, 50.0], "values": [50.0, 100.0, 50.0]})

        ref_id = crud.save_reference_beam_data(
            unit_name="Edge-Research",
            unit_id=unit_id,
            energy="6 MV FFF",
            beam_type="FFF",
            curve_type="PDD",
            field_size_nominal="10x10 cm",
            curve_data_json=dummy_curve,
            tpr20_10=0.6325,
            pdd_10=63.2,
            pdd_20=34.5,
            d_max_mm=14.0,
            notes="Edge baseline PDD",
        )
        assert ref_id > 0

        # Query all
        df = crud.get_reference_beam_data(unit_name="Edge-Research")
        assert len(df) == 1
        assert df.iloc[0]["energy"] == "6 MV FFF"
        assert df.iloc[0]["tpr20_10"] == pytest.approx(0.6325)

        # Get by ID
        rec = crud.get_reference_beam_by_id(ref_id)
        assert rec is not None
        assert rec["curve_type"] == "PDD"

        # Query PDD helper
        pdd_rec = crud.get_reference_pdd_for_unit_energy("Edge-Research", "6 MV FFF")
        assert pdd_rec is not None
        assert pdd_rec["id"] == ref_id

        # Delete
        deleted = crud.delete_reference_beam_data(ref_id)
        assert deleted is True
        assert crud.get_reference_beam_by_id(ref_id) is None

    def test_ion_chambers_crud(self, temp_db):
        """Verify adding, querying, and deleting certified ionization chambers."""
        cid = crud.add_ion_chamber(
            model="PTW 30013 Farmer (0.60 cm³, waterproof)",
            serial_no="010482",
            ndw_cgy=5.3850,
            user_label="Clinical Farmer Primary",
            cal_date="2024-05-15",
            cal_lab="PTW-Freiburg SSDL",
            cert_id="CAL-2024-88910",
            notes="Annual calibration",
        )
        assert cid > 0

        # Query all
        df = crud.get_ion_chambers(active_only=True)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["serial_no"] == "010482"
        assert row["ndw_cgy"] == pytest.approx(5.3850)
        assert row["ndw_gy"] == pytest.approx(0.053850)
        assert row["cal_lab"] == "PTW-Freiburg SSDL"

        # Query by ID
        rec = crud.get_ion_chamber_by_id(cid)
        assert rec is not None
        assert rec["serial_no"] == "010482"

        # Update via upsert
        crud.add_ion_chamber(
            model="PTW 30013 Farmer (0.60 cm³, waterproof)",
            serial_no="010482",
            ndw_cgy=5.3900,
            user_label="Clinical Farmer Updated",
            cal_date="2025-05-15",
        )
        df_up = crud.get_ion_chambers()
        assert len(df_up) == 1
        assert df_up.iloc[0]["ndw_cgy"] == pytest.approx(5.3900)

        # Delete
        deleted = crud.delete_ion_chamber(cid)
        assert deleted is True
        assert crud.get_ion_chamber_by_id(cid) is None

    def test_unit_energies_edge_and_truebeam_non_matching(self, temp_db):
        """
        Clinical Scenario: Ensure Edge and TrueBeam maintain distinct non-matched
        beam energy parameters and baselines.
        """
        tb_id = crud.add_unit(name="TrueBeam", modality="LINAC")
        edge_id = crud.add_unit(name="Edge", modality="LINAC")

        # TrueBeam 6 MV FFF baseline
        tb_eid = crud.add_unit_energy(
            unit_id=tb_id,
            energy_name="6 MV FFF",
            beam_type="Photon FFF",
            nominal_mv=6.0,
            d_max_mm=13.8,
            tpr20_10=0.6316,
            pdd_10=63.2,
        )
        assert tb_eid > 0

        # Edge 6 MV FFF baseline (distinct, independent measurement)
        edge_eid = crud.add_unit_energy(
            unit_id=edge_id,
            energy_name="6 MV FFF",
            beam_type="Photon FFF",
            nominal_mv=6.0,
            d_max_mm=14.2,
            tpr20_10=0.6338,
            pdd_10=63.5,
        )
        assert edge_eid > 0

        tb_energies = crud.get_unit_energies(unit_id=tb_id)
        assert len(tb_energies) == 1
        assert tb_energies.iloc[0]["tpr20_10"] == pytest.approx(0.6316)
        assert tb_energies.iloc[0]["d_max_mm"] == pytest.approx(13.8)

        edge_energies = crud.get_unit_energies(unit_id=edge_id)
        assert len(edge_energies) == 1
        assert edge_energies.iloc[0]["tpr20_10"] == pytest.approx(0.6338)
        assert edge_energies.iloc[0]["d_max_mm"] == pytest.approx(14.2)

        # Ensure no cross-contamination
        assert edge_energies.iloc[0]["tpr20_10"] != tb_energies.iloc[0]["tpr20_10"]

    def test_chamber_manager_parser_synthetic_pdf(self, tmp_path):
        """
        Verify chamber calibration certificate PDF parser on a generated
        calibration certificate with SSDL metadata and calibration factor.
        """
        import pypdf
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        import chamber_manager as cm

        pdf_file = tmp_path / "ptw_calibration_cert.pdf"
        c = canvas.Canvas(str(pdf_file), pagesize=letter)
        c.drawString(100, 750, "CALIBRATION CERTIFICATE FOR IONIZATION CHAMBER")
        c.drawString(100, 720, "Laboratory: PTW-Freiburg SSDL Secondary Standards")
        c.drawString(100, 700, "Certificate Number: CAL-2024-PTW-88910")
        c.drawString(100, 680, "Chamber Model: PTW 30013 Farmer 0.60 cm3")
        c.drawString(100, 660, "Serial Number: 010482")
        c.drawString(100, 640, "Calibration Date: 2024-05-15")
        c.drawString(100, 620, "Calibration Factor N_D,w = 5.385 cGy/nC under Co-60")
        c.drawString(100, 600, "Reference Conditions: 20.0 C, 101.325 kPa")
        c.save()

        # Parse with chamber_manager
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()

        parsed = cm.parse_chamber_certificate_pdf(pdf_bytes)
        assert "30013" in parsed["model"]
        assert parsed["serial_no"] == "010482"
        assert parsed["ndw_cgy"] == pytest.approx(5.385, abs=1e-3)
        assert parsed["ndw_gy"] == pytest.approx(0.05385, abs=1e-5)
        assert "PTW" in parsed["cal_lab"]
        assert "CAL-2024-PTW-88910" in parsed["cert_id"]


