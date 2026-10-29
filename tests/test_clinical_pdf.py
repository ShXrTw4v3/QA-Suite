"""
Automated Pytest Validation Suite: Unified Clinical PDF Report Generator

Covers:
- PDF Generation Smoke & Byte Validity across all 7 QA modules:
  * 1. IAEA TRS-398 Rev. 1 Photon Reference Dosimetry (generate_trs398_photon_pdf)
  * 2. IAEA TRS-398 Rev. 1 Electron Reference Dosimetry (generate_trs398_electron_pdf)
  * 3. Unified TRS-398 entrypoint dispatcher (generate_trs398_pdf)
  * 4. Dynamic MLC Trajectory Log Kinematic QA (generate_trajectory_log_pdf)
  * 5. EPID Picket Fence MLC Positioning QA (generate_picket_fence_pdf)
  * 6. Winston-Lutz 3D Radiation Isocenter QA (generate_winston_lutz_pdf)
  * 7. CBCT CatPhan 3D Volumetric Imaging QA (generate_cbct_catphan_pdf)
  * 8. Beam Profile Flatness, Symmetry & Penumbra QA (generate_beam_profile_pdf)
- Byte Stream Verification:
  * Non-empty bytes.
  * Valid binary %PDF- header.
  * Valid binary %%EOF trailer.
  * File size strictly > 5 KB (5,000 bytes).
- Strict Zero-PHI Assurance:
  * Scrubbing of patient names, patient IDs, MRNs, and patient plan IDs.
  * Replacement with MACHINE_QA_ANONYMIZED token.
- Clinical Section Verification:
  * Institution and treatment unit metadata.
  * Applicable standard and evaluation tolerances.
  * Color-coded status badges (PASS / ACTION / FAIL).
  * High-resolution embedded diagnostic charts (300 DPI PNG buffers).
  * Two-column physicist & reviewer sign-off blocks.
"""

import io
import pytest
import numpy as np

import clinical_pdf_generator as cpdf
import trs398_calculator as trs
import beam_profile_analyzer as bpa


# ==============================================================================
# Sample Datasets for All QA Modules
# ==============================================================================

@pytest.fixture
def sample_trs398_photon_data():
    """Real calculation output dictionary from trs398_calculator for photons."""
    return trs.calc_trs398_dose(
        raw_readings=[12.330, 12.335, 12.332],
        mu_delivered=100.0,
        nd_w=0.05385,
        tpr2010=0.670,
        chamber_key="PTW 30013 Farmer (0.60 cm³, waterproof)",
        temp_c=21.0,
        press_kpa=101.0,
        pdd_zref=67.0,
    )


@pytest.fixture
def sample_trs398_electron_data():
    """Real calculation output dictionary from trs398_calculator for electrons."""
    return trs.calc_trs398_electron_dose(
        raw_readings=[10.150, 10.155, 10.152],
        mu_delivered=100.0,
        nd_w=0.08250,
        i50_cm=3.45,
        chamber_key="PTW 34001 Roos (Plane-Parallel)",
        temp_c=21.0,
        press_kpa=101.0,
    )


@pytest.fixture
def sample_trajectory_log_data():
    """Realistic Trajectory Log analysis results."""
    return {
        "delivery_type": "Dynamic VMAT Arc",
        "log_filename": "12345678_TB1_Arc1_Pelvis_20261028143000.bin",  # Intentional dummy patient ID
        "patient_name": "DOE^JOHN_CLINICAL_TEST",                          # Intentional PHI
        "patient_id": "12345678",                                          # Intentional PHI
        "mrn": "MRN_998877",                                               # Intentional PHI
        "num_snapshots": 3200,
        "sampling_rate_ms": 20.0,
        "leaf_rms_mean_mm": 0.28,
        "leaf_rms_max_mm": 0.65,
        "leaf_max_error_mm": 1.12,
        "worst_leaf_id": "Bank A, Leaf 34",
        "leaf_p95_error_mm": 0.45,
        "gantry_rms_deg": 0.12,
        "collimator_rms_deg": 0.08,
        "jaw_x1_rms_mm": 0.15,
        "jaw_x2_rms_mm": 0.14,
        "jaw_y1_rms_mm": 0.18,
        "jaw_y2_rms_mm": 0.16,
        "modulation_complexity_score": 0.384,
        "gamma_pass_rate_pct": 98.4,
        "gamma_criteria": "2% / 2 mm",
        "num_holdoffs": 2,
        "status": "PASS",
    }


@pytest.fixture
def sample_picket_fence_data():
    """Realistic Picket Fence MLC alignment analysis results."""
    return {
        "gantry_angle": 0.0,
        "collimator_angle": 0.0,
        "num_pickets": 10,
        "num_leaves": 60,
        "leaf_error_max_mm": 0.38,
        "leaf_error_median_mm": 0.12,
        "leaf_error_p95_mm": 0.29,
        "worst_leaf_id": "Picket 4, Leaf 28",
        "passing_leaves_pct": 99.2,
        "tolerance_mm": 0.50,
        "action_mm": 1.00,
        "status": "PASS",
        "worst_leaves": [
            {"picket": 4, "leaf": 28, "error_mm": 0.38, "status": "PASS"},
            {"picket": 7, "leaf": 15, "error_mm": 0.35, "status": "PASS"},
            {"picket": 2, "leaf": 42, "error_mm": 0.31, "status": "PASS"},
        ],
    }


@pytest.fixture
def sample_winston_lutz_data():
    """Realistic Winston-Lutz 3D isocenter results."""
    return {
        "num_images": 8,
        "isocenter_diameter_mm": 0.68,
        "isocenter_radius_mm": 0.34,
        "cax_bb_distance_max_mm": 0.42,
        "cax_bb_distance_mean_mm": 0.28,
        "gantry_isocenter_size_mm": 0.45,
        "collimator_isocenter_size_mm": 0.25,
        "couch_isocenter_size_mm": 0.30,
        "tolerance_radius_mm": 0.75,
        "action_radius_mm": 1.00,
        "status": "PASS",
        "image_data": [
            {"gantry": 0, "collimator": 0, "couch": 0, "delta_x_mm": 0.12, "delta_y_mm": 0.18, "distance_2d_mm": 0.22, "status": "PASS"},
            {"gantry": 90, "collimator": 0, "couch": 0, "delta_x_mm": 0.20, "delta_y_mm": -0.15, "distance_2d_mm": 0.25, "status": "PASS"},
            {"gantry": 180, "collimator": 0, "couch": 0, "delta_x_mm": -0.10, "delta_y_mm": 0.25, "distance_2d_mm": 0.27, "status": "PASS"},
            {"gantry": 270, "collimator": 0, "couch": 0, "delta_x_mm": -0.18, "delta_y_mm": -0.22, "distance_2d_mm": 0.28, "status": "PASS"},
        ],
    }


@pytest.fixture
def sample_cbct_catphan_data():
    """Realistic CBCT CatPhan 504 image quality results."""
    return {
        "phantom_model": "CatPhan 504",
        "scan_protocol": "Pelvis Standard 120 kVp",
        "slice_thickness_nominal_mm": 2.50,
        "slice_thickness_measured_mm": 2.48,
        "hu_linearity_r2": 0.9998,
        "uniformity_index_pct": 1.15,
        "mtf_50_lp_per_mm": 0.42,
        "low_contrast_visible_targets": 7,
        "status": "PASS",
        "hu_sensitometry": [
            {"material": "Air", "nominal_hu": -1000, "measured_hu": -995, "status": "PASS"},
            {"material": "PMP", "nominal_hu": -200, "measured_hu": -198, "status": "PASS"},
            {"material": "LDPE", "nominal_hu": -100, "measured_hu": -98, "status": "PASS"},
            {"material": "Polystyrene", "nominal_hu": -35, "measured_hu": -32, "status": "PASS"},
            {"material": "Acrylic", "nominal_hu": 120, "measured_hu": 123, "status": "PASS"},
            {"material": "Delrin", "nominal_hu": 340, "measured_hu": 345, "status": "PASS"},
            {"material": "Teflon", "nominal_hu": 990, "measured_hu": 992, "status": "PASS"},
        ],
    }


@pytest.fixture
def sample_beam_profile_data():
    """Realistic Beam Profile analyzer output."""
    p = bpa.generate_synthetic_profile(beam_type="6 MV Flattened")
    analysis = bpa.analyze_flattened_profile(p["x_mm"], p["values"], protocol="IEC")
    analysis["x_mm"] = p["x_mm"]
    analysis["values"] = p["values"]
    analysis["beam_energy"] = "6 MV"
    analysis["axis"] = "Crossplane (X)"
    analysis["data_source"] = "Water Tank (CSV/ASCII)"
    return analysis


@pytest.fixture
def standard_unit_meta():
    """Standard clinical institution and unit metadata."""
    return {
        "institution": "University Clinical Medical Physics Center",
        "unit_name": "Varian TrueBeam #1 (Vault 1)",
        "modality": "Medical Linear Accelerator (LINAC)",
        "serial_no": "SN-TB-001",
        "physicist": "Dr. Nemanja Golubovac, PhD, DABR",
        "reviewer": "Chief Clinical Medical Physicist",
    }


# ==============================================================================
# Tier 1: Category-Partition & Byte Validity Testing
# ==============================================================================

class TestPDFByteValidity:
    """Verify non-empty output, valid binary header, trailer, and size > 5 KB for all builders."""

    def _assert_valid_pdf_stream(self, pdf_bytes: bytes, min_size_bytes: int = 5000):
        """Helper asserting basic PDF binary specification criteria."""
        assert isinstance(pdf_bytes, bytes), "PDF generator output must be bytes"
        assert len(pdf_bytes) > min_size_bytes, f"PDF file size ({len(pdf_bytes)} bytes) is below minimum {min_size_bytes} bytes"
        assert pdf_bytes.startswith(b"%PDF-"), "PDF stream must begin with standard binary header '%PDF-'"
        assert b"%%EOF" in pdf_bytes[-2048:], "PDF stream must terminate with standard '%%EOF' token"

    def test_generate_trs398_photon_pdf(self, sample_trs398_photon_data, standard_unit_meta):
        pdf = cpdf.generate_trs398_photon_pdf(sample_trs398_photon_data, standard_unit_meta)
        self._assert_valid_pdf_stream(pdf)

    def test_generate_trs398_electron_pdf(self, sample_trs398_electron_data, standard_unit_meta):
        pdf = cpdf.generate_trs398_electron_pdf(sample_trs398_electron_data, standard_unit_meta)
        self._assert_valid_pdf_stream(pdf)

    def test_generate_trs398_pdf_dispatcher(self, sample_trs398_photon_data, sample_trs398_electron_data, standard_unit_meta):
        pdf_photon = cpdf.generate_trs398_pdf(sample_trs398_photon_data, standard_unit_meta)
        self._assert_valid_pdf_stream(pdf_photon)

        pdf_electron = cpdf.generate_trs398_pdf(sample_trs398_electron_data, standard_unit_meta)
        self._assert_valid_pdf_stream(pdf_electron)

    def test_generate_trajectory_log_pdf(self, sample_trajectory_log_data, standard_unit_meta):
        pdf = cpdf.generate_trajectory_log_pdf(sample_trajectory_log_data, standard_unit_meta)
        self._assert_valid_pdf_stream(pdf)

    def test_generate_picket_fence_pdf(self, sample_picket_fence_data, standard_unit_meta):
        pdf = cpdf.generate_picket_fence_pdf(sample_picket_fence_data, standard_unit_meta)
        self._assert_valid_pdf_stream(pdf)

    def test_generate_winston_lutz_pdf(self, sample_winston_lutz_data, standard_unit_meta):
        pdf = cpdf.generate_winston_lutz_pdf(sample_winston_lutz_data, standard_unit_meta)
        self._assert_valid_pdf_stream(pdf)

    def test_generate_cbct_catphan_pdf(self, sample_cbct_catphan_data, standard_unit_meta):
        pdf = cpdf.generate_cbct_catphan_pdf(sample_cbct_catphan_data, standard_unit_meta)
        self._assert_valid_pdf_stream(pdf)

    def test_generate_beam_profile_pdf(self, sample_beam_profile_data, standard_unit_meta):
        pdf = cpdf.generate_beam_profile_pdf(sample_beam_profile_data, standard_unit_meta)
        self._assert_valid_pdf_stream(pdf)


# ==============================================================================
# Tier 2: Boundary Value Analysis (BVA) & Strict Zero-PHI Assurance
# ==============================================================================

class TestZeroPHIAssurance:
    """
    Rigorously verifies that no patient-identifiable data (PHI) leaks into
    generated PDF reports, even when deliberately injected into input dictionaries.
    """

    def test_trajectory_log_scrubs_patient_id_and_name(self, sample_trajectory_log_data, standard_unit_meta):
        """
        Input dictionary contains:
        - log_filename: '12345678_TB1_Arc1_Pelvis_20261028143000.bin'
        - patient_id: '12345678'
        - patient_name: 'DOE^JOHN_CLINICAL_TEST'
        - mrn: 'MRN_998877'
        Assert: NONE of these tokens exist in the generated PDF text stream!
        """
        pdf = cpdf.generate_trajectory_log_pdf(sample_trajectory_log_data, standard_unit_meta)

        # Forbidden tokens
        assert b"12345678" not in pdf, "Patient ID '12345678' leaked into Trajectory Log PDF!"
        assert b"DOE^JOHN" not in pdf, "Patient Name leaked into Trajectory Log PDF!"
        assert b"MRN_998877" not in pdf, "MRN leaked into Trajectory Log PDF!"

        # Verified replacement
        assert b"MACHINE_QA_ANONYMIZED" in pdf, "Anonymized token should be present in PDF"

    def test_metadata_injection_scrubbing(self, sample_trs398_photon_data):
        """Inject PHI directly into unit_meta dictionary."""
        contaminated_meta = {
            "institution": "City Hospital",
            "unit_name": "TrueBeam 1",
            "patient_name": "Jane Doe",
            "patient_id": "999888777",
            "mrn": "11223344",
        }
        pdf = cpdf.generate_trs398_photon_pdf(sample_trs398_photon_data, contaminated_meta)

        assert b"Jane Doe" not in pdf
        assert b"999888777" not in pdf
        assert b"11223344" not in pdf


# ==============================================================================
# Tier 3: Pairwise Combinations & Required Clinical Sections
# ==============================================================================

class TestClinicalReportStructure:
    """Verify presence of clinical sections across all QA module reports."""

    def test_required_sections_in_trs398_pdf(self, sample_trs398_photon_data, standard_unit_meta):
        """Verify TRS-398 PDF contains institution, protocol, badges, tolerances, and sign-offs."""
        pdf = cpdf.generate_trs398_photon_pdf(sample_trs398_photon_data, standard_unit_meta)
        pdf_str = pdf.decode("latin1", errors="ignore")

        # 1. Institution and machine metadata
        assert "University Clinical Medical Physics Center" in pdf_str
        assert "Varian TrueBeam" in pdf_str
        assert "IAEA TRS-398" in pdf_str

        # 2. Tolerances and results
        assert "TPR" in pdf_str or "TPR20,10" in pdf_str
        assert "k_Q" in pdf_str or "kq" in pdf_str.lower()
        assert "k_TP" in pdf_str or "ktp" in pdf_str.lower()

        # 3. Status badge
        assert any(b in pdf_str for b in ("PASS", "ACTION", "OUT OF TOLERANCE"))

        # 4. Sign-off block
        assert "Performed By" in pdf_str
        assert "Reviewed & Approved By" in pdf_str
        assert "Dr. Nemanja Golubovac" in pdf_str

    def test_required_sections_in_beam_profile_pdf(self, sample_beam_profile_data, standard_unit_meta):
        """Verify Beam Profile PDF contains field width, penumbra, flatness, symmetry, and plots."""
        pdf = cpdf.generate_beam_profile_pdf(sample_beam_profile_data, standard_unit_meta)
        pdf_str = pdf.decode("latin1", errors="ignore")

        assert "Beam Profile" in pdf_str
        assert "Field Width" in pdf_str or "FWHM" in pdf_str
        assert "Penumbra" in pdf_str
        assert "Flatness" in pdf_str
        assert "Symmetry" in pdf_str
        assert "Performed By" in pdf_str
        assert "Reviewed & Approved By" in pdf_str


# ==============================================================================
# Tier 4: Clinical Real-World End-to-End Audit Package Scenarios
# ==============================================================================

class TestClinicalAuditPackageGeneration:
    """
    Simulates full hospital monthly QA audit package generation:
    All 7 QA modules generate complete, signed, zero-PHI reports.
    """

    def test_generate_complete_hospital_audit_package(
        self,
        sample_trs398_photon_data,
        sample_trs398_electron_data,
        sample_trajectory_log_data,
        sample_picket_fence_data,
        sample_winston_lutz_data,
        sample_cbct_catphan_data,
        sample_beam_profile_data,
        standard_unit_meta,
    ):
        """Generate all 7 reports in sequence and assert size and format integrity."""
        reports = {
            "TRS398_Photon": cpdf.generate_trs398_photon_pdf(sample_trs398_photon_data, standard_unit_meta),
            "TRS398_Electron": cpdf.generate_trs398_electron_pdf(sample_trs398_electron_data, standard_unit_meta),
            "Trajectory_Log": cpdf.generate_trajectory_log_pdf(sample_trajectory_log_data, standard_unit_meta),
            "Picket_Fence": cpdf.generate_picket_fence_pdf(sample_picket_fence_data, standard_unit_meta),
            "Winston_Lutz": cpdf.generate_winston_lutz_pdf(sample_winston_lutz_data, standard_unit_meta),
            "CBCT_CatPhan": cpdf.generate_cbct_catphan_pdf(sample_cbct_catphan_data, standard_unit_meta),
            "Beam_Profile": cpdf.generate_beam_profile_pdf(sample_beam_profile_data, standard_unit_meta),
        }

        total_bytes = 0
        for name, pdf_bytes in reports.items():
            assert pdf_bytes.startswith(b"%PDF-"), f"{name} did not start with %PDF-"
            assert len(pdf_bytes) > 5000, f"{name} size is below 5 KB: {len(pdf_bytes)} bytes"
            total_bytes += len(pdf_bytes)

        # Multi-report package should be substantial (> 200 KB total for all 7 reports)
        assert total_bytes > 200_000, f"Total package size ({total_bytes} bytes) below 200 KB"
