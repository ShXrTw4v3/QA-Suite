"""
test_adversarial_pdf.py — Adversarial Fuzzing, Stress & PHI Leak Test Suite
=============================================================================
Adversarial test harness for clinical_pdf_generator.py in Radiotherapy QA Suite.
Stress-tests:
  1. Deep PHI injection attacks across all 8 PDF report builders.
  2. Malformed input fuzzing (None, NaN, empty strings, oversized strings, negative values, unclosed XML).
  3. High-throughput process memory stability, Matplotlib figure cleanup, and open file descriptors.
  4. PDF byte stream integrity (%PDF-, %%EOF, multi-page NumberedCanvas counts).
=============================================================================
"""

import io
import os
import re
import math
import resource
import gc
import pytest
import numpy as np
import matplotlib.pyplot as plt

import clinical_pdf_generator as cpdf

ALL_BUILDERS = [
    ("generate_trs398_photon_pdf", cpdf.generate_trs398_photon_pdf),
    ("generate_trs398_electron_pdf", cpdf.generate_trs398_electron_pdf),
    ("generate_trs398_pdf", cpdf.generate_trs398_pdf),
    ("generate_trajectory_log_pdf", cpdf.generate_trajectory_log_pdf),
    ("generate_picket_fence_pdf", cpdf.generate_picket_fence_pdf),
    ("generate_winston_lutz_pdf", cpdf.generate_winston_lutz_pdf),
    ("generate_cbct_catphan_pdf", cpdf.generate_cbct_catphan_pdf),
    ("generate_beam_profile_pdf", cpdf.generate_beam_profile_pdf),
]

DISPATCH_PHI_TAGS = {
    "PatientName": "SECRET_JOHN_DOE_99",
    "PatientID": "SECRET_PAT_987654321",
    "OtherPatientIDs": "SECRET_OTHER_ID_112233",
    "MedicalRecordNumber": "SECRET_MRN_778899",
    "PatientBirthDate": "1975-08-24",
    "PatientSex": "SECRET_SEX_M",
    "EthnicGroup": "SECRET_ETHNIC_GROUP",
    "StudyDescription": "SECRET_STUDY_PELVIS",
}


# =============================================================================
# Pillar 1: PHI Injection Attacks
# =============================================================================

class TestAdversarialPHIInjection:
    """Stress tests PHI scrubbing across deeply nested and diverse DICOM tags."""

    @pytest.mark.parametrize("builder_name,builder_func", ALL_BUILDERS)
    def test_deeply_nested_phi_tags_do_not_leak(self, builder_name, builder_func):
        """Verify deeply nested DICOM patient tags do not leak into output byte stream."""
        nested_calc_results = {
            "PatientName": DISPATCH_PHI_TAGS["PatientName"],
            "PatientID": DISPATCH_PHI_TAGS["PatientID"],
            "OtherPatientIDs": DISPATCH_PHI_TAGS["OtherPatientIDs"],
            "MedicalRecordNumber": DISPATCH_PHI_TAGS["MedicalRecordNumber"],
            "PatientBirthDate": DISPATCH_PHI_TAGS["PatientBirthDate"],
            "PatientSex": DISPATCH_PHI_TAGS["PatientSex"],
            "EthnicGroup": DISPATCH_PHI_TAGS["EthnicGroup"],
            "StudyDescription": DISPATCH_PHI_TAGS["StudyDescription"],
            "nested_level_1": {
                "nested_level_2": {
                    "PatientName": DISPATCH_PHI_TAGS["PatientName"],
                    "MedicalRecordNumber": DISPATCH_PHI_TAGS["MedicalRecordNumber"],
                }
            },
            "nested_list": [
                {"MedicalRecordNumber": DISPATCH_PHI_TAGS["MedicalRecordNumber"]},
                {"EthnicGroup": DISPATCH_PHI_TAGS["EthnicGroup"]},
            ],
        }
        nested_unit_meta = {
            "institution": "University Clinical Center",
            "PatientName": DISPATCH_PHI_TAGS["PatientName"],
            "MedicalRecordNumber": DISPATCH_PHI_TAGS["MedicalRecordNumber"],
            "OtherPatientIDs": DISPATCH_PHI_TAGS["OtherPatientIDs"],
            "PatientBirthDate": DISPATCH_PHI_TAGS["PatientBirthDate"],
            "PatientSex": DISPATCH_PHI_TAGS["PatientSex"],
            "EthnicGroup": DISPATCH_PHI_TAGS["EthnicGroup"],
            "StudyDescription": DISPATCH_PHI_TAGS["StudyDescription"],
            "custom_meta": {
                "PatientID": DISPATCH_PHI_TAGS["PatientID"],
            }
        }

        pdf_bytes = builder_func(nested_calc_results, unit_meta=nested_unit_meta)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")

        for tag_name, secret_val in DISPATCH_PHI_TAGS.items():
            assert secret_val.encode() not in pdf_bytes, (
                f"PHI LEAK DETECTED in {builder_name}: {tag_name} ('{secret_val}') found in PDF byte stream!"
            )

    @pytest.mark.parametrize("builder_name,builder_func", ALL_BUILDERS)
    def test_string_embedded_phi_tags_do_not_leak(self, builder_name, builder_func):
        """Verify embedded PHI tags within string parameters are sanitized and do not leak."""
        embedded_unit_meta = {
            "unit_name": (
                "TrueBeam SN1234 - MedicalRecordNumber: MRN_EMBEDDED_9999, "
                "PatientBirthDate: 1980-01-01, PatientSex: F, EthnicGroup: CAUCASIAN, "
                "StudyDescription: PROSTATE_CLINICAL"
            ),
            "physicist": "MedicalRecordNumber: MRN_PHYSICIST_1111",
            "institution": "Hospital - OtherPatientIDs: OTHER_ID_445566",
        }
        embedded_calc = {
            "chamber": "PTW 30013 - MedicalRecordNumber: MRN_CHAMBER_2222",
            "filename": "12345678_TB1_PatientBirthDate_19850101.bin",
        }

        pdf_bytes = builder_func(embedded_calc, unit_meta=embedded_unit_meta)
        leaks = [
            "MRN_EMBEDDED_9999",
            "1980-01-01",
            "MRN_PHYSICIST_1111",
            "OTHER_ID_445566",
            "MRN_CHAMBER_2222",
        ]
        for leak in leaks:
            assert leak.encode() not in pdf_bytes, (
                f"PHI LEAK DETECTED in {builder_name}: embedded string token '{leak}' found in PDF!"
            )


# =============================================================================
# Pillar 2: Malformed Data Fuzzing Stress Tests
# =============================================================================

class TestAdversarialMalformedData:
    """Stress tests graceful degradation when receiving malformed, missing, or extreme data."""

    @pytest.mark.parametrize("builder_name,builder_func", ALL_BUILDERS)
    def test_none_values_resilience(self, builder_name, builder_func):
        """Pass None values for all expected fields. Must fall back gracefully without TypeError."""
        malformed_none = {
            "diff_pct": None,
            "k_tp": None,
            "k_pol": None,
            "k_s": None,
            "k_vol": None,
            "k_elec": None,
            "k_tot": None,
            "nd_w_gy_per_nc": None,
            "tpr2010": None,
            "k_q": None,
            "m_raw_avg_nc": None,
            "m_q_nc": None,
            "d_w_zref_cgy": None,
            "dose_per_mu_zref_cgy": None,
            "pdd_zref_pct": None,
            "d_w_zmax_cgy": None,
            "dose_per_mu_zmax_cgy": None,
            "target_cgy_per_mu": None,
            "recommended_mu_trim_pct": None,
            "r50_cm": None,
            "z_ref_cm": None,
            "overall_rms": None,
            "worst_rms": None,
            "MCS": None,
            "LSV": None,
            "AAV": None,
            "percent_passing": None,
            "max_error": None,
            "median_error": None,
            "tolerance": None,
            "action_tolerance": None,
            "max_2d": None,
            "mean_2d": None,
            "iso_diameter": None,
            "uniformity_index": None,
            "mtf_50": None,
            "slice_thickness_mm": None,
            "hu_r2": None,
            "field_width_mm": None,
            "flatness_pct": None,
            "symmetry_pct": None,
            "penumbra_left_mm": None,
            "penumbra_right_mm": None,
            "cax_shift_mm": None,
        }
        unit_meta = {"unit_name": None, "institution": None, "physicist": None}
        pdf_bytes = builder_func(malformed_none, unit_meta=unit_meta)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")

    @pytest.mark.parametrize("builder_name,builder_func", ALL_BUILDERS)
    def test_nan_floats_resilience(self, builder_name, builder_func):
        """Pass NaN floats. Must handle gracefully without crashing or leaking Matplotlib figures."""
        initial_fignums = plt.get_fignums()
        nan_data = {
            "diff_pct": float("nan"),
            "k_tp": float("nan"),
            "k_q": float("nan"),
            "tpr2010": float("nan"),
            "overall_rms": float("nan"),
            "worst_rms": float("nan"),
            "percent_passing": float("nan"),
            "max_error": float("nan"),
            "max_2d": float("nan"),
            "mean_2d": float("nan"),
            "uniformity_index": float("nan"),
            "mtf_50": float("nan"),
            "flatness_pct": float("nan"),
            "symmetry_pct": float("nan"),
        }
        try:
            pdf_bytes = builder_func(nan_data, unit_meta={})
            assert isinstance(pdf_bytes, bytes)
            assert pdf_bytes.startswith(b"%PDF-")
        finally:
            assert plt.get_fignums() == initial_fignums, (
                f"Matplotlib figure leaked during {builder_name} with NaN: {plt.get_fignums()}"
            )

    @pytest.mark.parametrize("builder_name,builder_func", ALL_BUILDERS)
    def test_empty_strings_resilience(self, builder_name, builder_func):
        """Pass empty strings for numerical fields. Must not raise ValueError: could not convert string to float."""
        empty_str_data = {
            "diff_pct": "",
            "overall_rms": "",
            "worst_rms": "",
            "percent_passing": "",
            "max_error": "",
            "max_2d": "",
            "uniformity_index": "",
            "flatness_pct": "",
        }
        pdf_bytes = builder_func(empty_str_data, unit_meta={"institution": "", "unit_name": ""})
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")

    @pytest.mark.parametrize("builder_name,builder_func", ALL_BUILDERS)
    def test_negative_values_resilience(self, builder_name, builder_func):
        """Pass negative values into numerical metrics (e.g. negative RMS, negative penumbra)."""
        neg_data = {
            "diff_pct": -8.5,
            "overall_rms": -0.5,
            "worst_rms": -1.2,
            "max_error": -0.4,
            "penumbra_left_mm": -5.0,
            "flatness_pct": -2.0,
        }
        initial_fignums = plt.get_fignums()
        try:
            pdf_bytes = builder_func(neg_data, unit_meta={})
            assert isinstance(pdf_bytes, bytes)
            assert pdf_bytes.startswith(b"%PDF-")
        finally:
            assert plt.get_fignums() == initial_fignums, (
                f"Matplotlib figure leaked during {builder_name} with negative inputs: {plt.get_fignums()}"
            )

    @pytest.mark.parametrize("builder_name,builder_func", ALL_BUILDERS)
    def test_oversized_strings_resilience(self, builder_name, builder_func):
        """Pass 5,000-character strings into text fields. Must not crash with ReportLab LayoutError."""
        oversized = "A" * 5000
        data = {"chamber": oversized, "filename": oversized, "delivery_type": oversized}
        meta = {"institution": oversized, "unit_name": oversized, "physicist": oversized}
        pdf_bytes = builder_func(data, unit_meta=meta)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")

    @pytest.mark.parametrize("builder_name,builder_func", ALL_BUILDERS)
    def test_xml_special_characters_resilience(self, builder_name, builder_func):
        """Pass XML tags and unclosed entities like '<b not closed' and '&'. Must not raise paraparser syntax error."""
        malicious_xml = "Clinic & Hospital <b not closed <script>alert(1)</script> > <"
        data = {"chamber": malicious_xml, "filename": malicious_xml}
        meta = {"institution": malicious_xml, "unit_name": malicious_xml, "physicist": malicious_xml}
        pdf_bytes = builder_func(data, unit_meta=meta)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")


# =============================================================================
# Pillar 3: High-Throughput & Resource Leak Stress Tests
# =============================================================================

class TestAdversarialResourceStability:
    """Stress tests memory consumption, file handle leaks, and figure cleanup over high throughput."""

    def test_50_consecutive_pdf_memory_and_cleanup(self):
        """Execute 50 consecutive PDF reports in a tight loop and verify stability."""
        initial_fignums = plt.get_fignums()
        initial_open_fds = len(os.listdir("/dev/fd")) if os.path.exists("/dev/fd") else 0
        mem_start_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

        for i in range(50):
            builder_name, builder_func = ALL_BUILDERS[i % len(ALL_BUILDERS)]
            pdf_bytes = builder_func({}, unit_meta={})
            assert len(pdf_bytes) > 5000
            assert pdf_bytes.startswith(b"%PDF-")

        gc.collect()
        mem_end_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        final_fignums = plt.get_fignums()
        final_open_fds = len(os.listdir("/dev/fd")) if os.path.exists("/dev/fd") else 0

        # Assert no figure leaks
        assert final_fignums == initial_fignums, f"Leaked matplotlib figures: {final_fignums}"

        # Assert file descriptor stability (leaking file descriptors indicates unclosed buffers/files)
        if initial_open_fds > 0:
            fd_delta = final_open_fds - initial_open_fds
            assert fd_delta <= 5, f"Leaked file descriptors: {fd_delta} new open FDs"

        # Assert memory growth is bounded (< 200 MB delta, < 500 MB total RSS)
        mem_growth_mb = mem_end_mb - mem_start_mb
        assert mem_growth_mb < 200.0, f"Excessive memory growth during 50 PDFs: {mem_growth_mb:.2f} MB"
        assert mem_end_mb < 500.0, f"Excessive process RSS: {mem_end_mb:.2f} MB"


# =============================================================================
# Pillar 4: PDF Byte Stream Integrity & Extreme Lengths
# =============================================================================

class TestAdversarialPDFByteIntegrity:
    """Verifies %PDF- header, %%EOF trailer, and NumberedCanvas page count consistency."""

    @pytest.mark.parametrize("builder_name,builder_func", ALL_BUILDERS)
    def test_pdf_header_and_eof_trailer(self, builder_name, builder_func):
        """Every PDF generated must strictly start with %PDF- and end with %%EOF."""
        pdf_bytes = builder_func({}, unit_meta={})
        assert pdf_bytes.startswith(b"%PDF-"), f"{builder_name} missing %PDF- header"
        assert b"%%EOF" in pdf_bytes[-128:], f"{builder_name} missing %%EOF trailer"

    def test_extreme_worst_leaves_list_page_budget(self):
        """Verify picket fence builder with 500 worst leaf records maintains byte integrity."""
        pf_extreme = {
            "percent_passing": 72.5,
            "max_error": 1.45,
            "worst_leaves": [
                {"picket": (i % 10) + 1, "leaf": (i % 120) + 1, "error": 0.5 + (i % 100) * 0.01}
                for i in range(500)
            ],
        }
        pdf_bytes = cpdf.generate_picket_fence_pdf(pf_extreme)
        assert pdf_bytes.startswith(b"%PDF-")
        assert b"%%EOF" in pdf_bytes[-128:]
        pages = re.findall(rb"Page \d+ of \d+", pdf_bytes)
        assert len(pages) > 0
        # Verify page string format
        for p in pages:
            match = re.match(rb"Page (\d+) of (\d+)", p)
            assert match is not None
            cur_p, tot_p = int(match.group(1)), int(match.group(2))
            assert cur_p <= tot_p
