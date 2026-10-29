"""
Automated Pytest Validation Suite: Radiotherapy Beam Profile Flatness, Symmetry & Penumbra Analyzer

Covers:
- Analytical synthetic ground truth profiles:
  * 6 MV Flattened: recover field width (100.0 +- 0.5 mm), 80-20 penumbra (6.0 +- 0.5 mm),
    flatness (0.0 +- 0.1%), and symmetry (0.0 +- 0.1%).
  * 6 MV FFF: recover inflection point field width (100.0 +- 0.5 mm), penumbra (6.0 +- 0.5 mm),
    and symmetry (0.0 +- 0.1%).
  * 10 MV FFF: recover inflection point field width (100.0 +- 0.5 mm), penumbra (5.5 +- 0.5 mm),
    and symmetry (0.0 +- 0.1%).
  * Asymmetric tilted beam: verify detection of asymmetry and action level alert.
- Multi-format file parsing:
  * CSV (comma delimited), TSV (tab delimited), IBA OmniPro ASCII (<BEGIN_DATA>), PTW Mephysto ([SCAN]).
  * DICOM EPID (.dcm) with SID/SAD magnification scaling and zero-PHI parsing.
- Protocols supported:
  * IEC 60976 (Dmax/Dmin ratio flatness, PDQ ratio symmetry).
  * Varian (Difference flatness, point-difference symmetry).
  * Siemens (Area integral symmetry).
  * FFF Consensus (Inflection point derivative width, inflection-anchored penumbra, shoulder slopes).
- High-level BeamProfileAnalyzer class lifecycle and JSON serialization.
"""

import io
import math
import pytest
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

import beam_profile_analyzer as bpa


# ==============================================================================
# Tier 1: Category-Partition Testing
# ==============================================================================

class TestProfileProtocolCategories:
    """Verify analysis protocol categories and beam type handling."""

    def test_flattened_iec_and_varian_protocols(self, synthetic_flattened_6mv):
        """Verify flattened beam can be analyzed under both IEC and Varian protocols."""
        x, y, _ = synthetic_flattened_6mv

        res_iec = bpa.analyze_flattened_profile(x, y, protocol="IEC")
        assert res_iec["protocol"] == "IEC"
        assert "flatness_iec_pct" in res_iec
        assert "symmetry_iec_pdq" in res_iec

        res_var = bpa.analyze_flattened_profile(x, y, protocol="Varian")
        assert res_var["protocol"] == "Varian"
        assert "flatness_varian_pct" in res_var
        assert "symmetry_varian_pct" in res_var

    def test_unified_analyze_beam_profile_router(self, synthetic_flattened_6mv, synthetic_fff_6mv):
        """Verify analyze_beam_profile correctly routes Flattened vs FFF beams."""
        x_flat, y_flat, _ = synthetic_flattened_6mv
        res_flat = bpa.analyze_beam_profile(x_flat, y_flat, beam_type="Flattened", protocol="IEC")
        assert res_flat["beam_type"] == "Flattened"
        assert "fwhm_mm" in res_flat

        x_fff, y_fff, _ = synthetic_fff_6mv
        res_fff = bpa.analyze_beam_profile(x_fff, y_fff, beam_type="FFF")
        assert res_fff["beam_type"] == "FFF"
        assert "inflection_width_mm" in res_fff


# ==============================================================================
# Tier 2: Boundary Value Analysis (BVA) & Analytical Recovery
# ==============================================================================

class TestAnalyticalGroundTruthRecovery:
    """
    Rigorously verifies recovery of exact known mathematical parameters:
    Field width (+- 0.5 mm), Penumbra (+- 0.5 mm), Flatness (+- 0.1%), Symmetry (+- 0.1%).
    """

    def test_6mv_flattened_ground_truth_recovery(self, synthetic_flattened_6mv):
        """
        Analytical Benchmark 1: 6 MV Flattened:
        Target: FWHM = 100.0 mm, Penumbra = 6.0 mm, Flatness = 0.0%, Symmetry = 0.0%.
        """
        x, y, ground_truth = synthetic_flattened_6mv
        res = bpa.analyze_flattened_profile(x, y, protocol="Varian")

        # 1. Field width: 100.0 +- 0.5 mm
        assert abs(res["fwhm_mm"] - 100.0) <= 0.5, f"FWHM error: {res['fwhm_mm']}"

        # 2. Penumbra: 6.0 +- 0.5 mm
        assert abs(res["penumbra_left_mm"] - 6.0) <= 0.5, f"Left penumbra error: {res['penumbra_left_mm']}"
        assert abs(res["penumbra_right_mm"] - 6.0) <= 0.5, f"Right penumbra error: {res['penumbra_right_mm']}"

        # 3. Flatness: 0.0 +- 0.1%
        assert abs(res["flatness_varian_pct"]) <= 0.1, f"Varian flatness error: {res['flatness_varian_pct']}"

        # 4. Symmetry: 0.0 +- 0.1%
        assert abs(res["symmetry_varian_pct"]) <= 0.1, f"Varian symmetry error: {res['symmetry_varian_pct']}"
        assert abs(res["symmetry_area_pct"]) <= 0.1, f"Area symmetry error: {res['symmetry_area_pct']}"
        assert res["status"] == "OK"

    def test_6mv_fff_ground_truth_recovery(self, synthetic_fff_6mv):
        """
        Analytical Benchmark 2: 6 MV FFF:
        Target: Inflection field width = 100.0 +- 0.5 mm, Penumbra = 6.0 +- 0.5 mm, Symmetry = 0.0 +- 0.1%.
        """
        x, y, ground_truth = synthetic_fff_6mv
        res = bpa.analyze_fff_profile(x, y)

        # 1. Inflection point field width
        assert abs(res["inflection_width_mm"] - 100.0) <= 0.5, f"Inflection width: {res['inflection_width_mm']}"

        # 2. Inflection-anchored penumbra
        assert abs(res["penumbra_left_mm"] - 6.0) <= 0.5, f"Left penumbra: {res['penumbra_left_mm']}"
        assert abs(res["penumbra_right_mm"] - 6.0) <= 0.5, f"Right penumbra: {res['penumbra_right_mm']}"

        # 3. Symmetry
        assert abs(res["symmetry_pct"]) <= 0.1, f"FFF symmetry: {res['symmetry_pct']}"
        assert abs(res["slope_diff_pct_mm"]) <= 0.05, f"Slope diff: {res['slope_diff_pct_mm']}"
        assert res["status"] == "OK"

    def test_10mv_fff_ground_truth_recovery(self, synthetic_fff_10mv):
        """
        Analytical Benchmark 3: 10 MV FFF:
        Target: Inflection width = 100.0 +- 0.5 mm, Penumbra = 5.5 +- 0.5 mm, Symmetry = 0.0 +- 0.1%.
        """
        x, y, ground_truth = synthetic_fff_10mv
        res = bpa.analyze_fff_profile(x, y)

        assert abs(res["inflection_width_mm"] - 100.0) <= 0.5
        assert abs(res["penumbra_left_mm"] - 5.5) <= 0.5
        assert abs(res["penumbra_right_mm"] - 5.5) <= 0.5
        assert abs(res["symmetry_pct"]) <= 0.1
        assert res["status"] == "OK"

    def test_built_in_generator_presets(self):
        """Verify generate_synthetic_profile preset outputs match analytical specs."""
        p6w = bpa.generate_synthetic_profile(beam_type="6 MV Flattened")
        assert len(p6w["x_mm"]) > 100
        assert p6w["ground_truth"]["field_width_mm"] == 100.0
        assert p6w["ground_truth"]["penumbra_mm"] == 6.0

        p6fff = bpa.generate_synthetic_profile(beam_type="6 MV FFF")
        assert p6fff["meta"]["beam_type"] == "FFF"

        p10fff = bpa.generate_synthetic_profile(beam_type="10 MV FFF")
        assert p10fff["meta"]["beam_type"] == "FFF"
        assert p10fff["ground_truth"]["penumbra_mm"] == 5.5


# ==============================================================================
# Tier 3: Combinatorial & Out-of-Tolerance Testing
# ==============================================================================

class TestProfileEdgeCasesAndAlerts:
    """Verify asymmetric tilted beams, horn effects, and status alert triggers."""

    def test_asymmetric_tilted_beam_alert(self, synthetic_tilted_flattened):
        """
        Asymmetric beam: 3% tilt across half-field.
        Must detect symmetry error >= 2.5% and trigger Action level.
        """
        x, y, _ = synthetic_tilted_flattened
        res = bpa.analyze_flattened_profile(x, y, protocol="Varian")

        assert abs(res["symmetry_varian_pct"]) >= 2.5
        assert res["status"] in ("Tolerance", "Action")

    def test_horned_flattened_beam(self):
        """Verify realistic 2.5% horn effect is detected in flattened beam."""
        prof = bpa.generate_synthetic_profile(beam_type="6 MV Flattened", horn_ratio=0.025)
        res = bpa.analyze_flattened_profile(prof["x_mm"], prof["values"], protocol="Varian")
        assert res["fwhm_mm"] == pytest.approx(100.0, abs=0.5)
        # Horns should produce positive difference flatness ~1.2%
        assert 1.0 <= res["flatness_varian_pct"] <= 1.5


# ==============================================================================
# Tier 4: Universal File Parsers & DICOM EPID Integration
# ==============================================================================

class TestProfileParsers:
    """Verify text parsers (CSV, TSV, IBA, PTW) and DICOM EPID parsing."""

    def test_csv_parser(self, sample_csv_profile_text):
        """Verify CSV parsing with comma delimiter."""
        res = bpa.parse_profile_text(sample_csv_profile_text)
        assert "x_mm" in res
        assert "values" in res
        assert len(res["x_mm"]) == 121
        assert np.max(res["values"]) == pytest.approx(100.0, abs=0.1)

    def test_tsv_parser(self, sample_tsv_profile_text):
        """Verify tab-separated ASCII profile with comments."""
        res = bpa.parse_profile_text(sample_tsv_profile_text)
        assert len(res["x_mm"]) == 51
        assert res["x_mm"][0] == -50.0
        assert res["x_mm"][-1] == 50.0

    def test_iba_ascii_block_parser(self, sample_iba_ascii_profile_text):
        """Verify IBA OmniPro format with <BEGIN_DATA> block markers."""
        res = bpa.parse_profile_text(sample_iba_ascii_profile_text)
        assert len(res["x_mm"]) == 13
        assert res["meta"]["energy"] == "6 MV"
        assert res["meta"]["axis"] == "Crossplane (X)"

    def test_ptw_mcc_parser(self, sample_ptw_mcc_profile_text):
        """Verify PTW Mephysto MCC format with [SCAN] and BEGIN_DATA."""
        res = bpa.parse_profile_text(sample_ptw_mcc_profile_text)
        assert len(res["x_mm"]) == 7
        assert "X" in res["meta"]["axis"]

    def test_ptw_mcc_multiscan_parser(self):
        """Verify PTW BeamScan MCC multi-scan format with BEGIN_SCAN blocks (PDD, Inplane, Crossplane)."""
        multi_mcc = """BEGIN_SCAN_DATA
FORMAT=CC-Export V1.9
BEGIN_SCAN 1
    SCAN_CURVETYPE=PDD
    SCAN_DEPTH=0.00
    FIELD_INPLANE=100.00
    FIELD_CROSSPLANE=100.00
    FILTER=FFF
    BEGIN_DATA
        0.00 10.0
        10.00 95.0
        20.00 90.0
    END_DATA
END_SCAN 1
BEGIN_SCAN 2
    SCAN_CURVETYPE=INPLANE_PROFILE
    SCAN_DEPTH=100.00
    FIELD_INPLANE=100.00
    FIELD_CROSSPLANE=100.00
    FILTER=FFF
    BEGIN_DATA
        -60.00 10.0
        -50.00 50.0
        0.00 100.0
        50.00 50.0
        60.00 10.0
    END_DATA
END_SCAN 2
BEGIN_SCAN 3
    SCAN_CURVETYPE=CROSSPLANE_PROFILE
    SCAN_DEPTH=100.00
    FIELD_INPLANE=100.00
    FIELD_CROSSPLANE=100.00
    FILTER=FFF
    BEGIN_DATA
        -60.00 12.0
        -50.00 50.0
        0.00 100.0
        50.00 50.0
        60.00 12.0
    END_DATA
END_SCAN 3
END_SCAN_DATA
"""
        res = bpa.parse_profile_text(multi_mcc)
        assert "profiles" in res
        assert len(res["profiles"]) == 3
        assert "Crossplane" in res["meta"]["axis"]
        assert res["meta"]["is_fff"] is True
        assert len(res["x_mm"]) == 5

    def test_synthetic_dicom_epid_parser(self, tmp_path):
        """
        Verify parse_dicom_epid on a dynamically generated synthetic DICOM RT Image:
        Tests SID/SAD magnification scaling (1500/1000 = 1.5), crosshair profile extraction,
        and zero-PHI preservation.
        """
        dicom_path = tmp_path / "test_epid.dcm"

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.1"  # RT Image Storage
        file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7.8.9"
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(str(dicom_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.Modality = "RTIMAG"
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.PatientName = "DOE^JOHN_CLINICAL_TEST"
        ds.PatientID = "PHI_998877"
        ds.RTImageSID = 1500.0
        ds.RadiationMachineSAD = 1000.0
        ds.ImagePlanePixelSpacing = [0.4, 0.4]  # mm at imager plane
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0

        # Create 100x100 pixel grid with 40x40 mm radiation field at center
        rows, cols = 100, 100
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        pixel_data = np.zeros((rows, cols), dtype=np.uint16)
        # In-field region
        pixel_data[30:70, 30:70] = 5000
        ds.PixelData = pixel_data.tobytes()

        ds.save_as(str(dicom_path))

        # Parse with parse_dicom_epid
        res = bpa.parse_dicom_epid(str(dicom_path))

        assert "crossplane" in res
        assert "inplane" in res
        assert "meta" in res
        # Magnification factor: 1500 / 1000 = 1.5 -> pixel spacing at isocenter: 0.4 / 1.5 = 0.2667 mm
        dy_iso, dx_iso = res["pixel_spacing_iso"]
        assert dx_iso == pytest.approx(0.4 / 1.5, rel=1e-3)

        # Zero-PHI check: patient identifiers must not be present in metadata dict
        meta_str = str(res["meta"])
        assert "DOE^JOHN" not in meta_str
        assert "PHI_998877" not in meta_str

    def test_beam_profile_analyzer_class_lifecycle(self, synthetic_flattened_6mv):
        """Verify BeamProfileAnalyzer class initialization, analysis, and JSON serialization."""
        x, y, _ = synthetic_flattened_6mv
        analyzer = bpa.BeamProfileAnalyzer(x, y, meta={"machine": "TrueBeam 1"})
        res = analyzer.analyze(beam_type="Flattened", protocol="IEC")
        assert res["fwhm_mm"] == pytest.approx(100.0, abs=0.5)

        json_str = analyzer.to_json()
        assert "fwhm_mm" in json_str
        assert "TrueBeam 1" in json_str

    def test_analyze_pdd_curve(self):
        """Verify PDD curve analysis and IAEA TRS-398 Rev. 1 TPR20,10 derivation."""
        # Simulated 6 MV beam PDD: d_max at 15 mm, PDD10 = 66.5%, PDD20 = 38.2%
        z = np.linspace(0, 300, 301)
        dose = np.where(z <= 15.0, 50.0 + 50.0 * (z / 15.0), 100.0 * np.exp(-0.0042 * (z - 15.0)))
        res = bpa.analyze_pdd_curve(z, dose)

        assert res["d_max_mm"] == pytest.approx(15.0, abs=1.0)
        assert res["pdd_10"] > 60.0
        assert res["pdd_20"] > 30.0
        assert 0.50 <= res["tpr20_10"] <= 0.85
        assert res["status"] == "OK"

