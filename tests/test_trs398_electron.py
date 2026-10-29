"""
Automated Pytest Validation Suite: IAEA TRS-398 Rev. 1 (2024) High-Energy Electrons

Covers:
- Table 20 electron k_Q across ALL 8 chambers (2 plane-parallel, 6 cylindrical).
- Cylindrical chamber rejection: calling k_Q evaluation for cylindrical chambers
  when R_50 < 3.0 g/cm² strictly raises ValueError citing Table 19 & 20.
- Appendix II Eqs. (98)-(99) analytical fits vs Table 20 (tolerance <= 0.0010).
- Beam quality conversion: R_50 from I_50 (Eqs. 37-38 piecewise formula).
- Reference depth determination: z_ref = 0.6 * R_50 - 0.1 (Eq. 39).
- End-to-end clinical electron reference dosimetry (calc_trs398_electron_dose).
"""

import math
import pytest
import numpy as np

import trs398_calculator as trs


# ==============================================================================
# Tier 1: Category-Partition & Chamber Resolution
# ==============================================================================

class TestElectronChamberCategories:
    """Verify electron chamber classification, catalog size, and fuzzy resolution."""

    def test_total_electron_chambers(self):
        """Assert exactly 8 chambers in Table 20 (2 plane-parallel, 6 cylindrical)."""
        assert len(trs.KQ_ELECTRON_CHAMBERS) == 8
        assert len(trs.KQ_ELECTRON_FIT_PARAMS) == 8

        pp_chambers = [k for k, v in trs.KQ_ELECTRON_FIT_PARAMS.items() if v[3] == "pp"]
        cyl_chambers = [k for k, v in trs.KQ_ELECTRON_FIT_PARAMS.items() if v[3] == "cyl"]
        assert len(pp_chambers) == 2
        assert len(cyl_chambers) == 6

    @pytest.mark.parametrize(
        "query,expected_full_key",
        [
            ("Roos", "PTW 34001 Roos (Plane-Parallel)"),
            ("34001", "PTW 34001 Roos (Plane-Parallel)"),
            ("NACP", "IBA NACP-02 (Plane-Parallel)"),
            ("30013", "PTW 30013 Farmer (Cylindrical)"),
            ("30012", "PTW 30012 Farmer (Cylindrical)"),
            ("FC65-G", "IBA FC65-G Farmer (Cylindrical)"),
            ("2571", "NE 2571 Farmer (Cylindrical)"),
            ("A12", "Exradin A12 Farmer (Cylindrical)"),
            ("A19", "Exradin A19 Classic Farmer (Cylindrical)"),
        ],
    )
    def test_electron_chamber_fuzzy_resolution(self, query, expected_full_key):
        """Verify find_electron_chamber_key correctly resolves short strings."""
        assert trs.find_electron_chamber_key(query) == expected_full_key

    def test_unknown_electron_chamber_fallback(self):
        """Unknown chamber query falls back safely to default plane-parallel Roos."""
        resolved = trs.find_electron_chamber_key("NonExistentElectronChamber")
        assert resolved == "PTW 34001 Roos (Plane-Parallel)"


# ==============================================================================
# Tier 2: Boundary Value Analysis (BVA) & Safety Rejections
# ==============================================================================

CYLINDRICAL_CHAMBERS = [k for k, v in trs.KQ_ELECTRON_FIT_PARAMS.items() if v[3] == "cyl"]
PLANE_PARALLEL_CHAMBERS = [k for k, v in trs.KQ_ELECTRON_FIT_PARAMS.items() if v[3] == "pp"]


class TestElectronBoundaryValuesAndSafety:
    """Verify R50 boundaries, cylindrical rejection for R50 < 3.0 g/cm², and I50 transitions."""

    @pytest.mark.parametrize("chamber", CYLINDRICAL_CHAMBERS)
    @pytest.mark.parametrize("r50", [1.0, 1.4, 2.0, 2.5, 2.99])
    def test_cylindrical_chambers_strictly_rejected_below_3_cm(self, chamber, r50):
        """
        IAEA TRS-398 Rev. 1 Table 19 & Table 20 Requirement:
        Calling k_Q evaluation for cylindrical chambers when R_50 < 3.0 g/cm²
        MUST raise ValueError citing Table 19/20.
        """
        # Test analytical function
        with pytest.raises(ValueError, match="only recommended for R_50 >= 3.0 g/cm²"):
            trs.calc_kq_electron_analytical(chamber, r50)

        # Test interpolation function
        with pytest.raises(ValueError, match="requires R_50 >= 3.0 g/cm²"):
            trs.interpolate_kq_electron(chamber, r50)

    @pytest.mark.parametrize("chamber", CYLINDRICAL_CHAMBERS)
    def test_cylindrical_chambers_valid_at_boundary_3_cm(self, chamber):
        """Boundary test: R_50 = 3.00 g/cm² MUST succeed for all cylindrical chambers."""
        val = trs.interpolate_kq_electron(chamber, 3.0)
        assert val is not None
        assert 0.9000 <= val <= 0.9500

    @pytest.mark.parametrize("chamber", PLANE_PARALLEL_CHAMBERS)
    @pytest.mark.parametrize("r50", [1.0, 1.4, 2.0, 2.5, 3.0, 10.0])
    def test_plane_parallel_chambers_valid_across_full_range(self, chamber, r50):
        """Plane-parallel chambers are valid across the entire R50 range 1.0 to 10.0 g/cm²."""
        val = trs.interpolate_kq_electron(chamber, r50)
        assert val is not None
        assert 0.8800 <= val <= 0.9800

    def test_r50_from_i50_piecewise_formula(self):
        """
        IAEA TRS-398 Rev. 1 Eqs. (37)-(38):
        - For I_50 <= 10 cm: R_50 = 1.029 * I_50 - 0.06
        - For I_50 > 10 cm:  R_50 = 1.059 * I_50 - 0.37
        """
        # Test I_50 = 4.0 cm -> R_50 = 1.029 * 4.0 - 0.06 = 4.056
        assert trs.calc_electron_r50_from_i50(4.0) == pytest.approx(4.056, abs=1e-5)

        # Boundary test at I_50 = 10.0 cm -> 1.029 * 10.0 - 0.06 = 10.230
        assert trs.calc_electron_r50_from_i50(10.0) == pytest.approx(10.230, abs=1e-5)

        # Test I_50 = 12.0 cm -> R_50 = 1.059 * 12.0 - 0.37 = 12.338
        assert trs.calc_electron_r50_from_i50(12.0) == pytest.approx(12.338, abs=1e-5)

    def test_zref_evaluation_eq39(self):
        """
        IAEA TRS-398 Rev. 1 Eq. (39):
            z_ref = 0.6 * R_50 - 0.1
        """
        assert trs.calc_electron_zref(4.0) == pytest.approx(2.30, abs=1e-5)
        assert trs.calc_electron_zref(1.0) == pytest.approx(0.50, abs=1e-5)
        assert trs.calc_electron_zref(10.0) == pytest.approx(5.90, abs=1e-5)


# ==============================================================================
# Tier 3: Pairwise Combinations (Table 20 Nodes & Analytical Fits)
# ==============================================================================

class TestTable20AllChambers:
    """Verifies Table 20 node interpolation and analytical fits across all 8 chambers."""

    @pytest.mark.parametrize("chamber", trs.KQ_ELECTRON_CHAMBERS.keys())
    def test_table20_node_exact_interpolation(self, chamber):
        """Verify interpolate_kq_electron matches Table 20 at every valid node."""
        r50_nodes = trs.KQ_ELECTRON_R50S
        table_vals = trs.KQ_ELECTRON_CHAMBERS[chamber]

        for r50, expected_kq in zip(r50_nodes, table_vals):
            if expected_kq is not None:
                calc_val = trs.interpolate_kq_electron(chamber, r50)
                assert calc_val == expected_kq, f"Mismatch for {chamber} at R50={r50}"

    @pytest.mark.parametrize("chamber", PLANE_PARALLEL_CHAMBERS)
    def test_plane_parallel_analytical_fit_eq98(self, chamber):
        """
        Eq. (98): k_Q = a + b * exp(-R_50 / c)
        Assert analytical fit matches Table 20 within <= 0.0010 across all 14 points.
        """
        r50_nodes = trs.KQ_ELECTRON_R50S
        table_vals = trs.KQ_ELECTRON_CHAMBERS[chamber]

        for r50, expected_kq in zip(r50_nodes, table_vals):
            fit_val = trs.calc_kq_electron_analytical(chamber, r50)
            diff = abs(fit_val - expected_kq)
            assert diff <= 0.0010, (
                f"Eq. 98 fit deviation for {chamber} at R50={r50}: "
                f"fit={fit_val:.4f}, table={expected_kq:.4f}, diff={diff:.4f}"
            )

    @pytest.mark.parametrize("chamber", CYLINDRICAL_CHAMBERS)
    def test_cylindrical_analytical_fit_eq99(self, chamber):
        """
        Eq. (99): k_Q = a + b * (R_50 ** (-c))
        Assert analytical fit matches Table 20 within <= 0.0010 for all R_50 >= 3.0 g/cm².
        """
        r50_nodes = trs.KQ_ELECTRON_R50S
        table_vals = trs.KQ_ELECTRON_CHAMBERS[chamber]

        for r50, expected_kq in zip(r50_nodes, table_vals):
            if expected_kq is not None:
                fit_val = trs.calc_kq_electron_analytical(chamber, r50)
                diff = abs(fit_val - expected_kq)
                assert diff <= 0.0010, (
                    f"Eq. 99 fit deviation for {chamber} at R50={r50}: "
                    f"fit={fit_val:.4f}, table={expected_kq:.4f}, diff={diff:.4f}"
                )


# ==============================================================================
# Tier 4: Clinical Real-World End-to-End Scenarios
# ==============================================================================

class TestClinicalElectronReferenceDosimetry:
    """End-to-end clinical electron reference dosimetry pipeline tests."""

    def test_truebeam_9mev_electron_calibration_pipeline(self, standard_dosimetry_inputs):
        """
        Clinical Scenario: Varian TrueBeam 9 MeV electron beam reference dosimetry:
        - Roos plane-parallel chamber (PTW 34001)
        - i50 = 3.45 cm -> R50 = 3.49 g/cm²
        - z_ref = 0.6 * 3.49 - 0.1 = 1.994 cm
        """
        inputs = standard_dosimetry_inputs["electron_9mev"]
        result = trs.calc_trs398_electron_dose(
            raw_readings=inputs["readings"],
            mu_delivered=100.0,
            nd_w=inputs["n_dw"],
            i50_cm=inputs["i50_cm"],
            chamber_key=inputs["chamber"],
            temp_c=inputs["temp_c"],
            press_kpa=inputs["press_kpa"],
            m_plus=inputs["m_user"],
            m_minus=inputs["m_opposite"],
            v1=inputs["v_user"],
            v2=inputs["v_reduced"],
            m1=inputs["m_user"],
            m2=inputs["m_reduced"],
        )

        assert result["protocol_edition"] == trs.PROTOCOL_EDITION
        assert result["r50_cm"] == pytest.approx(1.029 * 3.45 - 0.06, rel=1e-4)
        assert result["z_ref_cm"] == pytest.approx(0.6 * result["r50_cm"] - 0.1, rel=1e-4)
        assert 0.9200 <= result["kq"] <= 0.9400
        assert 1.0000 <= result["k_tp"] <= 1.0200
        assert result["dose_per_mu_zref_cgy"] > 0.50

    def test_electron_pipeline_cylindrical_safety_interlock(self):
        """
        Safety Test: Attempting to calibrate a 6 MeV beam (R_50 < 3.0)
        using a cylindrical chamber in calc_trs398_electron_dose MUST fail.
        """
        with pytest.raises(ValueError, match="requires R_50 >= 3.0 g/cm²"):
            trs.calc_trs398_electron_dose(
                raw_readings=[10.0, 10.0, 10.0],
                r50_cm=2.1,  # Below 3.0 g/cm²
                chamber_key="PTW 30013 Farmer (Cylindrical)",
            )
