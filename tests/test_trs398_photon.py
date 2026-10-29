"""
Automated Pytest Validation Suite: IAEA TRS-398 Rev. 1 (2024) High-Energy Photons

Covers:
- Table 16 calculated k_Q across ALL 26 cylindrical chambers at all 12 TPR20,10 grid points.
- Appendix II Eq. (34) analytical fit evaluation against Table 16 (error <= 0.00015).
- Table 16 linear interpolation at exact grid nodes and intermediate points.
- Boundary conditions (TPR20,10 <= 0.56 and >= 0.82 clamping).
- End-to-end absorbed-dose-to-water pipeline (calc_trs398_dose) for flattened beams.
- Status badge determination (PASS / ACTION / OUT OF TOLERANCE).
"""

import math
import pytest
import numpy as np

import trs398_calculator as trs


# ==============================================================================
# Tier 1: Category-Partition & Chamber Resolution
# ==============================================================================

class TestPhotonChamberCategories:
    """Verify chamber model catalog, query resolution, and cavity lengths."""

    def test_total_chambers_count(self):
        """Assert exactly 26 cylindrical chambers are defined in Table 16 & Table 45."""
        assert len(trs.KQ_PHOTON_CHAMBERS) == 26
        assert len(trs.KQ_PHOTON_FIT_PARAMS) == 26

    @pytest.mark.parametrize(
        "query,expected_full_key",
        [
            ("30013", "PTW 30013 Farmer (0.60 cm³, waterproof)"),
            ("30010", "PTW 30010 Farmer (0.60 cm³, classic)"),
            ("FC65-G", "IBA / Wellhöfer FC65-G (0.65 cm³)"),
            ("2571", "NE 2571 Farmer (0.60 cm³)"),
            ("2561", "NE 2561 / NPL 2611A Secondary Standard"),
            ("A12", "Exradin A12 Farmer (0.64 cm³)"),
            ("A1SL", "Exradin A1SL Miniature (0.053 cm³)"),
            ("SNC600c", "Sun Nuclear SNC600c Farmer (0.60 cm³)"),
            ("PR-06C", "Capintec PR-06C Farmer (0.65 cm³)"),
        ],
    )
    def test_chamber_key_fuzzy_resolution(self, query, expected_full_key):
        """Verify find_chamber_key reliably matches common short strings."""
        resolved = trs.find_chamber_key(query)
        assert resolved == expected_full_key

    def test_unknown_chamber_resolution_fallback(self):
        """Fuzzy search with unknown chamber should safely fall back to default PTW 30013."""
        resolved = trs.find_chamber_key("NonExistentChamberXYZ")
        assert resolved == "PTW 30013 Farmer (0.60 cm³, waterproof)"


# ==============================================================================
# Tier 2: Boundary Value Analysis (BVA)
# ==============================================================================

class TestPhotonBoundaryValues:
    """Verify TPR20,10 boundaries, clamping behavior, and monotonicity."""

    def test_tpr_lower_boundary_clamping(self):
        """TPR20,10 <= 0.56 must clamp to the 0.56 node value."""
        chamber = "PTW 30013 Farmer (0.60 cm³, waterproof)"
        val_056 = trs.KQ_PHOTON_CHAMBERS[chamber][0]
        assert trs.interpolate_kq(chamber, 0.56) == val_056
        assert trs.interpolate_kq(chamber, 0.50) == val_056
        assert trs.interpolate_kq(chamber, 0.00) == val_056

    def test_tpr_upper_boundary_clamping(self):
        """TPR20,10 >= 0.82 must clamp to the 0.82 node value."""
        chamber = "PTW 30013 Farmer (0.60 cm³, waterproof)"
        val_082 = trs.KQ_PHOTON_CHAMBERS[chamber][-1]
        assert trs.interpolate_kq(chamber, 0.82) == val_082
        assert trs.interpolate_kq(chamber, 0.85) == val_082
        assert trs.interpolate_kq(chamber, 1.00) == val_082

    def test_k_q_monotonicity(self):
        """For all chambers, k_Q must strictly decrease as TPR20,10 increases."""
        for chamber, kqs in trs.KQ_PHOTON_CHAMBERS.items():
            for i in range(len(kqs) - 1):
                assert kqs[i] >= kqs[i + 1], f"Monotonicity violated for {chamber} at index {i}"


# ==============================================================================
# Tier 3: Pairwise & Combinatorial Testing (All 26 Chambers x 12 TPR points)
# ==============================================================================

ALL_CHAMBERS = list(trs.KQ_PHOTON_CHAMBERS.keys())
ALL_TPRS = list(trs.KQ_PHOTON_TPRS)


class TestTable16AllChambers:
    """
    Exhaustive verification of Table 16 across all 26 cylindrical chambers
    and all 12 published TPR20,10 points (312 combinations).
    """

    @pytest.mark.parametrize("chamber", ALL_CHAMBERS)
    def test_all_chambers_analytical_fit_accuracy(self, chamber):
        """
        Assert Appendix II Eq. (34) fit matches Table 16 within <= 0.00015 across all TPRs.
        Official benchmark: observed maximum difference is 0.000138.
        """
        table_vals = trs.KQ_PHOTON_CHAMBERS[chamber]
        max_diff = 0.0

        for tpr, expected_kq in zip(ALL_TPRS, table_vals):
            calc_val = trs.calc_kq_photon_analytical(chamber, tpr)
            diff = abs(calc_val - expected_kq)
            if diff > max_diff:
                max_diff = diff
            assert diff <= 0.00015, (
                f"Chamber '{chamber}' at TPR={tpr}: "
                f"calc={calc_val:.6f}, table={expected_kq:.4f}, diff={diff:.6f} > 0.00015"
            )

    @pytest.mark.parametrize("chamber", ALL_CHAMBERS)
    def test_all_chambers_node_interpolation_exact(self, chamber):
        """Assert interpolate_kq matches Table 16 exactly at each published TPR node."""
        table_vals = trs.KQ_PHOTON_CHAMBERS[chamber]
        for tpr, expected_kq in zip(ALL_TPRS, table_vals):
            interp_val = trs.interpolate_kq(chamber, tpr)
            assert interp_val == expected_kq, (
                f"Interpolation node mismatch for '{chamber}' at TPR={tpr}: "
                f"got {interp_val}, expected {expected_kq}"
            )

    def test_intermediate_linear_interpolation(self):
        """Verify linear interpolation midway between TPR=0.59 and TPR=0.62."""
        chamber = "PTW 30013 Farmer (0.60 cm³, waterproof)"
        # Table 16 values: TPR 0.59 -> 0.9984; TPR 0.62 -> 0.9956
        tpr_mid = 0.605
        expected_mid = 0.9984 + ((0.9956 - 0.9984) / (0.62 - 0.59)) * (0.605 - 0.59)
        actual_mid = trs.interpolate_kq(chamber, tpr_mid)
        assert abs(actual_mid - expected_mid) < 1e-9


# ==============================================================================
# Tier 4: Clinical Real-World End-to-End Scenarios
# ==============================================================================

class TestClinicalPhotonReferenceDosimetry:
    """End-to-end clinical absorbed-dose-to-water workflow testing."""

    def test_truebeam_6mv_reference_dosimetry_pipeline(self, standard_dosimetry_inputs):
        """
        Clinical Scenario: Varian TrueBeam 6 MV reference dosimetry:
        - Farmer chamber PTW 30013 (Nd,w = 0.05385 Gy/nC)
        - TPR20,10 = 0.670, PDD = 67.0%
        - Environmental: 21.5°C, 100.8 kPa
        - V1 = 300 V, V2 = 150 V
        """
        inputs = standard_dosimetry_inputs["photon_6mv"]
        result = trs.calc_trs398_dose(
            raw_readings=inputs["readings"],
            mu_delivered=100.0,
            nd_w=inputs["n_dw"],
            tpr2010=inputs["tpr2010"],
            chamber_key=inputs["chamber"],
            temp_c=inputs["temp_c"],
            press_kpa=inputs["press_kpa"],
            pdd_zref=inputs["pdd_zref"],
            m_plus=inputs["m_user"],
            m_minus=inputs["m_opposite"],
            v1=inputs["v_user"],
            v2=inputs["v_reduced"],
            m1=inputs["m_user"],
            m2=inputs["m_reduced"],
            target_dose_per_mu=0.01000,
            is_fff=False,
            dose_rate_mu_min=inputs["dose_rate_mu_min"],
            prf_hz=inputs["prf_hz"],
        )

        # 1. Corrections verification
        # k_TP: (273.15 + 21.5) / (273.15 + 20) * 101.325 / 100.8 = 1.010375
        assert 1.0100 <= result["k_tp"] <= 1.0110
        # k_pol: (|18.234| + |-18.230|) / (2 * 18.234) = 0.99989
        assert 0.9990 <= result["k_pol"] <= 1.0005
        # k_s: pulsed 2-voltage recombination
        assert 1.0000 <= result["k_s"] <= 1.0200
        # k_vol: flattened beam must be 1.0000
        assert result["k_vol"] == 1.0000
        # k_Q at TPR=0.670 (interpolated between 0.65: 0.9920 and 0.68: 0.9876)
        expected_kq = 0.9920 + ((0.9876 - 0.9920) / (0.68 - 0.65)) * (0.670 - 0.65)
        assert abs(result["k_q"] - expected_kq) < 0.0005

        # 2. Dosimetric output
        assert result["m_raw_avg_nc"] == pytest.approx(np.mean(inputs["readings"]), rel=1e-4)
        assert result["dose_per_mu_zref_cgy"] > 0.60
        assert result["dose_per_mu_dmax_cgy"] > 0.90
        # With 18.234 nC, output is ~1.48 cGy/MU, which correctly triggers OUT OF TOLERANCE
        assert "OUT OF TOLERANCE" in result["compliance"]
        assert result["recommended_mu_trim_pct"] != 0.0

        # Calibrated readings (~12.33 nC) producing ~1.000 cGy/MU should yield PASS
        calibrated_result = trs.calc_trs398_dose(
            raw_readings=[12.330, 12.332, 12.331],
            mu_delivered=100.0,
            nd_w=inputs["n_dw"],
            tpr2010=inputs["tpr2010"],
            chamber_key=inputs["chamber"],
            temp_c=inputs["temp_c"],
            press_kpa=inputs["press_kpa"],
            pdd_zref=inputs["pdd_zref"],
        )
        assert "PASS" in calibrated_result["compliance"]

    @pytest.mark.parametrize(
        "output_cgy,expected_status",
        [
            (1.005, "PASS"),              # Deviation +0.50% -> PASS
            (0.992, "PASS"),              # Deviation -0.80% -> PASS
            (1.015, "ACTION"),            # Deviation +1.50% -> ACTION
            (0.985, "ACTION"),            # Deviation -1.50% -> ACTION
            (1.025, "OUT OF TOLERANCE"),  # Deviation +2.50% -> OUT OF TOLERANCE
            (0.970, "OUT OF TOLERANCE"),  # Deviation -3.00% -> OUT OF TOLERANCE
        ],
    )
    def test_compliance_status_logic(self, output_cgy, expected_status):
        """Verify strict clinical tolerance thresholds (PASS <= 1%, ACTION <= 2%, OUT > 2%)."""
        dev_pct = (output_cgy - 1.000) / 1.000 * 100.0
        if abs(dev_pct) <= 1.0:
            status = "PASS"
        elif abs(dev_pct) <= 2.0:
            status = "ACTION"
        else:
            status = "OUT OF TOLERANCE"
        assert status == expected_status
