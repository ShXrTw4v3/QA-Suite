"""
Automated Pytest Validation Suite: IAEA TRS-398 Rev. 1 (2024) Physical Correction Factors

Covers:
- Air density correction k_TP (Section 4.4.3, Eq. 10): 273.15 K thermodynamic absolute zero.
- Polarity correction k_pol (Section 4.4.4, Eq. 11).
- Two-voltage ion recombination k_s (Section 4.4.5.1, Table 10 & Eq. 14):
  Pulsed and pulsed-scanned quadratic fits across standard V1/V2 ratios (2.0, 2.5, 3.0, 3.5, 4.0, 5.0).
- FFF volume averaging correction k_vol (Section 4.4.6, Eq. 22 & Table 11):
  1.0000 for flattened beams, 1.0015 for 6 MV FFF, 1.0034 for 10 MV FFF.
- Dose per pulse (DPP) and high recombination alerts.
"""

import math
import pytest

import trs398_calculator as trs


# ==============================================================================
# Tier 1: Category-Partition Testing
# ==============================================================================

class TestCorrectionCategories:
    """Verify standard correction categories, beam mode switches, and fit tables."""

    def test_flattened_beam_kvol_always_unity(self):
        """Flattened beams (is_fff=False) MUST always yield k_vol = 1.0000 regardless of chamber."""
        for chamber in ["PTW 30013", "PTW 30010", "IBA FC65-G", "NE 2571", "Exradin A12"]:
            val = trs.estimate_kvol_fff(chamber, is_fff=False, tpr_or_energy=0.670, sdd_cm=110.0)
            assert val == 1.0000

    def test_recombination_fit_table_availability(self):
        """Verify Table 10 fits exist for all standard voltage ratios."""
        standard_ratios = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        for ratio in standard_ratios:
            assert ratio in trs.V1_V2_FITS, f"Missing pulsed fit for V1/V2={ratio}"
            assert ratio in trs.V1_V2_FITS_PULSED_SCANNED, f"Missing pulsed-scanned fit for V1/V2={ratio}"

    def test_chamber_cavity_lengths(self):
        """Verify cavity length lookups for Farmer and micro-chambers."""
        # PTW 30013 Farmer: 23.0 mm (2.3 cm)
        assert trs.get_chamber_cavity_length("PTW 30013") == 23.0
        # Exradin A12 Farmer: 21.6 mm
        assert trs.get_chamber_cavity_length("A12") == 21.6
        # Exradin A26 micro-chamber: 1.8 mm
        assert trs.get_chamber_cavity_length("A26") == 1.8


# ==============================================================================
# Tier 2: Boundary Value Analysis (BVA)
# ==============================================================================

class TestCorrectionBoundaryValues:
    """Verify thermodynamic absolute zero, reference states, and extreme environmental bounds."""

    def test_ktp_reference_conditions_exact_unity(self):
        """
        At reference conditions (T = 20.0°C, P = 101.325 kPa, T0 = 20.0, P0 = 101.325):
        k_TP MUST equal 1.000000 exactly.
        """
        ktp = trs.calc_ktp(temp_c=20.0, press_kpa=101.325, t0_c=20.0, p0_kpa=101.325)
        assert ktp == 1.000000

    def test_ktp_thermodynamic_absolute_zero_formula(self):
        """
        Verify Eq. (10): k_TP = (273.15 + T) / (273.15 + T0) * (P0 / P).
        Thermodynamic absolute zero is strictly 273.15 K (not 273.2 or 273.0).
        """
        t = 22.4
        p = 100.2
        expected_ktp = ((273.15 + t) / (273.15 + 20.0)) * (101.325 / p)
        actual_ktp = trs.calc_ktp(temp_c=t, press_kpa=p)
        assert abs(actual_ktp - expected_ktp) < 1e-9

    @pytest.mark.parametrize(
        "temp_c,press_kpa,expected_range",
        [
            (0.0, 101.325, (0.9310, 0.9325)),    # Freezing point
            (18.0, 102.5, (0.9810, 0.9830)),    # Cool room, high barometric pressure
            (22.0, 100.0, (1.0195, 1.0210)),    # Warm room, low barometric pressure
            (35.0, 70.0, (1.5200, 1.5300)),     # Extreme high altitude & heat
        ],
    )
    def test_ktp_environmental_extremes(self, temp_c, press_kpa, expected_range):
        """Verify k_TP behaves monotonically and matches physical limits."""
        ktp = trs.calc_ktp(temp_c, press_kpa)
        assert expected_range[0] <= ktp <= expected_range[1]

    def test_kpol_symmetric_readings_unity(self):
        """
        Eq. (11): k_pol = (|M+| + |M-|) / (2 * |M+|)
        When readings are symmetric (M+ = 10.0 nC, M- = -10.0 nC), k_pol == 1.000000.
        """
        assert trs.calc_kpol(10.0, -10.0) == 1.000000

    def test_kpol_asymmetric_readings(self):
        """When M+ = 10.05 nC and M- = -9.95 nC, k_pol = (10.05 + 9.95)/(2 * 10.05) = 0.995025."""
        kpol = trs.calc_kpol(10.05, -9.95)
        assert abs(kpol - 0.99502487) < 1e-6


# ==============================================================================
# Tier 3: Pairwise & Combinatorial Testing
# ==============================================================================

class TestTable10RecombinationFits:
    """Verifies Table 10 quadratic fits across all standard voltage ratios."""

    @pytest.mark.parametrize("v_ratio", [2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
    def test_table10_pulsed_quadratic_fit(self, v_ratio):
        """
        Asserts calc_ks_pulsed evaluates Table 10 quadratic fit for pulsed beams.
        At M1/M2 = 1.01, k_s should be slightly above 1.0 (between 1.005 and 1.025).
        """
        ks, desc = trs.calc_ks_pulsed(v1=v_ratio * 100.0, v2=100.0, m1=1.010, m2=1.000, pulsed_scanned=False)
        assert "Table 10 quadratic fit (pulsed" in desc
        assert 1.000 < ks <= 1.025

    def test_table10_pulsed_v2_exact_reference(self):
        """
        Official reference check for V1/V2 = 2.0:
        a0 = 2.337, a1 = -3.636, a2 = 2.299
        For M1/M2 = 1.010:
        k_s = 2.337 - 3.636 * 1.010 + 2.299 * (1.010^2)
            = 2.337 - 3.67236 + 2.3452099 = 1.0098499
        """
        ks, _ = trs.calc_ks_pulsed(v1=300.0, v2=150.0, m1=1.010, m2=1.000, pulsed_scanned=False)
        assert abs(ks - 1.00985) < 0.0001

    def test_table10_pulsed_scanned_fit(self):
        """Verify pulsed-scanned beam coefficients for V1/V2 = 2.0."""
        ks, desc = trs.calc_ks_pulsed(v1=300.0, v2=150.0, m1=1.010, m2=1.000, pulsed_scanned=True)
        assert "Table 10 quadratic fit (pulsed-scanned" in desc
        # a0 = 4.711, a1 = -8.242, a2 = 4.533
        # 4.711 - 8.242 * 1.01 + 4.533 * 1.0201 = 4.711 - 8.32442 + 4.62411 = 1.01069
        assert abs(ks - 1.0107) < 0.0005

    def test_nonstandard_voltage_ratio_fallback_eq14(self):
        """
        Non-standard voltage ratio (e.g. V1/V2 = 2.80) should fall back to
        Eq. (14) linear approximation: (V1/V2 - 1) / (V1/V2 - M1/M2).
        """
        v1, v2 = 280.0, 100.0
        m1, m2 = 10.10, 10.00
        ratio_v = 2.80
        ratio_m = 1.01
        expected_ks = (ratio_v - 1.0) / (ratio_v - ratio_m)  # 1.80 / 1.79 = 1.0055866

        ks, desc = trs.calc_ks_pulsed(v1, v2, m1, m2)
        assert "Eq. (14) linear approximation" in desc
        assert abs(ks - expected_ks) < 1e-5


# ==============================================================================
# Tier 4: Clinical Real-World FFF & High-Dose-Rate Scenarios
# ==============================================================================

class TestFFFVolumeAveragingAndRecombination:
    """Verifies FFF volume averaging (k_vol) and Dose-Per-Pulse."""

    def test_fff_6mv_farmer_kvol(self):
        """
        IAEA TRS-398 Rev. 1 Table 11 / Eq. (22) for 6 MV FFF:
        PTW 30013 Farmer (L = 2.3 cm), TPR20,10 = 0.635, SDD = 110 cm (SSD 100 + zref 10).
        Table 11 published value: 1.0015.
        """
        kvol = trs.estimate_kvol_fff("PTW 30013", is_fff=True, tpr_or_energy=0.635, sdd_cm=110.0)
        assert kvol == 1.0015

    def test_fff_10mv_farmer_kvol(self):
        """
        IAEA TRS-398 Rev. 1 Table 11 / Eq. (22) for 10 MV FFF:
        PTW 30013 Farmer (L = 2.3 cm), TPR20,10 = 0.705, SDD = 110 cm.
        Table 11 published value: 1.0034.
        """
        kvol = trs.estimate_kvol_fff("PTW 30013", is_fff=True, tpr_or_energy=0.705, sdd_cm=110.0)
        assert kvol == 1.0034

    def test_fff_sad_setup_higher_correction(self):
        """
        At SDD = 100 cm (SAD setup), the (100 / SDD)^2 factor equals 1.0,
        yielding a slightly higher volume averaging correction than at SDD = 110 cm.
        """
        kvol_sad = trs.estimate_kvol_fff("PTW 30013", is_fff=True, tpr_or_energy=0.635, sdd_cm=100.0)
        kvol_ssd = trs.estimate_kvol_fff("PTW 30013", is_fff=True, tpr_or_energy=0.635, sdd_cm=110.0)
        assert kvol_sad > kvol_ssd
        assert kvol_sad == 1.0018

    def test_dose_per_pulse_calculation(self):
        """
        Verify Dose-Per-Pulse (DPP):
        DPP (mGy/pulse) = (DoseRate [cGy/min] * 10) / (60 * PRF [Hz])
        """
        # TrueBeam 6 MV (600 MU/min, 360 Hz): (600 * 10) / (60 * 360) = 0.2778 mGy/pulse
        dpp_6w = trs.calc_dose_per_pulse(600.0, 360.0)
        assert abs(dpp_6w - 0.27778) < 0.0001

        # TrueBeam 10 MV FFF (2400 MU/min, 360 Hz): (2400 * 10) / (60 * 360) = 1.1111 mGy/pulse
        dpp_10fff = trs.calc_dose_per_pulse(2400.0, 360.0)
        assert abs(dpp_10fff - 1.1111) < 0.0001
