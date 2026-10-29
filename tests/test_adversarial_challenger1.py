"""
test_adversarial_challenger1.py — Empirical Adversarial Stress Testing & Numerical Oracles
========================================================================================
Challenger 1: Adversarial verification of beam_profile_analyzer.py and trs398_calculator.py.

Stress Testing Dimensions:
1. Extreme Field Sizes: 1x1 cm, 2x2 cm small fields, 40x40 cm maximum fields, sub-cm fields.
2. Noise and SNR Degradation: Gaussian/Poisson noise, derivative stability, inflection point robustness.
3. Asymmetric Profiles: Steep wedge tilts, shifted CAX, asymmetric jaw collimation.
4. Degenerate & Malformed Text Files: Corrupted headers, NaNs/Infs, negative doses, duplicate coordinates, non-uniform spacing.
5. Boundary Tests on TRS-398 Rev. 1: Out-of-table TPR20,10, extreme T (-10 to 50 C), extreme P (70 to 115 kPa),
   recombination singularities, electron cylindrical chamber cutoff (R50 < 3.0 g/cm^2).
"""

import math
import numpy as np
import pytest
import beam_profile_analyzer as bpa
import trs398_calculator as trs


# ==============================================================================
# SECTION 1: EXTREME FIELD SIZES (Small Fields & Large Fields)
# ==============================================================================

class TestExtremeFieldSizes:
    """Stress tests for field sizes ranging from 5 mm to 400 mm."""

    def test_small_field_1x1cm_flattened(self):
        """
        1x1 cm (10 mm) small field evaluation on flattened beam.
        Note: The synthetic flattened generator uses raw_eval division which displaces
        the 50% point by ~0.9 mm for small fields; we verify numerical stability without NaN/inf.
        """
        syn = bpa.generate_synthetic_profile(
            beam_type="Flattened",
            field_size_mm=10.0,
            penumbra_mm=2.5,
            step_size_mm=0.2,
            range_mm=30.0,
        )
        res = bpa.analyze_flattened_profile(syn["x_mm"], syn["values"], protocol="IEC")
        
        # Verify complete numerical sanity (no NaN or Inf)
        assert not np.isnan(res["fwhm_mm"])
        assert not np.isnan(res["penumbra_left_mm"])
        assert not np.isnan(res["penumbra_right_mm"])
        assert not np.isnan(res["flatness_iec_pct"])
        assert not np.isnan(res["symmetry_iec_pdq"])
        assert abs(res["fwhm_mm"] - 10.0) < 1.0

    def test_small_field_1x1cm_fff(self):
        """1x1 cm (10 mm) small field evaluation on FFF beam."""
        syn = bpa.generate_synthetic_profile(
            beam_type="6 MV FFF",
            field_size_mm=10.0,
            penumbra_mm=2.5,
            step_size_mm=0.2,
            range_mm=30.0,
        )
        res = bpa.analyze_fff_profile(syn["x_mm"], syn["values"], smoothing_radius_mm=0.3)
        
        assert not np.isnan(res["inflection_width_mm"])
        assert not np.isnan(res["penumbra_left_mm"])
        assert not np.isnan(res["penumbra_right_mm"])
        assert not np.isnan(res["symmetry_pct"])
        assert abs(res["inflection_width_mm"] - 10.0) < 0.6

    def test_small_field_2x2cm_flattened_and_fff(self):
        """2x2 cm (20 mm) field evaluation."""
        syn_flat = bpa.generate_synthetic_profile(
            beam_type="Flattened", field_size_mm=20.0, penumbra_mm=3.5, step_size_mm=0.2, range_mm=40.0
        )
        res_flat = bpa.analyze_flattened_profile(syn_flat["x_mm"], syn_flat["values"])
        assert abs(res_flat["fwhm_mm"] - 20.0) < 1.0

        syn_fff = bpa.generate_synthetic_profile(
            beam_type="6 MV FFF", field_size_mm=20.0, penumbra_mm=3.5, step_size_mm=0.2, range_mm=40.0
        )
        res_fff = bpa.analyze_fff_profile(syn_fff["x_mm"], syn_fff["values"])
        assert abs(res_fff["inflection_width_mm"] - 20.0) < 0.6

    def test_large_field_40x40cm_flattened(self):
        """40x40 cm (400 mm) maximum field size evaluation."""
        syn = bpa.generate_synthetic_profile(
            beam_type="Flattened",
            field_size_mm=400.0,
            penumbra_mm=7.0,
            step_size_mm=1.0,
            range_mm=250.0,
        )
        res = bpa.analyze_flattened_profile(syn["x_mm"], syn["values"], protocol="Varian")
        assert not np.isnan(res["fwhm_mm"])
        assert abs(res["fwhm_mm"] - 400.0) < 1.0
        assert res["eval_region"]["left"] == pytest.approx(-160.0, abs=1.0)
        assert res["eval_region"]["right"] == pytest.approx(160.0, abs=1.0)

    def test_large_field_40x40cm_fff(self):
        """40x40 cm (400 mm) maximum field size FFF evaluation."""
        syn = bpa.generate_synthetic_profile(
            beam_type="10 MV FFF",
            field_size_mm=400.0,
            penumbra_mm=6.5,
            step_size_mm=1.0,
            range_mm=250.0,
        )
        res = bpa.analyze_fff_profile(syn["x_mm"], syn["values"], smoothing_radius_mm=1.0)
        assert not np.isnan(res["inflection_width_mm"])
        assert abs(res["inflection_width_mm"] - 400.0) < 2.0

    def test_sub_centimeter_field_5mm(self):
        """5 mm micro-field (stereotactic collimator / MLC slit)."""
        syn = bpa.generate_synthetic_profile(
            beam_type="Flattened",
            field_size_mm=5.0,
            penumbra_mm=1.5,
            step_size_mm=0.1,
            range_mm=15.0,
        )
        res = bpa.analyze_flattened_profile(syn["x_mm"], syn["values"])
        assert not np.isnan(res["fwhm_mm"])
        assert abs(res["fwhm_mm"] - 5.0) < 1.0


# ==============================================================================
# SECTION 2: NOISE AND SNR DEGRADATION
# ==============================================================================

class TestNoiseAndSNR:
    """Stress tests profile analysis under severe noise levels."""

    @pytest.mark.parametrize("noise_sigma", [0.5, 1.0, 2.0, 5.0])
    def test_flattened_gaussian_noise_robustness(self, noise_sigma):
        """Flattened profile with additive Gaussian noise."""
        np.random.seed(42)
        syn = bpa.generate_synthetic_profile(field_size_mm=100.0, penumbra_mm=6.0, step_size_mm=0.5)
        x = syn["x_mm"]
        y_noisy = syn["values"] + np.random.normal(0, noise_sigma, size=len(x))
        
        res = bpa.analyze_flattened_profile(x, y_noisy)
        # Even with 5% noise, FWHM should remain within 1.5 mm of 100 mm
        assert abs(res["fwhm_mm"] - 100.0) < 1.5
        assert not np.isnan(res["flatness_iec_pct"])
        assert not np.isnan(res["symmetry_iec_pdq"])

    @pytest.mark.parametrize("noise_sigma", [0.5, 1.0, 2.0])
    def test_fff_inflection_point_gaussian_noise(self, noise_sigma):
        """FFF profile inflection point stability under noise with Gaussian smoothing."""
        np.random.seed(42)
        syn = bpa.generate_synthetic_profile(beam_type="6 MV FFF", field_size_mm=100.0, penumbra_mm=6.0, step_size_mm=0.5)
        x = syn["x_mm"]
        y_noisy = syn["values"] + np.random.normal(0, noise_sigma, size=len(x))
        
        # Using 1.0 mm smoothing radius to filter noise
        res = bpa.analyze_fff_profile(x, y_noisy, smoothing_radius_mm=1.0)
        assert not np.isnan(res["inflection_width_mm"])
        # Inflection width should be stable within 1.5 mm
        assert abs(res["inflection_width_mm"] - 100.0) < 1.5

    def test_single_spike_outlier_rejection(self):
        """Single bad pixel / detector spike inside the field."""
        syn = bpa.generate_synthetic_profile(field_size_mm=100.0, step_size_mm=0.5)
        x = syn["x_mm"]
        y = np.copy(syn["values"])
        # Inject extreme spike at x = 20 mm
        spike_idx = np.argmin(np.abs(x - 20.0))
        y[spike_idx] += 50.0  # +50% dose spike
        
        # In flattened profile, d_max will reflect the spike
        res = bpa.analyze_flattened_profile(x, y)
        assert not np.isnan(res["fwhm_mm"])
        # Field width crossing 50% is unaffected by spike in plateau
        assert abs(res["fwhm_mm"] - 100.0) < 0.5


# ==============================================================================
# SECTION 3: ASYMMETRIC PROFILES & TILTS
# ==============================================================================

class TestAsymmetricProfiles:
    """Stress tests asymmetric wedges and shifted CAX."""

    @pytest.mark.parametrize("tilt", [0.05, 0.15, 0.30, 0.50])
    def test_steep_wedge_tilt(self, tilt):
        """Profiles with varying degrees of wedge tilt."""
        syn = bpa.generate_synthetic_profile(
            beam_type="Flattened",
            field_size_mm=100.0,
            cax_tilt=tilt,
            step_size_mm=0.5,
        )
        res_iec = bpa.analyze_flattened_profile(syn["x_mm"], syn["values"], protocol="IEC")
        res_var = bpa.analyze_flattened_profile(syn["x_mm"], syn["values"], protocol="Varian")
        
        # Symmetry metrics should monotonically increase with tilt
        assert res_iec["symmetry_iec_pdq"] > 1.0
        assert res_var["symmetry_varian_pct"] > 0.0
        if tilt >= 0.15:
            assert res_iec["status"] in ("Tolerance", "Action")

    def test_shifted_cax_flattened(self):
        """Beam profile physically shifted by +10 mm relative to machine coordinate center."""
        syn = bpa.generate_synthetic_profile(field_size_mm=100.0, step_size_mm=0.5)
        # Shift coordinates: beam center now at x = 10 mm
        x_shifted = syn["x_mm"] + 10.0
        
        # If cax_pos is supplied as 10.0 mm
        res = bpa.analyze_flattened_profile(x_shifted, syn["values"], cax_pos=10.0)
        assert abs(res["fwhm_mm"] - 100.0) < 0.5
        assert abs(res["cax_offset_mm"] - 10.0) < 0.5

    def test_asymmetric_field_collimation(self):
        """Asymmetric collimation: Left edge at -30 mm, Right edge at +70 mm (width 100 mm)."""
        x = np.arange(-100.0, 100.5, 0.5)
        sigma = 6.0 / (4.0 * np.log(2.0))
        left_step = 1.0 / (1.0 + np.exp(-np.clip((x - (-30.0)) / sigma, -100, 100)))
        right_step = 1.0 / (1.0 + np.exp(np.clip((x - 70.0) / sigma, -100, 100)))
        y = left_step * right_step * 100.0
        
        # Center of field is at (-30 + 70)/2 = +20 mm
        res = bpa.analyze_flattened_profile(x, y, cax_pos=20.0)
        assert abs(res["fwhm_mm"] - 100.0) < 0.5
        assert abs(res["cax_offset_mm"] - 20.0) < 0.5


# ==============================================================================
# SECTION 4: DEGENERATE & MALFORMED TEXT FILES
# ==============================================================================

class TestDegenerateTextFiles:
    """Stress tests parser handling of malformed and corrupted ASCII files."""

    def test_empty_content_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="No numeric profile data"):
            bpa.parse_profile_text("")

    def test_comments_only_raises(self):
        """File with only comments should raise ValueError."""
        content = "# IBA OmniPro Export\n# Date: 2026-10-28\n% Machine: TrueBeam\n"
        with pytest.raises(ValueError, match="No numeric profile data"):
            bpa.parse_profile_text(content)

    def test_corrupted_tokens_and_nans(self):
        """File containing corrupt lines, text strings, and valid numeric rows."""
        content = """# Corrupted Header
some invalid text line
-50.0   0.5
-40.0   2.0
corrupt_token_here
-30.0   15.0
-20.0   80.0
-10.0   98.0
0.0     100.0
10.0    98.5
20.0    81.0
30.0    16.0
40.0    2.5
50.0    0.6
"""
        parsed = bpa.parse_profile_text(content)
        assert len(parsed["x_mm"]) == 11
        assert parsed["x_mm"][0] == -50.0
        assert parsed["x_mm"][-1] == 50.0

    def test_mixed_delimiters(self):
        """Mixed tabs, commas, and spaces in data rows."""
        content = """# Mixed delimiters
-40.0,\t80.0
-30.0, 98.0
0.0\t100.0
30.0;98.5
40.0   81.0
"""
        parsed = bpa.parse_profile_text(content)
        assert len(parsed["x_mm"]) == 5
        assert parsed["values"][2] == 100.0

    def test_unsorted_and_duplicate_coordinates(self):
        """Coordinates not in ascending order and containing duplicates (with coordinates > 25 mm)."""
        content = """# Unsorted with duplicates
30.0    98.5
0.0     100.0
-30.0   98.0
0.0     100.0
-40.0   80.0
40.0    81.0
"""
        parsed = bpa.parse_profile_text(content)
        # Should be strictly ascending and duplicates removed
        assert np.all(np.diff(parsed["x_mm"]) > 0)
        assert len(parsed["x_mm"]) == 5
        assert parsed["x_mm"][0] == -40.0
        assert parsed["x_mm"][-1] == 40.0

    def test_small_field_coordinate_scaling_vulnerability(self):
        """
        Adversarial proof: When a file with coordinates |x| <= 25.0 is parsed,
        the parser multiplies coordinates by 10 (assuming cm). We document this behavior.
        """
        content = """# Small Field Scan
-20.0   80.0
0.0     100.0
20.0    80.0
"""
        parsed = bpa.parse_profile_text(content)
        # Demonstrates the autoscale heuristic: 20 becomes 200 mm
        assert parsed["x_mm"][0] == -200.0
        assert parsed["x_mm"][-1] == 200.0

    def test_coordinate_cm_to_mm_autoscale(self):
        """When spatial coordinates are in cm (|x| <= 25 cm), autoscale to mm."""
        content = """# Coordinates in cm
-15.0   1.2
-10.0   50.0
0.0     100.0
10.0    50.0
15.0    1.1
"""
        parsed = bpa.parse_profile_text(content)
        # Should be scaled to mm (-150 to +150)
        assert parsed["x_mm"][0] == -150.0
        assert parsed["x_mm"][-1] == 150.0


# ==============================================================================
# SECTION 5: BOUNDARY TESTS ON TRS-398 REV. 1
# ==============================================================================

class TestTRS398BoundaryConditions:
    """Stress tests boundary and extreme physical conditions for TRS-398 Rev. 1."""

    def test_tpr2010_below_table_minimum(self):
        """TPR20,10 = 0.50 (below Table 16 min of 0.56). Should clamp gracefully."""
        kq_interp = trs.interpolate_kq("PTW 30013", 0.50)
        kq_min = trs.KQ_PHOTON_CHAMBERS["PTW 30013 Farmer (0.60 cm³, waterproof)"][0]
        assert kq_interp == kq_min

        # Analytical formula remains valid and smooth
        kq_ana = trs.calc_kq_photon_analytical("PTW 30013", 0.50)
        assert not np.isnan(kq_ana)
        assert kq_ana > 0.99

    def test_tpr2010_above_table_maximum(self):
        """TPR20,10 = 0.90 (above Table 16 max of 0.82). Should clamp gracefully."""
        kq_interp = trs.interpolate_kq("PTW 30013", 0.90)
        kq_max = trs.KQ_PHOTON_CHAMBERS["PTW 30013 Farmer (0.60 cm³, waterproof)"][-1]
        assert kq_interp == kq_max

        kq_ana = trs.calc_kq_photon_analytical("PTW 30013", 0.90)
        assert not np.isnan(kq_ana)
        assert kq_ana < 0.94

    def test_estimate_kvol_fff_boundary(self):
        """Test FFF volume averaging k_vol across realistic and edge TPRs."""
        # For non-FFF, always exactly 1.0000
        assert trs.estimate_kvol_fff("PTW 30013", is_fff=False, tpr_or_energy=0.635) == 1.0000

        # For FFF at standard 6 MV FFF (TPR = 0.635)
        kvol_6fff = trs.estimate_kvol_fff("PTW 30013", is_fff=True, tpr_or_energy=0.635, sdd_cm=110.0)
        assert kvol_6fff == 1.0015

        # For FFF at standard 10 MV FFF (TPR = 0.705)
        kvol_10fff = trs.estimate_kvol_fff("PTW 30013", is_fff=True, tpr_or_energy=0.705, sdd_cm=110.0)
        assert kvol_10fff == 1.0034

    def test_estimate_kvol_fff_above_088_collapse_vulnerability(self):
        """
        Verify that for high-energy FFF beams with TPR20,10 = 0.90 (> 0.88),
        estimate_kvol_fff correctly evaluates k_vol = 1.0087 (> kvol_088 = 1.0081).
        """
        kvol_088 = trs.estimate_kvol_fff("PTW 30013", is_fff=True, tpr_or_energy=0.88, sdd_cm=110.0)
        kvol_090 = trs.estimate_kvol_fff("PTW 30013", is_fff=True, tpr_or_energy=0.90, sdd_cm=110.0)
        # Expected from physics: kvol_090 > kvol_088
        assert kvol_088 == 1.0081
        assert kvol_090 == 1.0087  # Remediated: correctly evaluates to 1.0087

    @pytest.mark.parametrize("temp_c, press_kpa", [
        (-10.0, 101.325),   # Cold chamber room / unheated storage
        (0.0, 101.325),     # Freezing point
        (20.0, 101.325),    # Reference conditions (kTP = 1.0000)
        (35.0, 101.325),    # Hot summer linac room
        (50.0, 101.325),    # Extreme heat
        (20.0, 70.0),       # High altitude (e.g. Mexico City / La Paz ~70 kPa)
        (20.0, 85.0),       # Moderate altitude (~1500m)
        (20.0, 115.0),      # High atmospheric pressure / hyperbaric facility
    ])
    def test_ktp_extreme_temperatures_and_pressures(self, temp_c, press_kpa):
        """Verify k_TP air density correction formula under extreme temperatures and pressures."""
        ktp = trs.calc_ktp(temp_c, press_kpa)
        assert not np.isnan(ktp)
        assert not np.isinf(ktp)
        assert ktp > 0.0

        # Reference condition exact check
        if temp_c == 20.0 and press_kpa == 101.325:
            assert pytest.approx(ktp, rel=1e-5) == 1.0000

        # Lower pressure means thinner air -> higher k_TP (more correction needed)
        if press_kpa == 70.0:
            assert ktp > 1.40

    def test_electron_cylindrical_chamber_rejection(self):
        """IAEA TRS-398 Rev. 1 Table 19 strictly forbids cylindrical chambers for R50 < 3.0 g/cm^2."""
        cyl_chambers = [
            "PTW 30013 Farmer (Cylindrical)",
            "PTW 30012 Farmer (Cylindrical)",
            "IBA FC65-G Farmer (Cylindrical)",
            "NE 2571 Farmer (Cylindrical)",
            "Exradin A12 Farmer (Cylindrical)",
            "Exradin A19 Classic Farmer (Cylindrical)",
        ]
        for ch in cyl_chambers:
            # Interpolation must raise ValueError
            with pytest.raises(ValueError, match="requires R_50 >= 3.0"):
                trs.interpolate_kq_electron(ch, r50_cm=2.5)

            # Analytical formula must raise ValueError
            with pytest.raises(ValueError, match="only recommended for R_50 >= 3.0"):
                trs.calc_kq_electron_analytical(ch, r50_cm=2.5)

    def test_electron_plane_parallel_full_range(self):
        """Plane-parallel chambers (PTW Roos, IBA NACP) must remain valid across R50 1.0 to 10.0."""
        for r50 in [1.0, 1.4, 2.5, 3.0, 5.0, 10.0]:
            kq_roos = trs.interpolate_kq_electron("PTW 34001 Roos", r50)
            assert not np.isnan(kq_roos)
            assert 0.88 <= kq_roos <= 0.98

            kq_nacp = trs.interpolate_kq_electron("IBA NACP-02", r50)
            assert not np.isnan(kq_nacp)
            assert 0.88 <= kq_nacp <= 0.98

    def test_recombination_ks_boundary_conditions(self):
        """Test two-voltage recombination k_s near unity and with non-standard voltage ratios."""
        # Standard V1/V2 = 300/100 = 3.0, M1/M2 = 1.010 (1% recombination)
        ks, method = trs.calc_ks_pulsed(300.0, 100.0, 10.10, 10.00)
        assert 1.00 < ks < 1.03
        assert "Table 10" in method

        # Non-standard voltage ratio: V1/V2 = 400/150 = 2.67 -> falls back to linear approximation Eq. 14
        ks_lin, method_lin = trs.calc_ks_pulsed(400.0, 150.0, 10.10, 10.00)
        assert not np.isnan(ks_lin)
        assert 1.00 < ks_lin < 1.03

        # Zero voltage or reading -> returns 1.0 safely without crash
        ks_zero, msg_zero = trs.calc_ks_pulsed(300.0, 0.0, 10.0, 10.0)
        assert ks_zero == 1.0
        assert "Invalid" in msg_zero


# ==============================================================================
# SECTION 6: COMPREHENSIVE END-TO-END PIPELINE UNDER STRESS
# ==============================================================================

class TestFullPipelinesUnderStress:
    """End-to-end full pipeline execution with adversarial conditions."""

    def test_trs398_photon_full_pipeline_edge_inputs(self):
        """Run calc_trs398_dose with extreme conditions."""
        res = trs.calc_trs398_dose(
            raw_readings=[10.5, 10.51, 10.49],
            temp_c=32.5,
            press_kpa=95.4,
            tpr2010=0.68,
            chamber_key="PTW 30013",
            is_fff=True,
            dose_rate_mu_min=1400.0,
            v1=300.0,
            v2=100.0,
            m1=10.55,
            m2=10.42,
        )
        assert res["k_tp"] > 1.0
        assert res["k_vol"] > 1.0
        assert res["k_s"] > 1.0
        assert res["compliance"] is not None
        assert not np.isnan(res["d_w_zref_gy"])

    def test_trs398_electron_full_pipeline_edge_inputs(self):
        """Run calc_trs398_electron_dose with low energy electron beam."""
        res = trs.calc_trs398_electron_dose(
            raw_readings=[8.2, 8.21, 8.19],
            r50_cm=1.8,  # Low energy (approx 4-6 MeV electron)
            chamber_key="PTW 34001 Roos (Plane-Parallel)",
            temp_c=24.0,
            press_kpa=100.2,
        )
        assert res["chamber_type"] == "Plane-Parallel"
        assert res["z_ref_cm"] == pytest.approx(0.6 * 1.8 - 0.1, abs=1e-4)
        assert not np.isnan(res["d_w_zref_gy"])
