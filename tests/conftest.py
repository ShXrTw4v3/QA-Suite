"""
Shared Pytest Fixtures for Radiotherapy QA Suite Validation Suite.

Provides:
- Isolated temporary SQLite databases and schema initialisation.
- Standard mock LINAC, electrometer, and ion chamber configurations.
- Analytical synthetic beam profile mathematical fixtures.
- Realistic profile export text strings (CSV, TSV, IBA ASCII, PTW MCC).
- Clinical calculation results and metadata fixtures for PDF verification.
"""

import sys
import os
import tempfile
import sqlite3
import numpy as np
import pytest

# Ensure projects/qa-suite root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import db_setup
import crud


# ==============================================================================
# Database Fixtures
# ==============================================================================

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """
    Creates an isolated temporary SQLite database for a single test,
    redirects db_setup.DB_PATH and crud.DB_PATH, and initialises the schema.
    """
    db_file = tmp_path / "test_qa_data.db"
    db_path_str = str(db_file)
    monkeypatch.setattr(db_setup, "DB_PATH", db_path_str)
    monkeypatch.setattr(crud, "DB_PATH", db_path_str)

    # Initialise tables
    db_setup.init_db()

    yield db_path_str

    # Clean up connection if any remain
    if os.path.exists(db_path_str):
        try:
            os.remove(db_path_str)
        except OSError:
            pass


@pytest.fixture
def seeded_db(temp_db):
    """
    Creates a temporary database pre-seeded with baseline units and standard QA tests.
    """
    db_setup.seed_db()
    return temp_db


# ==============================================================================
# Mock LINAC & Dosimetry Configurations
# ==============================================================================

@pytest.fixture
def mock_linac_configs():
    """
    Standard clinical treatment unit specifications.
    """
    return {
        "truebeam_1": {
            "name": "Varian TrueBeam #1",
            "modality": "LINAC",
            "manufacturer": "Varian Medical Systems",
            "model": "TrueBeam 2.7",
            "serial_no": "SN-TB-001",
            "institution": "Clinical Medical Physics Center",
            "location": "Vault 1",
            "energies_photon": ["6 MV", "10 MV", "6 MV FFF", "10 MV FFF"],
            "energies_electron": ["6 MeV", "9 MeV", "12 MeV", "15 MeV"],
        },
        "versa_hd": {
            "name": "Elekta Versa HD #1",
            "modality": "LINAC",
            "manufacturer": "Elekta AB",
            "model": "Versa HD",
            "serial_no": "SN-VHD-104",
            "institution": "University Radiotherapy Institute",
            "location": "Vault 2",
            "energies_photon": ["6 MV", "10 MV", "15 MV", "6 MV FFF"],
            "energies_electron": ["6 MeV", "10 MeV", "15 MeV"],
        },
    }


@pytest.fixture
def standard_dosimetry_inputs():
    """
    Standard measurement inputs for TRS-398 photon and electron reference dosimetry.
    """
    return {
        "photon_6mv": {
            "unit_name": "Varian TrueBeam #1",
            "institution": "Radiotherapy Department",
            "physicist": "Dr. N. Golubovac",
            "beam_type": "photon",
            "energy": "6 MV",
            "is_fff": False,
            "chamber": "PTW 30013 Farmer (0.60 cm³, waterproof)",
            "n_dw": 0.05385,  # Gy/nC
            "tpr2010": 0.670,
            "pdd_zref": 67.0,  # % at 10 cm depth
            "readings": [18.230, 18.235, 18.237],  # nC
            "temp_c": 21.5,
            "press_kpa": 100.8,
            "v_user": 300.0,
            "v_reduced": 150.0,
            "m_user": 18.234,
            "m_reduced": 18.150,
            "m_opposite": -18.230,
            "dose_rate_mu_min": 600.0,
            "prf_hz": 360.0,
        },
        "photon_6fff": {
            "unit_name": "Varian TrueBeam #1",
            "institution": "Radiotherapy Department",
            "physicist": "Dr. N. Golubovac",
            "beam_type": "photon",
            "energy": "6 MV FFF",
            "is_fff": True,
            "chamber": "PTW 30013 Farmer (0.60 cm³, waterproof)",
            "n_dw": 0.05385,
            "tpr2010": 0.635,
            "pdd_zref": 63.5,
            "readings": [18.510, 18.512, 18.515],
            "temp_c": 20.0,
            "press_kpa": 101.325,
            "v_user": 300.0,
            "v_reduced": 150.0,
            "m_user": 18.512,
            "m_reduced": 18.320,
            "m_opposite": -18.510,
            "dose_rate_mu_min": 1400.0,
            "prf_hz": 360.0,
            "sdd_cm": 110.0,
        },
        "electron_9mev": {
            "unit_name": "Varian TrueBeam #1",
            "institution": "Radiotherapy Department",
            "physicist": "Dr. N. Golubovac",
            "beam_type": "electron",
            "energy": "9 MeV",
            "chamber": "PTW 34001 Roos (0.35 cm³)",
            "n_dw": 0.08250,
            "i50_cm": 3.45,
            "r50_cm": 3.49,
            "readings": [12.140, 12.145, 12.142],
            "temp_c": 21.0,
            "press_kpa": 101.0,
            "v_user": 200.0,
            "v_reduced": 100.0,
            "m_user": 12.142,
            "m_reduced": 12.080,
            "m_opposite": -12.140,
        },
    }


# ==============================================================================
# Analytical Synthetic Beam Profile Models
# ==============================================================================

def make_analytical_flattened_profile(
    field_width_mm: float = 100.0,
    penumbra_target_mm: float = 6.0,
    horn_factor: float = 0.0,
    tilt_factor: float = 0.0,
    cax_offset_mm: float = 0.0,
    x_range_mm: float = 120.0,
    dx_mm: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generates an exact mathematical flattened photon profile using the
    logistic sigmoid model:
        S(x; x0, sigma) = 1 / (1 + exp((x - x0)/sigma))
    where sigma = P80-20 / (4 * ln(2)).
    """
    sigma = penumbra_target_mm / (4.0 * np.log(2.0))
    x = np.arange(-x_range_mm, x_range_mm + dx_mm / 2.0, dx_mm)
    x_shifted = x - cax_offset_mm

    left_edge = -field_width_mm / 2.0
    right_edge = field_width_mm / 2.0

    step_left = 1.0 / (1.0 + np.exp(-(x_shifted - left_edge) / sigma))
    step_right = 1.0 / (1.0 + np.exp((x_shifted - right_edge) / sigma))
    raw_plateau = step_left * step_right
    x_eval_bound = 0.40 * field_width_mm
    plateau = np.where(np.abs(x_shifted) <= x_eval_bound, 1.0, raw_plateau)

    # Horn factor: h * (x / (0.4 * W))^2
    horn = 1.0 + horn_factor * ((x_shifted / (0.4 * field_width_mm)) ** 2)
    # Tilt factor: (1 + alpha * x / (0.5 * W))
    tilt = 1.0 + tilt_factor * (x_shifted / (0.5 * field_width_mm))

    y = plateau * horn * tilt * 100.0  # Percentage dose

    ground_truth = {
        "field_width_mm": field_width_mm,
        "penumbra_target_mm": penumbra_target_mm,
        "horn_factor": horn_factor,
        "tilt_factor": tilt_factor,
        "cax_offset_mm": cax_offset_mm,
        "expected_fwhm_mm": field_width_mm,
        "expected_penumbra_mm": penumbra_target_mm,
    }
    return x, y, ground_truth


def make_analytical_fff_profile(
    field_width_mm: float = 100.0,
    penumbra_target_mm: float = 6.0,
    lambda_decay: float = 0.0035,  # ~3.5%/cm for 6 FFF
    tilt_factor: float = 0.0,
    x_range_mm: float = 120.0,
    dx_mm: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generates an exact mathematical FFF unflattened profile:
        D(x) = exp(-lambda * |x|) * StepLeft * StepRight
    with inflection points positioned at +/- (field_width_mm / 2.0).
    """
    sigma = penumbra_target_mm / (4.0 * np.log(2.0))
    x = np.arange(-x_range_mm, x_range_mm + dx_mm / 2.0, dx_mm)

    left_edge = -field_width_mm / 2.0
    right_edge = field_width_mm / 2.0

    step_left = 1.0 / (1.0 + np.exp(-(x - left_edge) / sigma))
    step_right = 1.0 / (1.0 + np.exp((x - right_edge) / sigma))
    collimator_jaw = step_left * step_right

    forward_peak = np.exp(-lambda_decay * np.abs(x))
    tilt = 1.0 + tilt_factor * (x / (0.5 * field_width_mm))

    y = forward_peak * collimator_jaw * tilt * 100.0

    ground_truth = {
        "field_width_mm": field_width_mm,
        "penumbra_target_mm": penumbra_target_mm,
        "lambda_decay": lambda_decay,
        "expected_inflection_width_mm": field_width_mm,
    }
    return x, y, ground_truth


@pytest.fixture
def synthetic_flattened_6mv():
    """Synthetic 6 MV Flattened Profile with exact known geometry."""
    import beam_profile_analyzer as bpa
    p = bpa.generate_synthetic_profile(beam_type="6 MV Flattened")
    return p["x_mm"], p["values"], p["ground_truth"]


@pytest.fixture
def synthetic_fff_6mv():
    """Synthetic 6 MV FFF Profile with exact known geometry."""
    import beam_profile_analyzer as bpa
    p = bpa.generate_synthetic_profile(beam_type="6 MV FFF")
    return p["x_mm"], p["values"], p["ground_truth"]


@pytest.fixture
def synthetic_fff_10mv():
    """Synthetic 10 MV FFF Profile with exact known geometry."""
    import beam_profile_analyzer as bpa
    p = bpa.generate_synthetic_profile(beam_type="10 MV FFF")
    return p["x_mm"], p["values"], p["ground_truth"]


@pytest.fixture
def synthetic_tilted_flattened():
    """Synthetic Flattened Profile with 3% tilt across half-field."""
    return make_analytical_flattened_profile(100.0, 6.0, horn_factor=0.0, tilt_factor=0.03)


# ==============================================================================
# Sample Profile File Formats
# ==============================================================================

@pytest.fixture
def sample_csv_profile_text():
    """Generic CSV 2-column profile string."""
    header = "Position_mm,Dose_pct\n"
    rows = []
    for x in np.linspace(-60, 60, 121):
        # Top-hat with steep edge
        d = 100.0 if abs(x) <= 47.0 else (50.0 if abs(x) == 50.0 else (0.0 if abs(x) >= 53.0 else 100.0 - (abs(x) - 47.0) * 16.666))
        rows.append(f"{x:.1f},{d:.2f}")
    return header + "\n".join(rows)


@pytest.fixture
def sample_tsv_profile_text():
    """Tab-separated 2-column profile string."""
    header = "# SNC Profiler Export\n# Axis: Crossplane\nDistance\tValue\n"
    rows = []
    for x in np.linspace(-50, 50, 51):
        rows.append(f"{x:.1f}\t{100.0 - 0.05 * x * x:.2f}")
    return header + "\n".join(rows)


@pytest.fixture
def sample_iba_ascii_profile_text():
    """IBA OmniPro-Accept style ASCII block format."""
    return """<BEGIN_SCAN_DATA>
PARAM_ENERGY=6 MV
PARAM_FIELD_SIZE=100 100
PARAM_SCAN_TYPE=CROSSPLANE
PARAM_DEPTH=100.0
<BEGIN_DATA>
-60.00 0.50
-55.00 1.20
-50.00 50.00
-45.00 98.50
-40.00 99.80
-20.00 100.00
0.00 100.00
20.00 100.00
40.00 99.80
45.00 98.50
50.00 50.00
55.00 1.20
60.00 0.50
<END_DATA>
<END_SCAN_DATA>"""


@pytest.fixture
def sample_ptw_mcc_profile_text():
    """PTW Mephysto MCC style profile export."""
    return """[SCAN]
SCAN_AXIS = X
FIELD = 100 100
ENERGY = 6.000
DEPTH = 100.0
BEGIN_DATA
-60.00 0.52
-50.00 50.00
-40.00 99.75
0.00 100.00
40.00 99.81
50.00 50.00
60.00 0.48
END_DATA
"""
