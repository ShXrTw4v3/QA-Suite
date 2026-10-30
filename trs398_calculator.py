#!/usr/bin/env python3
"""
IAEA TRS-398 (Rev. 1, 2024) Reference Dosimetry CLI & Python Engine
High-Energy Photon Beams (Flattened & Flattening Filter-Free / FFF) and Electron Beams

Protocol: IAEA Technical Reports Series No. 398, Rev. 1 (2024)
"Absorbed Dose Determination in External Beam Radiotherapy: An International Code of Practice for Dosimetry Based on Standards of Absorbed Dose to Water"
Official Reference: https://www-pub.iaea.org/MTCD/Publications/PDF/p15048-DOC-010-398-Rev1_web.pdf

Traceability:
  - Air density correction k_TP: Section 4.4.3, Eq. (10) (273.15 K)
  - Polarity correction k_pol: Section 4.4.4, Eq. (11)
  - Two-voltage ion recombination k_s: Section 4.4.5.1, Eq. (13)-(14) & Table 10
  - FFF volume averaging k_vol: Section 4.4.6, Eq. (22) & Table 11
  - High-energy photon k_Q: Section 6.5.1, Eq. (34), Table 16 & Appendix II Table 45
  - High-energy electron R_50, z_ref & k_Q: Section 7.2-7.5, Eq. (37), Eq. (39), Table 20 & Appendix II Table 47
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import math
import sys
import argparse
import json

PROTOCOL_EDITION = "IAEA TRS-398 Rev. 1 (2024)"

# ==============================================================================
# IAEA TRS-398 Rev. 1 (2024) — Section 6 & Appendix II: High-Energy Photons
# ==============================================================================

# Table 16 (p. 90-93): TPR20,10 grid (12 points from 0.56 to 0.82)
KQ_PHOTON_TPRS: Tuple[float, ...] = (
    0.56, 0.59, 0.62, 0.65, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82
)

# Table 45 (p. 227-228): Chamber-type specific parameters (a, b) for Eq. (34)
KQ_PHOTON_FIT_PARAMS: Dict[str, Tuple[float, float]] = {
    "PTW 30013 Farmer (0.60 cm³, waterproof)": (1.18273, -0.13256),
    "PTW 30010 Farmer (0.60 cm³, classic)": (1.12594, -0.10740),
    "PTW 30011 Farmer (0.60 cm³)": (1.10850, -0.10107),
    "PTW 30012 Farmer (0.60 cm³)": (1.12442, -0.10415),
    "PTW 31010 Semiflex (0.125 cm³)": (1.23755, -0.15295),
    "PTW 31013 Semiflex (0.30 cm³)": (1.19297, -0.13366),
    "PTW 31016 PinPoint 3D (0.016 cm³)": (1.11650, -0.10841),
    "PTW 31021 Semiflex 3D (0.07 cm³)": (1.29612, -0.16514),
    "PTW 31022 PinPoint 3D (0.016 cm³)": (1.14435, -0.11130),
    "IBA / Wellhöfer FC65-G (0.65 cm³)": (1.09752, -0.09642),
    "IBA FC65-P Farmer (0.65 cm³)": (1.12374, -0.10784),
    "IBA FC23-C Short Farmer (0.23 cm³)": (1.09189, -0.09346),
    "IBA CC13 (0.13 cm³)": (1.11441, -0.10260),
    "IBA CC25 (0.25 cm³)": (1.08981, -0.09254),
    "NE 2571 Farmer (0.60 cm³)": (1.08918, -0.09222),
    "NE 2561 / NPL 2611A Secondary Standard": (1.07699, -0.08732),
    "Exradin A12 Farmer (0.64 cm³)": (1.09783, -0.09544),
    "Exradin A19 Classic Farmer (0.62 cm³)": (1.12024, -0.10493),
    "Exradin A12S Short Farmer (0.24 cm³)": (1.11499, -0.10057),
    "Exradin A1SL Miniature (0.053 cm³)": (1.21633, -0.13351),
    "Exradin A18 (0.125 cm³)": (1.10487, -0.09670),
    "Exradin A26 (0.015 cm³)": (1.09587, -0.09383),
    "Exradin A28 (0.125 cm³)": (1.12453, -0.10278),
    "Sun Nuclear SNC600c Farmer (0.60 cm³)": (1.06800, -0.08485),
    "Sun Nuclear SNC125c (0.125 cm³)": (1.09700, -0.09749),
    "Capintec PR-06C Farmer (0.65 cm³)": (1.06833, -0.08262),
}

# Table 16 (p. 90-93): Calculated values of k_Q for high-energy photon beams (4 decimal places)
KQ_PHOTON_CHAMBERS: Dict[str, Tuple[float, ...]] = {
    # PTW Chambers
    "PTW 30013 Farmer (0.60 cm³, waterproof)": (
        1.0007, 0.9984, 0.9956, 0.9920, 0.9876, 0.9840, 0.9800, 0.9753, 0.9699, 0.9636, 0.9565, 0.9484
    ),
    "PTW 30010 Farmer (0.60 cm³, classic)": (
        1.0005, 0.9989, 0.9967, 0.9938, 0.9901, 0.9869, 0.9832, 0.9787, 0.9734, 0.9671, 0.9595, 0.9506
    ),
    "PTW 30011 Farmer (0.60 cm³)": (
        1.0005, 0.9989, 0.9969, 0.9942, 0.9906, 0.9875, 0.9838, 0.9793, 0.9739, 0.9674, 0.9595, 0.9501
    ),
    "PTW 30012 Farmer (0.60 cm³)": (
        1.0004, 0.9990, 0.9970, 0.9944, 0.9910, 0.9881, 0.9846, 0.9804, 0.9754, 0.9694, 0.9622, 0.9536
    ),
    "PTW 31010 Semiflex (0.125 cm³)": (
        1.0008, 0.9982, 0.9952, 0.9914, 0.9869, 0.9835, 0.9795, 0.9750, 0.9700, 0.9643, 0.9579, 0.9507
    ),
    "PTW 31013 Semiflex (0.30 cm³)": (
        1.0007, 0.9985, 0.9958, 0.9924, 0.9882, 0.9848, 0.9810, 0.9765, 0.9714, 0.9655, 0.9588, 0.9511
    ),
    "PTW 31016 PinPoint 3D (0.016 cm³)": (
        1.0006, 0.9987, 0.9962, 0.9930, 0.9888, 0.9853, 0.9812, 0.9762, 0.9703, 0.9632, 0.9549, 0.9451
    ),
    "PTW 31021 Semiflex 3D (0.07 cm³)": (
        1.0007, 0.9984, 0.9957, 0.9925, 0.9886, 0.9856, 0.9823, 0.9786, 0.9744, 0.9697, 0.9645, 0.9587
    ),
    "PTW 31022 PinPoint 3D (0.016 cm³)": (
        1.0005, 0.9989, 0.9968, 0.9940, 0.9905, 0.9875, 0.9840, 0.9798, 0.9749, 0.9690, 0.9621, 0.9540
    ),
    # IBA Chambers
    "IBA / Wellhöfer FC65-G (0.65 cm³)": (
        1.0004, 0.9990, 0.9972, 0.9946, 0.9912, 0.9882, 0.9846, 0.9802, 0.9748, 0.9683, 0.9603, 0.9507
    ),
    "IBA FC65-P Farmer (0.65 cm³)": (
        1.0005, 0.9988, 0.9966, 0.9936, 0.9897, 0.9865, 0.9826, 0.9780, 0.9725, 0.9660, 0.9583, 0.9491
    ),
    "IBA FC23-C Short Farmer (0.23 cm³)": (
        1.0004, 0.9991, 0.9974, 0.9950, 0.9917, 0.9888, 0.9853, 0.9810, 0.9758, 0.9693, 0.9614, 0.9519
    ),
    "IBA CC13 (0.13 cm³)": (
        1.0005, 0.9989, 0.9969, 0.9942, 0.9906, 0.9876, 0.9839, 0.9795, 0.9742, 0.9678, 0.9601, 0.9510
    ),
    "IBA CC25 (0.25 cm³)": (
        1.0004, 0.9991, 0.9974, 0.9950, 0.9918, 0.9890, 0.9855, 0.9812, 0.9760, 0.9695, 0.9617, 0.9521
    ),
    # NE Chambers
    "NE 2571 Farmer (0.60 cm³)": (
        1.0004, 0.9991, 0.9974, 0.9951, 0.9919, 0.9891, 0.9856, 0.9813, 0.9761, 0.9697, 0.9618, 0.9522
    ),
    "NE 2561 / NPL 2611A Secondary Standard": (
        1.0003, 0.9992, 0.9977, 0.9955, 0.9925, 0.9898, 0.9865, 0.9823, 0.9771, 0.9706, 0.9627, 0.9528
    ),
    # Exradin Chambers
    "Exradin A12 Farmer (0.64 cm³)": (
        1.0004, 0.9991, 0.9973, 0.9948, 0.9915, 0.9887, 0.9852, 0.9809, 0.9756, 0.9693, 0.9615, 0.9521
    ),
    "Exradin A19 Classic Farmer (0.62 cm³)": (
        1.0005, 0.9989, 0.9968, 0.9940, 0.9904, 0.9873, 0.9836, 0.9792, 0.9738, 0.9675, 0.9599, 0.9509
    ),
    "Exradin A12S Short Farmer (0.24 cm³)": (
        1.0004, 0.9990, 0.9972, 0.9947, 0.9913, 0.9885, 0.9850, 0.9809, 0.9758, 0.9698, 0.9624, 0.9537
    ),
    "Exradin A1SL Miniature (0.053 cm³)": (
        1.0006, 0.9987, 0.9965, 0.9936, 0.9901, 0.9873, 0.9840, 0.9802, 0.9759, 0.9709, 0.9652, 0.9586
    ),
    "Exradin A18 (0.125 cm³)": (
        1.0004, 0.9991, 0.9973, 0.9949, 0.9917, 0.9889, 0.9855, 0.9814, 0.9764, 0.9702, 0.9628, 0.9538
    ),
    "Exradin A26 (0.015 cm³)": (
        1.0004, 0.9991, 0.9974, 0.9951, 0.9919, 0.9891, 0.9857, 0.9816, 0.9765, 0.9702, 0.9626, 0.9533
    ),
    "Exradin A28 (0.125 cm³)": (
        1.0004, 0.9990, 0.9972, 0.9947, 0.9914, 0.9886, 0.9853, 0.9813, 0.9764, 0.9706, 0.9636, 0.9552
    ),
    # Sun Nuclear & Capintec Chambers
    "Sun Nuclear SNC600c Farmer (0.60 cm³)": (
        1.0004, 0.9993, 0.9978, 0.9957, 0.9926, 0.9899, 0.9866, 0.9823, 0.9770, 0.9703, 0.9620, 0.9517
    ),
    "Sun Nuclear SNC125c (0.125 cm³)": (
        1.0005, 0.9991, 0.9971, 0.9944, 0.9908, 0.9878, 0.9840, 0.9794, 0.9739, 0.9671, 0.9590, 0.9492
    ),
    "Capintec PR-06C Farmer (0.65 cm³)": (
        1.0003, 0.9993, 0.9980, 0.9961, 0.9934, 0.9909, 0.9878, 0.9839, 0.9790, 0.9727, 0.9649, 0.9551
    ),
}

# Table 4 (p. 40): Chamber cavity lengths in mm for volume averaging (k_vol) evaluation (Eq. 22)
CHAMBER_LENGTHS_MM: Dict[str, float] = {
    "PTW 30013": 23.0,
    "PTW 30010": 23.0,
    "PTW 30011": 23.0,
    "PTW 30012": 23.0,
    "PTW 31010": 6.5,
    "PTW 31013": 16.3,
    "PTW 31016": 2.9,
    "PTW 31021": 4.8,
    "PTW 31022": 2.9,
    "FC65-G": 23.0,
    "FC65-P": 23.0,
    "FC23-C": 10.4,
    "CC13": 5.8,
    "CC25": 10.0,
    "2571": 24.1,
    "2561": 9.2,
    "2611A": 9.2,
    "A12S": 7.2,
    "A12": 21.6,
    "A19": 21.6,
    "A1SL": 5.7,
    "A18": 6.3,
    "A26": 1.8,
    "A28": 6.3,
    "SNC600c": 23.0,
    "SNC125c": 7.0,
    "PR-06C": 22.0,
}

# ==============================================================================
# IAEA TRS-398 Rev. 1 (2024) — Section 7 & Appendix II: Electron Beams
# ==============================================================================

# Table 20 (p. 110): R_50 grid in g/cm²
KQ_ELECTRON_R50S: Tuple[float, ...] = (
    1.0, 1.4, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0
)

# Table 20 (p. 110): Calculated k_Q values for electron beams calibrated in 60Co
# Note: Cylindrical chambers are valid only for R_50 >= 3.0 g/cm² (None for R_50 < 3.0)
KQ_ELECTRON_CHAMBERS: Dict[str, Tuple[Optional[float], ...]] = {
    # Plane-parallel chambers (valid across full R_50 range 1.0 - 10.0 g/cm²)
    "PTW 34001 Roos (Plane-Parallel)": (
        0.9743, 0.9645, 0.9518, 0.9428, 0.9349, 0.9281, 0.9222, 0.9171, 0.9127, 0.9088, 0.9055, 0.9001, 0.8960, 0.8907
    ),
    "IBA NACP-02 (Plane-Parallel)": (
        0.9679, 0.9580, 0.9451, 0.9360, 0.9281, 0.9214, 0.9155, 0.9104, 0.9061, 0.9023, 0.8990, 0.8938, 0.8899, 0.8848
    ),
    # Cylindrical chambers (valid for R_50 >= 3.0 g/cm²)
    "PTW 30013 Farmer (Cylindrical)": (
        None, None, None, None, 0.9300, 0.9247, 0.9210, 0.9180, 0.9155, 0.9135, 0.9118, 0.9090, 0.9068, 0.9037
    ),
    "PTW 30012 Farmer (Cylindrical)": (
        None, None, None, None, 0.9384, 0.9331, 0.9290, 0.9257, 0.9230, 0.9207, 0.9187, 0.9156, 0.9131, 0.9094
    ),
    "IBA FC65-G Farmer (Cylindrical)": (
        None, None, None, None, 0.9328, 0.9279, 0.9241, 0.9210, 0.9184, 0.9163, 0.9144, 0.9114, 0.9090, 0.9055
    ),
    "NE 2571 Farmer (Cylindrical)": (
        None, None, None, None, 0.9297, 0.9248, 0.9210, 0.9181, 0.9157, 0.9137, 0.9121, 0.9095, 0.9075, 0.9047
    ),
    "Exradin A12 Farmer (Cylindrical)": (
        None, None, None, None, 0.9353, 0.9304, 0.9264, 0.9232, 0.9205, 0.9182, 0.9163, 0.9130, 0.9104, 0.9066
    ),
    "Exradin A19 Classic Farmer (Cylindrical)": (
        None, None, None, None, 0.9320, 0.9273, 0.9236, 0.9205, 0.9178, 0.9156, 0.9136, 0.9103, 0.9077, 0.9036
    ),
}

# Table 47 (p. 235): Fitting parameters (a, b, c, chamber_type) for Eqs. (98) and (99)
KQ_ELECTRON_FIT_PARAMS: Dict[str, Tuple[float, float, float, str]] = {
    "PTW 34001 Roos (Plane-Parallel)": (0.884, 0.120, 3.511, "pp"),
    "IBA NACP-02 (Plane-Parallel)": (0.879, 0.120, 3.398, "pp"),
    "PTW 30013 Farmer (Cylindrical)": (0.888, 0.101, 0.816, "cyl"),
    "PTW 30012 Farmer (Cylindrical)": (0.889, 0.111, 0.728, "cyl"),
    "IBA FC65-G Farmer (Cylindrical)": (0.884, 0.103, 0.680, "cyl"),
    "NE 2571 Farmer (Cylindrical)": (0.892, 0.103, 0.920, "cyl"),
    "Exradin A12 Farmer (Cylindrical)": (0.880, 0.108, 0.607, "cyl"),
    "Exradin A19 Classic Farmer (Cylindrical)": (0.870, 0.108, 0.505, "cyl"),
}

# ==============================================================================
# IAEA TRS-398 Rev. 1 (2024) — Table 10 (p. 62): Two-Voltage Recombination Fits
# ==============================================================================

V1_V2_FITS: Dict[float, Dict[str, float]] = {
    2.0: {"a0": 2.337, "a1": -3.636, "a2": 2.299},
    2.5: {"a0": 1.474, "a1": -1.587, "a2": 1.114},
    3.0: {"a0": 1.198, "a1": -0.875, "a2": 0.677},
    3.5: {"a0": 1.080, "a1": -0.542, "a2": 0.463},
    4.0: {"a0": 1.022, "a1": -0.363, "a2": 0.341},
    5.0: {"a0": 0.975, "a1": -0.188, "a2": 0.214},
}

V1_V2_FITS_PULSED_SCANNED: Dict[float, Dict[str, float]] = {
    2.0: {"a0": 4.711, "a1": -8.242, "a2": 4.533},
    2.5: {"a0": 2.719, "a1": -3.977, "a2": 2.261},
    3.0: {"a0": 2.001, "a1": -2.402, "a2": 1.404},
    3.5: {"a0": 1.665, "a1": -1.647, "a2": 0.984},
    4.0: {"a0": 1.468, "a1": -1.200, "a2": 0.734},
    5.0: {"a0": 1.279, "a1": -0.750, "a2": 0.474},
}

# Standard Clinical Presets Library (with TRS-398 Rev. 1 Eq. 22 k_vol values at SDD = 110 cm)
PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "TrueBeam_6MV_W": {
        "name": "Varian TrueBeam - 6 MV (Flattened)",
        "energy": "6MV",
        "is_fff": False,
        "tpr2010": 0.670,
        "pdd_zref": 67.0,
        "dmax": 1.5,
        "dose_rate": 600,
        "chamber": "PTW 30013 Farmer (0.60 cm³, waterproof)",
        "nd_w": 0.05385,
        "k_vol": 1.0000
    },
    "TrueBeam_6MV_FFF": {
        "name": "Varian TrueBeam - 6 MV FFF",
        "energy": "6MV FFF",
        "is_fff": True,
        "tpr2010": 0.635,
        "pdd_zref": 63.5,
        "dmax": 1.5,
        "dose_rate": 1400,
        "chamber": "PTW 30013 Farmer (0.60 cm³, waterproof)",
        "nd_w": 0.05385,
        "k_vol": 1.0015  # TRS-398 Rev. 1 Eq. (22) for L=2.3 cm, TPR20,10=0.635, SDD=110 cm
    },
    "TrueBeam_10MV_W": {
        "name": "Varian TrueBeam - 10 MV (Flattened)",
        "energy": "10MV",
        "is_fff": False,
        "tpr2010": 0.735,
        "pdd_zref": 73.5,
        "dmax": 2.4,
        "dose_rate": 600,
        "chamber": "PTW 30013 Farmer (0.60 cm³, waterproof)",
        "nd_w": 0.05385,
        "k_vol": 1.0000
    },
    "TrueBeam_10MV_FFF": {
        "name": "Varian TrueBeam - 10 MV FFF",
        "energy": "10MV FFF",
        "is_fff": True,
        "tpr2010": 0.705,
        "pdd_zref": 70.5,
        "dmax": 2.2,
        "dose_rate": 2400,
        "chamber": "PTW 30013 Farmer (0.60 cm³, waterproof)",
        "nd_w": 0.05385,
        "k_vol": 1.0034  # TRS-398 Rev. 1 Eq. (22) for L=2.3 cm, TPR20,10=0.705, SDD=110 cm
    },
    "VersaHD_6MV_FFF": {
        "name": "Elekta Versa HD - 6 MV FFF",
        "energy": "6MV FFF",
        "is_fff": True,
        "tpr2010": 0.638,
        "pdd_zref": 63.8,
        "dmax": 1.5,
        "dose_rate": 1400,
        "chamber": "PTW 30013 Farmer (0.60 cm³, waterproof)",
        "nd_w": 0.05385,
        "k_vol": 1.0016  # TRS-398 Rev. 1 Eq. (22) for L=2.3 cm, TPR20,10=0.638, SDD=110 cm
    }
}


def find_chamber_key(query: str) -> str:
    """Finds matching photon chamber key from partial query string."""
    query_clean = query.lower().strip()
    # 1. Exact substring match
    for key in KQ_PHOTON_CHAMBERS:
        if query_clean in key.lower():
            return key

    # 2. Token / model matching (e.g. "fc65-g", "30013", "cc13", "31010")
    import re
    tokens = [t for t in re.split(r"[\s\-_/]+", query_clean) if len(t) > 1]
    best_key = None
    best_matches = 0
    for key in KQ_PHOTON_CHAMBERS:
        key_lower = key.lower()
        matches = sum(1 for t in tokens if t in key_lower)
        if matches > best_matches:
            best_matches = matches
            best_key = key

    if best_key and best_matches >= max(1, len(tokens) - 1):
        return best_key

    return "PTW 30013 Farmer (0.60 cm³, waterproof)"


def find_electron_chamber_key(query: str) -> str:
    """Finds matching electron chamber key from partial query string."""
    query_lower = query.lower()
    for key in KQ_ELECTRON_CHAMBERS:
        if query_lower in key.lower():
            return key
    return "PTW 34001 Roos (Plane-Parallel)"


def get_chamber_cavity_length(chamber_key: str) -> float:
    """Returns cavity length in mm for volume averaging correction (Table 4)."""
    for key, length in CHAMBER_LENGTHS_MM.items():
        if key.lower() in chamber_key.lower():
            return length
    return 23.0  # Default Farmer length


def estimate_kvol_fff(
    chamber_key: str,
    is_fff: bool,
    tpr_or_energy: float = 0.635,
    sdd_cm: float = 110.0
) -> float:
    """
    Calculates generic volume averaging correction factor k_vol for FFF photon beams
    according to IAEA TRS-398 Rev. 1 (2024) Section 4.4.6, Eq. (22) & Table 11 (p. 67-68):
        k_vol = 1 + (0.0062 * TPR20,10 - 0.0036) * (100 / SDD)^2 * L^2
    where L is the chamber cavity length in cm and SDD is source-detector distance in cm
    (SDD = SSD + z_ref = 110 cm for SSD=100 cm, z_ref=10 cm, or 100 cm for SAD setup).

    Accepts either TPR20,10 (0.50 <= tpr_or_energy <= 0.86) or nominal energy in MV (e.g. 6, 10).
    """
    if not is_fff:
        return 1.0000

    if 0.45 <= tpr_or_energy <= 0.95:
        tpr2010 = tpr_or_energy
    else:
        # Fallback if caller passed nominal MV (e.g. 6 or 10)
        tpr2010 = 0.705 if tpr_or_energy >= 10.0 else 0.635

    length_cm = get_chamber_cavity_length(chamber_key) / 10.0
    sdd = sdd_cm if sdd_cm > 0 else 110.0
    kvol = 1.0 + (0.0062 * tpr2010 - 0.0036) * ((100.0 / sdd) ** 2) * (length_cm ** 2)
    return round(max(1.0000, kvol), 4)


def calc_kq_photon_analytical(chamber_key: str, tpr2010: float) -> float:
    """
    Calculates photon beam quality correction factor k_Q directly from
    IAEA TRS-398 Rev. 1 (2024) Section 6.5.1, Eq. (34) (p. 89) and Table 45 (p. 227-228):
        k_Q(TPR20,10) = (1 + exp((a - 0.57) / b)) / (1 + exp((a - TPR20,10) / b))
    """
    resolved_key = find_chamber_key(chamber_key)
    a, b = KQ_PHOTON_FIT_PARAMS[resolved_key]
    num = 1.0 + math.exp((a - 0.57) / b)
    den = 1.0 + math.exp((a - tpr2010) / b)
    return num / den


def interpolate_kq(chamber_key: str, tpr2010: float) -> float:
    """
    Linearly interpolates k_Q for a given chamber and TPR20,10 from
    IAEA TRS-398 Rev. 1 (2024) Table 16 (p. 90-93).
    """
    resolved_key = find_chamber_key(chamber_key)
    tprs = KQ_PHOTON_TPRS
    kqs = KQ_PHOTON_CHAMBERS[resolved_key]

    if tpr2010 <= tprs[0]:
        return kqs[0]
    if tpr2010 >= tprs[-1]:
        return kqs[-1]

    for i in range(len(tprs) - 1):
        if tprs[i] <= tpr2010 <= tprs[i + 1]:
            slope = (kqs[i + 1] - kqs[i]) / (tprs[i + 1] - tprs[i])
            return kqs[i] + slope * (tpr2010 - tprs[i])
    return kqs[0]


def calc_ktp(temp_c: float, press_kpa: float, t0_c: float = 20.0, p0_kpa: float = 101.325) -> float:
    """
    Calculates temperature and pressure air density correction factor k_TP
    according to IAEA TRS-398 Rev. 1 (2024) Section 4.4.3, Eq. (10) (p. 56):
        k_TP = ((273.15 + T) / (273.15 + T0)) * (P0 / P)
    """
    return ((273.15 + temp_c) / (273.15 + t0_c)) * (p0_kpa / press_kpa)


def calc_kpol(m_user: float, m_opp: float) -> float:
    """Calculates polarity correction factor k_pol (TRS-398 Rev. 1 Section 4.4.4, Eq. 11)."""
    if abs(m_user) < 1e-12:
        return 1.0
    return (abs(m_user) + abs(m_opp)) / (2.0 * abs(m_user))


def calc_ks_pulsed(v1: float, v2: float, m1: float, m2: float, pulsed_scanned: bool = False) -> Tuple[float, str]:
    """
    Calculates ion recombination factor k_s using IAEA TRS-398 Rev. 1 (2024)
    Section 4.4.5.1, Eq. (13)-(14) and Table 10 (p. 62) quadratic fit coefficients.
    """
    if abs(v2) < 1e-12 or abs(m2) < 1e-12:
        return 1.0, "Invalid voltage/reading ratio"

    ratio_v = abs(v1 / v2)
    ratio_m = m1 / m2
    fits_table = V1_V2_FITS_PULSED_SCANNED if pulsed_scanned else V1_V2_FITS
    beam_label = "pulsed-scanned" if pulsed_scanned else "pulsed"

    best_match = None
    min_diff = 1e9
    for standard_ratio in sorted(fits_table.keys()):
        diff = abs(ratio_v - standard_ratio)
        if diff < min_diff:
            min_diff = diff
            best_match = standard_ratio

    if min_diff <= 0.08 and best_match is not None:
        coeffs = fits_table[best_match]
        ks = coeffs["a0"] + coeffs["a1"] * ratio_m + coeffs["a2"] * (ratio_m ** 2)
        return ks, f"TRS-398 Rev. 1 Table 10 quadratic fit ({beam_label}, V1/V2 = {best_match:.1f})"
    else:
        denom = ratio_v - ratio_m
        if abs(denom) > 1e-6:
            ks = (ratio_v - 1.0) / denom
            return ks, f"TRS-398 Rev. 1 Eq. (14) linear approximation (V1/V2 = {ratio_v:.2f})"
        return 1.0, "Undefined (M1/M2 equals V1/V2)"


def calc_dose_per_pulse(dose_rate_mu_min: float, prf_hz: float = 360.0) -> float:
    """
    Estimates dose per pulse (DPP) in mGy/pulse.
    DPP = (DoseRate in cGy/min * 10 mGy/cGy) / (60 s/min * PRF Hz)
    """
    if prf_hz <= 0:
        return 0.0
    return (dose_rate_mu_min * 10.0) / (60.0 * prf_hz)


def calc_trs398_dose(
    raw_readings: List[float],
    mu_delivered: float = 100.0,
    nd_w: float = 0.05385,  # in Gy/nC (53.85 mGy/nC)
    tpr2010: float = 0.67,
    chamber_key: str = "30013",
    temp_c: float = 20.0,
    press_kpa: float = 101.325,
    t0_c: float = 20.0,
    p0_kpa: float = 101.325,
    k_elec: float = 1.000,
    m_plus: Optional[float] = None,
    m_minus: Optional[float] = None,
    v1: Optional[float] = 300.0,
    v2: Optional[float] = 100.0,
    m1: Optional[float] = None,
    m2: Optional[float] = None,
    pdd_zref: float = 67.0,  # % at z_ref = 10 cm
    target_dose_per_mu: float = 0.01000,  # 1.000 cGy/MU = 0.01 Gy/MU
    is_fff: bool = False,
    k_vol: Optional[float] = None,
    dose_rate_mu_min: float = 600.0,
    prf_hz: float = 360.0,
    sdd_cm: float = 110.0,
) -> Dict[str, Any]:
    """
    Performs full IAEA TRS-398 Rev. 1 (2024) photon beam reference dosimetry calculation
    with Flattening Filter-Free (FFF) support.
    """
    if not raw_readings:
        raise ValueError("At least one electrometer reading must be provided.")

    m_raw_avg = sum(raw_readings) / len(raw_readings)
    std_dev = math.sqrt(sum((x - m_raw_avg) ** 2 for x in raw_readings) / (len(raw_readings) - 1)) if len(raw_readings) > 1 else 0.0
    cv_pct = (std_dev / m_raw_avg * 100.0) if m_raw_avg != 0 else 0.0

    # Air density correction (Eq. 10)
    ktp = calc_ktp(temp_c, press_kpa, t0_c, p0_kpa)

    # Polarity correction (Eq. 11)
    if m_plus is not None and m_minus is not None and abs(m_plus) > 0:
        k_pol = calc_kpol(m_plus, m_minus)
    else:
        k_pol = 1.000

    # Ion recombination (Table 10)
    if v1 is not None and v2 is not None and m1 is not None and m2 is not None and m1 > 0 and m2 > 0:
        k_s, ks_method = calc_ks_pulsed(v1, v2, m1, m2)
    else:
        k_s = 1.000
        ks_method = "Unity default (not measured)"

    # Volume averaging correction for FFF beams (Eq. 22 & Table 11)
    resolved_chamber = find_chamber_key(chamber_key)
    if k_vol is None:
        k_vol = estimate_kvol_fff(resolved_chamber, is_fff, tpr2010, sdd_cm=sdd_cm)
    elif not is_fff:
        k_vol = 1.0000

    # Total influence correction factor
    k_tot = ktp * k_pol * k_s * k_elec

    # Fully corrected chamber reading M_Q
    m_q = m_raw_avg * k_tot

    # Beam quality correction factor k_Q from Table 16 / Eq. (34)
    k_q = interpolate_kq(resolved_chamber, tpr2010)
    k_q_analytical = calc_kq_photon_analytical(resolved_chamber, tpr2010)

    # Dose per pulse estimation
    dpp_mgy = calc_dose_per_pulse(dose_rate_mu_min, prf_hz)

    # Absorbed dose to water at reference depth z_ref = 10 cm (Gy)
    # TRS-398 Rev. 1 Section 6.3 / 4.4.6: D_w,Q(z_ref) = M_Q * N_D,w,Q0 * k_Q * k_vol
    d_w_zref = m_q * nd_w * k_q * k_vol

    # Dose per MU at reference depth z_ref
    dose_per_mu_zref = d_w_zref / mu_delivered if mu_delivered > 0 else 0.0
    cgy_per_mu_zref = dose_per_mu_zref * 100.0

    # Absorbed dose at z_max (d_max) using PDD(z_ref)
    # TRS-398 Rev. 1 Eq. (32): D_w,Q(z_max) = 100 * D_w,Q(z_ref) / PDD(z_ref)
    d_w_zmax = d_w_zref * (100.0 / pdd_zref) if pdd_zref > 0 else d_w_zref
    dose_per_mu_zmax = d_w_zmax / mu_delivered if mu_delivered > 0 else 0.0
    cgy_per_mu_zmax = dose_per_mu_zmax * 100.0

    # Output deviation from target
    target_cgy_per_mu = target_dose_per_mu * 100.0
    diff_pct = ((cgy_per_mu_zmax - target_cgy_per_mu) / target_cgy_per_mu) * 100.0 if target_cgy_per_mu > 0 else 0.0

    # Recommended Linac adjustment factor
    adj_pct = -diff_pct / (1.0 + diff_pct / 100.0) if (1.0 + diff_pct / 100.0) != 0 else 0.0

    # Clinical tolerance check
    if abs(diff_pct) <= 1.0:
        compliance = "PASS (Within ±1.0% clinical tolerance)"
    elif abs(diff_pct) <= 2.0:
        compliance = "ACTION LEVEL (±1.0% to ±2.0% warning)"
    else:
        compliance = "OUT OF TOLERANCE (> ±2.0% - immediate recalibration required)"

    # High recombination warning for FFF
    ks_warning = None
    if k_s > 1.03:
        ks_warning = f"High ion recombination (k_s = {k_s:.4f}). Typical for high dose-per-pulse FFF beams. Verify Jaffé plot linearity (TRS-398 Rev. 1 Section 4.4.5.1)."

    return {
        "protocol_edition": PROTOCOL_EDITION,
        "beam_mode": "Flattening Filter-Free (FFF)" if is_fff else "Conventional Flattened (W)",
        "is_fff": is_fff,
        "k_vol": k_vol,
        "dose_rate_mu_min": dose_rate_mu_min,
        "dose_per_pulse_mgy": dpp_mgy,
        "ks_warning": ks_warning,
        "m_raw_avg_nc": m_raw_avg,
        "raw_readings": raw_readings,
        "std_dev_nc": std_dev,
        "cv_pct": cv_pct,
        "k_tp": ktp,
        "k_pol": k_pol,
        "k_s": k_s,
        "k_s_method": ks_method,
        "k_elec": k_elec,
        "k_tot": k_tot,
        "m_q_nc": m_q,
        "k_q": k_q,
        "kq": k_q,
        "k_q_analytical": k_q_analytical,
        "tpr2010": tpr2010,
        "pdd_zref_pct": pdd_zref,
        "chamber": resolved_chamber,
        "nd_w_gy_per_nc": nd_w,
        "mu_delivered": mu_delivered,
        "d_w_zref_gy": d_w_zref,
        "d_w_zref_cgy": d_w_zref * 100.0,
        "dose_per_mu_zref_cgy": cgy_per_mu_zref,
        "d_w_zmax_gy": d_w_zmax,
        "d_w_zmax_cgy": d_w_zmax * 100.0,
        "dose_per_mu_zmax_cgy": cgy_per_mu_zmax,
        "dose_per_mu_dmax_cgy": cgy_per_mu_zmax,
        "target_cgy_per_mu": target_cgy_per_mu,
        "diff_pct": diff_pct,
        "recommended_mu_trim_pct": adj_pct,
        "compliance": compliance
    }


# ==============================================================================
# IAEA TRS-398 Rev. 1 (2024) — Section 7 Electron Beam Reference Dosimetry
# ==============================================================================

def calc_electron_r50_from_i50(r50_ion_cm: float) -> float:
    """
    Converts half-value depth of ionization R_50,ion (I_50) in water to R_50 (g/cm²)
    according to IAEA TRS-398 Rev. 1 (2024) Section 7.2.2, Eq. (37)-(38) (p. 107):
      - R_50 = 1.029 * R_50,ion - 0.06  (for R_50,ion <= 10 g/cm²)
      - R_50 = 1.059 * R_50,ion - 0.37  (for R_50,ion > 10 g/cm²)
    """
    if r50_ion_cm <= 10.0:
        return 1.029 * r50_ion_cm - 0.06
    return 1.059 * r50_ion_cm - 0.37


def calc_electron_zref(r50_cm: float) -> float:
    """
    Calculates reference depth z_ref (g/cm²) in water for electron beams
    according to IAEA TRS-398 Rev. 1 (2024) Section 7.3.2, Eq. (39) (p. 108):
        z_ref = 0.6 * R_50 - 0.1
    """
    return 0.6 * r50_cm - 0.1


def calc_kq_electron_analytical(chamber_key: str, r50_cm: float) -> float:
    """
    Calculates electron beam quality correction factor k_Q directly from
    IAEA TRS-398 Rev. 1 (2024) Appendix II, Eqs. (98)-(99) & Table 47 (p. 235):
      - Plane-parallel (Eq. 98): k_Q,pp = a + b * exp(-R_50 / c)
      - Cylindrical (Eq. 99):    k_Q,cyl = a + b * (R_50 ** (-c))  [for R_50 >= 3.0 g/cm²]
    """
    resolved = find_electron_chamber_key(chamber_key)
    a, b, c, ch_type = KQ_ELECTRON_FIT_PARAMS[resolved]
    if ch_type == "cyl" and r50_cm < 3.0:
        raise ValueError(
            f"Cylindrical chamber '{resolved}' is only recommended for R_50 >= 3.0 g/cm² "
            f"(received R_50 = {r50_cm:.2f} g/cm²). Use a plane-parallel chamber per IAEA TRS-398 Rev. 1 Table 19 & Table 20."
        )
    if ch_type == "pp":
        return a + b * math.exp(-r50_cm / c)
    return a + b * (r50_cm ** (-c))


def interpolate_kq_electron(chamber_key: str, r50_cm: float) -> float:
    """
    Linearly interpolates electron k_Q from IAEA TRS-398 Rev. 1 (2024) Table 20 (p. 110).
    Raises ValueError if a cylindrical chamber is selected for R_50 < 3.0 g/cm².
    """
    resolved = find_electron_chamber_key(chamber_key)
    r50s = KQ_ELECTRON_R50S
    kqs = KQ_ELECTRON_CHAMBERS[resolved]
    _, _, _, ch_type = KQ_ELECTRON_FIT_PARAMS[resolved]

    if ch_type == "cyl" and r50_cm < 3.0:
        raise ValueError(
            f"Cylindrical chamber '{resolved}' is not valid for R_50 = {r50_cm:.2f} g/cm² "
            f"(requires R_50 >= 3.0 g/cm² per IAEA TRS-398 Rev. 1 Table 19 & Table 20)."
        )

    valid_pairs = [(r, k) for r, k in zip(r50s, kqs) if k is not None]
    if r50_cm <= valid_pairs[0][0]:
        return float(valid_pairs[0][1])
    if r50_cm >= valid_pairs[-1][0]:
        return float(valid_pairs[-1][1])

    for i in range(len(valid_pairs) - 1):
        r0, k0 = valid_pairs[i]
        r1, k1 = valid_pairs[i + 1]
        if r0 <= r50_cm <= r1:
            slope = (k1 - k0) / (r1 - r0)
            return float(k0 + slope * (r50_cm - r0))
    return float(valid_pairs[0][1])


def calc_trs398_electron_dose(
    raw_readings: List[float],
    mu_delivered: float = 100.0,
    nd_w: float = 0.05385,  # in Gy/nC
    i50_cm: Optional[float] = None,
    r50_cm: Optional[float] = None,
    chamber_key: str = "PTW 34001 Roos (Plane-Parallel)",
    temp_c: float = 20.0,
    press_kpa: float = 101.325,
    t0_c: float = 20.0,
    p0_kpa: float = 101.325,
    k_elec: float = 1.000,
    m_plus: Optional[float] = None,
    m_minus: Optional[float] = None,
    v1: Optional[float] = 300.0,
    v2: Optional[float] = 100.0,
    m1: Optional[float] = None,
    m2: Optional[float] = None,
    pdd_zref: float = 100.0,  # % at z_ref relative to z_max
    target_dose_per_mu: float = 0.01000,  # 1.000 cGy/MU
) -> Dict[str, Any]:
    """
    Performs full IAEA TRS-398 Rev. 1 (2024) Section 7 electron beam reference dosimetry.
    """
    if not raw_readings:
        raise ValueError("At least one electrometer reading must be provided.")
    if r50_cm is None:
        if i50_cm is None:
            raise ValueError("Either r50_cm or i50_cm must be provided.")
        r50_cm = calc_electron_r50_from_i50(i50_cm)
    if r50_cm <= 0:
        raise ValueError("R_50 must be strictly positive")

    z_ref_cm = calc_electron_zref(r50_cm)
    resolved_chamber = find_electron_chamber_key(chamber_key)
    _, _, _, ch_type = KQ_ELECTRON_FIT_PARAMS[resolved_chamber]

    m_raw_avg = sum(raw_readings) / len(raw_readings)
    std_dev = math.sqrt(sum((x - m_raw_avg) ** 2 for x in raw_readings) / (len(raw_readings) - 1)) if len(raw_readings) > 1 else 0.0
    cv_pct = (std_dev / m_raw_avg * 100.0) if m_raw_avg != 0 else 0.0

    ktp = calc_ktp(temp_c, press_kpa, t0_c, p0_kpa)
    k_pol = calc_kpol(m_plus, m_minus) if (m_plus is not None and m_minus is not None and abs(m_plus) > 0) else 1.000

    if v1 is not None and v2 is not None and m1 is not None and m2 is not None and m1 > 0 and m2 > 0:
        k_s, ks_method = calc_ks_pulsed(v1, v2, m1, m2)
    else:
        k_s = 1.000
        ks_method = "Unity default (not measured)"

    k_tot = ktp * k_pol * k_s * k_elec
    m_q = m_raw_avg * k_tot

    k_q = interpolate_kq_electron(resolved_chamber, r50_cm)
    k_q_analytical = calc_kq_electron_analytical(resolved_chamber, r50_cm)

    d_w_zref = m_q * nd_w * k_q
    cgy_per_mu_zref = (d_w_zref * 100.0) / mu_delivered if mu_delivered > 0 else 0.0

    d_w_zmax = d_w_zref * (100.0 / pdd_zref) if pdd_zref > 0 else d_w_zref
    cgy_per_mu_zmax = (d_w_zmax * 100.0) / mu_delivered if mu_delivered > 0 else 0.0

    target_cgy_per_mu = target_dose_per_mu * 100.0
    diff_pct = ((cgy_per_mu_zmax - target_cgy_per_mu) / target_cgy_per_mu) * 100.0 if target_cgy_per_mu > 0 else 0.0
    adj_pct = -diff_pct / (1.0 + diff_pct / 100.0) if (1.0 + diff_pct / 100.0) != 0 else 0.0

    return {
        "protocol_edition": PROTOCOL_EDITION,
        "chamber": resolved_chamber,
        "chamber_type": "Plane-Parallel" if ch_type == "pp" else "Cylindrical (0.5 r_cyl shift deeper than z_ref)",
        "i50_cm": i50_cm,
        "r50_cm": r50_cm,
        "z_ref_cm": z_ref_cm,
        "pdd_zref_pct": pdd_zref,
        "m_raw_avg_nc": m_raw_avg,
        "std_dev_nc": std_dev,
        "cv_pct": cv_pct,
        "k_tp": ktp,
        "k_pol": k_pol,
        "k_s": k_s,
        "k_s_method": ks_method,
        "k_elec": k_elec,
        "k_tot": k_tot,
        "m_q_nc": m_q,
        "k_q": k_q,
        "kq": k_q,
        "k_q_analytical": k_q_analytical,
        "nd_w_gy_per_nc": nd_w,
        "mu_delivered": mu_delivered,
        "d_w_zref_gy": d_w_zref,
        "d_w_zref_cgy": d_w_zref * 100.0,
        "dose_per_mu_zref_cgy": cgy_per_mu_zref,
        "d_w_zmax_gy": d_w_zmax,
        "d_w_zmax_cgy": d_w_zmax * 100.0,
        "dose_per_mu_zmax_cgy": cgy_per_mu_zmax,
        "dose_per_mu_dmax_cgy": cgy_per_mu_zmax,
        "target_cgy_per_mu": target_cgy_per_mu,
        "diff_pct": diff_pct,
        "recommended_mu_trim_pct": adj_pct,
    }


def print_cli_report(res: Dict[str, Any]):
    """Prints a clinical report in terminal."""
    sign = "+" if res["diff_pct"] >= 0 else ""
    trim_sign = "+" if res["recommended_mu_trim_pct"] >= 0 else ""

    print("\n" + "=" * 76)
    print("  IAEA TRS-398 (REV. 1, 2024) REFERENCE DOSIMETRY: HIGH-ENERGY PHOTONS")
    print("=" * 76)
    print(f"Protocol Edition:        {res['protocol_edition']}")
    print(f"Beam Quality Mode:       {res['beam_mode']}")
    print(f"Chamber:                 {res['chamber']}")
    print(f"Beam Quality Index:      TPR20,10 = {res['tpr2010']:.3f}")
    print(f"Setup Geometry:          SSD = 100 cm | z_ref = 10.0 cm | d_max = 1.5 cm")
    print(f"Percentage Depth Dose:   PDD(10) = {res['pdd_zref_pct']:.1f}%")
    print(f"Calibration Factor:      N_D,w = {res['nd_w_gy_per_nc']:.6f} Gy/nC ({res['nd_w_gy_per_nc']*1000:.2f} mGy/nC)")
    if res["is_fff"]:
        print(f"Dose Rate & Pulse Dose:  {res['dose_rate_mu_min']:.0f} MU/min  |  DPP = {res['dose_per_pulse_mgy']:.3f} mGy/pulse")
    print("-" * 76)
    print("CORRECTION FACTORS:")
    print(f"  k_TP (Air density):    {res['k_tp']:.4f}  (Eq. 10, 273.15 K)")
    print(f"  k_pol (Polarity):      {res['k_pol']:.4f}  (Eq. 11)")
    print(f"  k_s (Recombination):   {res['k_s']:.4f}  [{res['k_s_method']}]")
    print(f"  k_elec (Electrometer): {res['k_elec']:.4f}")
    print(f"  Total Correction ktot: {res['k_tot']:.4f}")
    print(f"  Beam Quality k_Q:      {res['k_q']:.4f}  (TRS-398 Rev. 1 Table 16 / Eq. 34)")
    if res["is_fff"]:
        print(f"  Volume Averaging kvol: {res['k_vol']:.4f}  (TRS-398 Rev. 1 Eq. 22 / Table 11)")
    if res["ks_warning"]:
        print(f"  ⚠️  WARNING:           {res['ks_warning']}")
    print("-" * 76)
    print("MEASUREMENTS & STATISTICS:")
    print(f"  Delivered:             {res['mu_delivered']:.0f} MU")
    print(f"  Raw Readings:          {res['raw_readings']} nC")
    print(f"  Mean Reading M_raw:    {res['m_raw_avg_nc']:.3f} nC (σ = {res['std_dev_nc']:.4f}, CV = {res['cv_pct']:.2f}%)")
    print(f"  Fully Corrected M_Q:   {res['m_q_nc']:.4f} nC")
    print("-" * 76)
    print("ABSORBED DOSE DETERMINATION TO WATER:")
    print(f"  D_w,Q(z_ref = 10 cm):  {res['d_w_zref_gy']:.4f} Gy  ({res['d_w_zref_cgy']:.2f} cGy)  -->  {res['dose_per_mu_zref_cgy']:.4f} cGy/MU")
    print(f"  D_w,Q(d_max = 1.5 cm): {res['d_w_zmax_gy']:.4f} Gy  ({res['d_w_zmax_cgy']:.2f} cGy)  -->  {res['dose_per_mu_zmax_cgy']:.4f} cGy/MU")
    print("-" * 76)
    print("CLINICAL COMPLIANCE & CALIBRATION:")
    print(f"  Nominal Target Output: {res['target_cgy_per_mu']:.4f} cGy/MU at d_max")
    print(f"  Beam Output Deviation: {sign}{res['diff_pct']:.2f}%")
    print(f"  Status:                {res['compliance']}")
    print(f"  Recommended Trim:      {trim_sign}{res['recommended_mu_trim_pct']:.2f}% linac MU potentiometer adjustment")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="IAEA TRS-398 Rev. 1 (2024) High-Energy Photon Reference Dosimetry Calculator (Flattened & FFF Beams)"
    )
    parser.add_argument("--preset", type=str, choices=list(PRESET_CONFIGS.keys()),
                        help="Load predefined clinical preset (e.g. TrueBeam_6MV_FFF, TrueBeam_10MV_FFF)")
    parser.add_argument("--readings", "-r", type=float, nargs="+", default=[12.34, 12.35, 12.35],
                        help="Electrometer raw readings in nC (default: 12.34 12.35 12.35)")
    parser.add_argument("--mu", type=float, default=100.0, help="Monitor units delivered (default: 100)")
    parser.add_argument("--tpr", type=float, default=None, help="Beam Quality Index TPR20,10")
    parser.add_argument("--chamber", type=str, default="30013", help="Chamber name or code (default: 30013 for PTW Farmer)")
    parser.add_argument("--ndw", type=float, default=0.05385, help="Calibration factor N_D,w in Gy/nC (default: 0.05385)")
    parser.add_argument("--temp", type=float, default=20.0, help="Ambient temperature in °C (default: 20.0)")
    parser.add_argument("--press", type=float, default=101.325, help="Ambient pressure in kPa (default: 101.325)")
    parser.add_argument("--pdd", type=float, default=None, help="PDD(10) in percent at SSD 100 cm")
    parser.add_argument("--fff", action="store_true", help="Enable Flattening Filter-Free (FFF) beam mode")
    parser.add_argument("--kvol", type=float, default=None, help="Volume averaging correction factor k_vol (for FFF)")
    parser.add_argument("--doserate", type=float, default=600.0, help="Dose rate in MU/min (default: 600 for W, 1400 for 6FFF)")
    parser.add_argument("--mplus", type=float, default=12.35, help="Reading at +bias voltage")
    parser.add_argument("--mminus", type=float, default=12.35, help="Reading at -bias voltage")
    parser.add_argument("--v1", type=float, default=300.0, help="Normal operating voltage (default: 300 V)")
    parser.add_argument("--v2", type=float, default=100.0, help="Reduced voltage (default: 100 V)")
    parser.add_argument("--m1", type=float, default=12.35, help="Reading at V1")
    parser.add_argument("--m2", type=float, default=12.28, help="Reading at V2")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")

    args = parser.parse_args()

    # Handle preset configuration
    tpr = args.tpr
    pdd = args.pdd
    is_fff = args.fff
    k_vol = args.kvol
    dose_rate = args.doserate
    chamber = args.chamber
    ndw = args.ndw

    if args.preset and args.preset in PRESET_CONFIGS:
        preset = PRESET_CONFIGS[args.preset]
        if tpr is None:
            tpr = preset["tpr2010"]
        if pdd is None:
            pdd = preset["pdd_zref"]
        is_fff = preset["is_fff"]
        if k_vol is None:
            k_vol = preset["k_vol"]
        dose_rate = preset["dose_rate"]
        chamber = preset["chamber"]
        ndw = preset["nd_w"]

    if tpr is None:
        tpr = 0.635 if is_fff else 0.670
    if pdd is None:
        pdd = 63.5 if is_fff else 67.0
    if is_fff and dose_rate == 600.0:
        dose_rate = 1400.0

    res = calc_trs398_dose(
        raw_readings=args.readings,
        mu_delivered=args.mu,
        nd_w=ndw,
        tpr2010=tpr,
        chamber_key=chamber,
        temp_c=args.temp,
        press_kpa=args.press,
        pdd_zref=pdd,
        m_plus=args.mplus,
        m_minus=args.mminus,
        v1=args.v1,
        v2=args.v2,
        m1=args.m1,
        m2=args.m2,
        is_fff=is_fff,
        k_vol=k_vol,
        dose_rate_mu_min=dose_rate
    )

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print_cli_report(res)


if __name__ == "__main__":
    main()
