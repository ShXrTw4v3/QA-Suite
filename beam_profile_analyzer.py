"""
beam_profile_analyzer.py — Radiotherapy Beam Profile Flatness, Symmetry & Penumbra Analyzer
==========================================================================================
Engine for analyzing flattened and flattening filter free (FFF) beam profiles
from water tanks, 1D/2D detector arrays, DICOM EPID images, and analytical reference models.

Protocols Supported:
- IEC 60976 (Flatness ratio Dmax/Dmin, PDQ Symmetry ratio)
- Varian (Difference Flatness, Point-Difference Symmetry)
- Siemens (Area / Integral Symmetry)
- FFF Beam Consensus (Inflection point derivative width, inflection-anchored penumbra,
  shoulder slopes, and point-difference / area symmetry)
"""

import io
import re
import json
import logging
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate, ndimage, optimize, integrate

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. ANALYTICAL SYNTHETIC REFERENCE PROFILES (Ground Truth Benchmarks)
# ===========================================================================

def generate_synthetic_profile(
    beam_type: str = "Flattened",
    energy_mv: Optional[float] = None,
    field_size_mm: float = 100.0,
    penumbra_mm: Optional[float] = None,
    horn_ratio: float = 0.0,
    cax_tilt: float = 0.0,
    step_size_mm: float = 0.5,
    range_mm: float = 150.0,
) -> Dict[str, Any]:
    """
    Generate an analytical reference beam profile with mathematically known ground truth.

    Supports presets:
    - '6 MV Flattened' (or 'Flattened')
    - '6 MV FFF'
    - '10 MV FFF'

    Uses a logistic sigmoid model for the physical penumbra step:
        S(x; x0, sigma) = 1 / (1 + exp((x - x0) / sigma))
    where sigma = P_80_20 / (4 * ln(2)).

    Parameters:
        beam_type: 'Flattened', 'FFF', '6 MV Flattened', '6 MV FFF', '10 MV FFF'
        energy_mv: Nominal photon energy in MV (defaults: 6.0 or 10.0 based on preset)
        field_size_mm: Collimated field width (default 100.0 mm)
        penumbra_mm: Target 80%-20% penumbra width (default: 6.0 mm for 6 MV, 5.5 mm for 10 MV)
        horn_ratio: Flatness horn amplitude h (e.g. 0.0 for flat, 0.025 for 2.5% horns)
        cax_tilt: Asymmetry tilt factor alpha (e.g. 0.0 for symmetric, 0.025 for tilted)
        step_size_mm: Spatial resolution in mm (default 0.5 mm)
        range_mm: Half-range of calculation in mm (default +/- 150 mm)

    Returns:
        Dict with 'x_mm', 'values', 'meta', and 'ground_truth'.
    """
    bt_upper = str(beam_type).upper()
    is_fff = "FFF" in bt_upper

    # Preset autodetection
    if energy_mv is None:
        if "10" in bt_upper:
            energy_mv = 10.0
        else:
            energy_mv = 6.0

    if penumbra_mm is None:
        if energy_mv >= 9.0:
            penumbra_mm = 5.5
        else:
            penumbra_mm = 6.0

    half_w = field_size_mm / 2.0
    sigma = penumbra_mm / (4.0 * np.log(2.0))
    x = np.arange(-range_mm, range_mm + step_size_mm, step_size_mm)

    # Sigmoidal edge transmission: Left rising step * Right falling step
    left_step = 1.0 / (1.0 + np.exp(-np.clip((x + half_w) / sigma, -100.0, 100.0)))
    right_step = 1.0 / (1.0 + np.exp(np.clip((x - half_w) / sigma, -100.0, 100.0)))
    raw_plateau = left_step * right_step

    if not is_fff:
        # Flattened beam:
        # Guarantees exact 100.00% flatness for h=0 across inner 80% (|x| <= 0.40 W)
        # while preserving true 50% crossing at |x| = half_w:
        x_eval_bound = 0.40 * field_size_mm
        plateau = np.where(np.abs(x) <= x_eval_bound, 1.0, raw_plateau)

        if horn_ratio > 0.0:
            horn_curve = 1.0 + horn_ratio * (x / max(x_eval_bound, 1.0)) ** 2
            profile = np.where(np.abs(x) <= x_eval_bound, horn_curve, plateau * (1.0 + horn_ratio))
        else:
            profile = plateau
    else:
        # FFF beam: forward-peaked bremsstrahlung conical roll-off
        # Attenuation constant: ~0.0035 /mm for 6 MV FFF, ~0.0065 /mm for 10 MV FFF
        if energy_mv >= 9.0:
            lambda_att = 0.0065
        else:
            lambda_att = 0.0035

        conical_peak = np.exp(-lambda_att * np.abs(x))
        profile = conical_peak * raw_plateau

    # Apply asymmetry tilt if requested: D_tilt(x) = D(x) * (1 + tilt * x / half_w)
    if cax_tilt != 0.0:
        profile = profile * (1.0 + cax_tilt * (x / max(half_w, 1.0)))

    # Normalize central axis dose to 100.0%
    cax_idx = np.argmin(np.abs(x))
    cax_val = profile[cax_idx]
    if cax_val > 0:
        values = (profile / cax_val) * 100.0
    else:
        values = profile * 100.0

    # Ground truth calculations
    eval_mask = np.abs(x) <= 0.40 * field_size_mm
    eval_vals = values[eval_mask]
    gt_d_max = float(np.max(eval_vals))
    gt_d_min = float(np.min(eval_vals))

    if not is_fff:
        gt_flatness_iec = float(100.0 * (gt_d_max / max(gt_d_min, 1e-4)))
        gt_flatness_varian = float(100.0 * (gt_d_max - gt_d_min) / max(gt_d_max + gt_d_min, 1e-4))
    else:
        gt_flatness_iec = None
        gt_flatness_varian = None

    meta = {
        "data_source": "Synthetic Reference",
        "beam_type": "FFF" if is_fff else "Flattened",
        "energy": f"{int(energy_mv)} MV{' FFF' if is_fff else ''}",
        "field_size_nominal": f"{int(field_size_mm/10)}x{int(field_size_mm/10)} cm",
        "axis": "Crossplane (X)",
        "step_size_mm": step_size_mm,
    }

    ground_truth = {
        "field_width_mm": field_size_mm,
        "penumbra_mm": penumbra_mm,
        "flatness_varian_pct": round(gt_flatness_varian, 3) if gt_flatness_varian is not None else None,
        "flatness_iec_pct": round(gt_flatness_iec, 3) if gt_flatness_iec is not None else None,
        "symmetry_pct": 0.0 if cax_tilt == 0.0 else round(cax_tilt * 100.0 * 1.6, 2),
    }

    return {
        "x_mm": x,
        "values": values,
        "meta": meta,
        "ground_truth": ground_truth,
    }


# ===========================================================================
# 2. UNIVERSAL 1D/2D TEXT & ASCII PROFILE PARSER
# ===========================================================================

def _try_parse_ptw_mcc_scans(content: str) -> Optional[Dict[str, Any]]:
    """
    Parses multi-scan PTW Mephysto / BeamScan MCC format files containing
    BEGIN_SCAN ... END_SCAN blocks (e.g. Inplane, Crossplane, and PDD curves).
    """
    scan_blocks = re.findall(r"BEGIN_SCAN\s+\d+(.*?)(?:END_SCAN|$)", content, re.DOTALL | re.IGNORECASE)
    if not scan_blocks:
        return None

    profiles = {}
    primary = None

    for i, sc in enumerate(scan_blocks, 1):
        m_type = re.search(r"SCAN_CURVETYPE\s*=\s*([^\r\n]+)", sc, re.IGNORECASE)
        m_depth = re.search(r"SCAN_DEPTH\s*=\s*([^\r\n]+)", sc, re.IGNORECASE)
        m_field_x = re.search(r"FIELD_CROSSPLANE\s*=\s*([^\r\n]+)", sc, re.IGNORECASE)
        m_field_y = re.search(r"FIELD_INPLANE\s*=\s*([^\r\n]+)", sc, re.IGNORECASE)
        m_energy = re.search(r"ENERGY\s*=\s*([^\r\n]+)", sc, re.IGNORECASE)
        m_filter = re.search(r"FILTER\s*=\s*([^\r\n]+)", sc, re.IGNORECASE)
        m_linac = re.search(r"LINAC\s*=\s*([^\r\n]+)", sc, re.IGNORECASE)

        ctype = m_type.group(1).strip() if m_type else f"Scan_{i}"
        try:
            depth = float(m_depth.group(1).strip()) if m_depth else 0.0
        except ValueError:
            depth = 0.0
        try:
            fx = float(m_field_x.group(1).strip()) if m_field_x else 100.0
        except ValueError:
            fx = 100.0
        try:
            fy = float(m_field_y.group(1).strip()) if m_field_y else 100.0
        except ValueError:
            fy = 100.0

        energy_str = m_energy.group(1).strip() if m_energy else "6.0"
        filt = m_filter.group(1).strip().upper() if m_filter else "FF"
        linac = m_linac.group(1).strip() if m_linac else "Linac"

        is_fff = (filt == "FFF") or ("FFF" in energy_str.upper())
        beam_type = "FFF" if is_fff else "Flattened"

        data_match = re.search(r"BEGIN_DATA\s*(.*?)\s*END_DATA", sc, re.DOTALL | re.IGNORECASE)
        if not data_match:
            continue

        rows = []
        for line in data_match.group(1).strip().splitlines():
            tokens = re.split(r"[\t,;]+|\s+", line.strip())
            if len(tokens) >= 2:
                try:
                    rows.append([float(tokens[0]), float(tokens[1])])
                except ValueError:
                    pass
        if len(rows) < 3:
            continue

        arr = np.array(rows)
        sort_idx = np.argsort(arr[:, 0])
        x_mm = arr[sort_idx, 0]
        y_vals = arr[sort_idx, 1]

        # Determine axis label
        upper_type = ctype.upper()
        if "CROSS" in upper_type:
            axis_name = "Crossplane (X)"
        elif "INPLANE" in upper_type:
            axis_name = "Inplane (Y)"
        elif "PDD" in upper_type:
            axis_name = "PDD (Z)"
        else:
            axis_name = ctype.replace("_PROFILE", "").replace("_", " ").title()

        label = f"{axis_name} - {fx/10:.0f}x{fy/10:.0f}cm d={depth:.0f}mm ({filt})"
        meta = {
            "data_source": "PTW BeamScan MCC",
            "beam_type": beam_type,
            "axis": axis_name,
            "field_size_nominal": f"{fx/10:.0f}x{fy/10:.0f} cm",
            "energy": f"{energy_str} MV {filt}",
            "unit_name": linac,
            "depth_mm": str(depth),
            "unit": "mm",
            "is_fff": is_fff,
        }
        prof_entry = {"x_mm": x_mm, "values": y_vals, "meta": meta}
        profiles[label] = prof_entry

        if primary is None:
            primary = prof_entry
        elif "Crossplane" in axis_name:
            primary = prof_entry
        elif "Inplane" in axis_name and "Crossplane" not in primary["meta"]["axis"]:
            primary = prof_entry

    if not profiles or primary is None:
        return None

    return {
        "x_mm": primary["x_mm"],
        "values": primary["values"],
        "meta": primary["meta"],
        "profiles": profiles,
    }


def _try_parse_varian_w2cad(content: str) -> Optional[Dict[str, Any]]:
    """
    Parse Varian Eclipse W2CAD beam data export files ($STOM ... $ENOM / $ENOF).
    Extracts OPD (Open Field Depth Dose / PDD) and OPP (Open Field Profile) curves.
    """
    lines = content.splitlines()
    global_meta: Dict[str, str] = {}
    for l in lines[:30]:
        l_str = l.strip()
        if l_str.startswith("#"):
            if "machine:" in l_str:
                global_meta["machine"] = l_str.split("machine:")[1].strip()
            elif "algorithm:" in l_str:
                global_meta["algorithm"] = l_str.split("algorithm:")[1].strip()
            elif "beam data:" in l_str:
                global_meta["beam_data"] = l_str.split("beam data:")[1].strip()
        elif l_str.startswith("%") and not l_str.startswith("%TYPE"):
            parts = l_str.split(None, 1)
            if len(parts) == 2:
                global_meta[parts[0][1:].upper()] = parts[1]

    profiles = {}
    primary = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("%TYPE"):
            type_val = line.split()[1].upper()
            meta = dict(global_meta)
            meta["TYPE"] = type_val
            j = i + 1
            pts = []
            while j < len(lines):
                l = lines[j].strip()
                if l.startswith("%"):
                    parts = l.split(None, 1)
                    if len(parts) == 2:
                        meta[parts[0][1:].upper()] = parts[1]
                elif l.startswith("<") and l.endswith(">"):
                    nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", l)]
                    if len(nums) >= 4:
                        pts.append(nums[:4])
                elif l.startswith("$ENOM") or (l.startswith("%TYPE") and pts):
                    break
                j += 1

            if pts:
                arr = np.array(pts)
                is_pdd = (type_val == "OPD") or ("DEPTH" in meta.get("machine", "").upper())

                if is_pdd:
                    # Column 2 (Z) is depth in mm, Column 3 is relative dose (%)
                    coords = arr[:, 2]
                    doses = arr[:, 3]
                    axis_name = "PDD (Z)"
                else:
                    # Column 0 (X) is off-axis position in mm, Column 3 is relative dose (%)
                    var_x = float(np.ptp(arr[:, 0]))
                    var_y = float(np.ptp(arr[:, 1]))
                    if var_y > var_x:
                        coords = arr[:, 1]
                        axis_name = "Inplane (Y)"
                    else:
                        coords = arr[:, 0]
                        axis_name = "Crossplane (X)"
                    doses = arr[:, 3]

                sort_idx = np.argsort(coords)
                x_mm = coords[sort_idx]
                y_vals = doses[sort_idx]

                flsz_str = meta.get("FLSZ", "100*100").replace("*", "x")
                dpth_val = meta.get("DPTH", "0")
                energy_raw = meta.get("ENERGY", "6.0")
                mach_str = meta.get("machine", "")

                is_fff = False
                if "10X-FFF" in mach_str.upper() or "10FFF" in mach_str.upper():
                    energy_label = "10 MV FFF"
                    is_fff = True
                elif "6X-FFF" in mach_str.upper() or "6FFF" in mach_str.upper():
                    energy_label = "6 MV FFF"
                    is_fff = True
                elif "15X" in mach_str.upper() or "15" in energy_raw:
                    energy_label = "15 MV"
                elif "10X" in mach_str.upper() or "10" in energy_raw:
                    energy_label = "10 MV"
                elif "6X" in mach_str.upper() or "6" in energy_raw:
                    energy_label = "6 MV"
                else:
                    try:
                        energy_num = float(energy_raw)
                        energy_label = f"{energy_num:.0f} MV"
                    except Exception:
                        energy_label = f"{energy_raw} MV"
                    if "FFF" in mach_str.upper() or "FFF" in energy_raw.upper():
                        is_fff = True
                        if "FFF" not in energy_label:
                            energy_label += " FFF"

                label = f"{axis_name} - {flsz_str} d={dpth_val}mm ({energy_label})"
                scan_meta = {
                    "data_source": "Varian Eclipse W2CAD",
                    "beam_type": "FFF" if is_fff else "Flattened",
                    "axis": axis_name,
                    "field_size_nominal": f"{flsz_str} mm",
                    "energy": energy_label,
                    "unit_name": mach_str or "Varian TrueBeam",
                    "depth_mm": str(dpth_val),
                    "unit": "mm",
                    "is_fff": is_fff,
                    "is_pdd": is_pdd,
                }
                prof_entry = {"x_mm": x_mm, "values": y_vals, "meta": scan_meta}
                profiles[label] = prof_entry

                if primary is None:
                    primary = prof_entry
                elif "100x100" in flsz_str:
                    primary = prof_entry
            i = j - 1
        i += 1

    if not profiles or primary is None:
        return None

    return {
        "x_mm": primary["x_mm"],
        "values": primary["values"],
        "meta": primary["meta"],
        "profiles": profiles,
    }


def parse_profile_text(content: Union[str, bytes]) -> Dict[str, Any]:
    """
    Universally parse 1D and 2D beam profile exports from ASCII, CSV, TSV,
    IBA OmniPro (<BEGIN_DATA>), PTW Mephysto ([SCAN] / BEGIN_DATA / BEGIN_SCAN),
    and Sun Nuclear SNC Profiler formats.

    Normalizes coordinate spacing to millimeters and doses to percentages.

    Returns:
        Dict:
            'x_mm': np.ndarray of coordinates (sorted ascending)
            'values': np.ndarray of doses
            'meta': dict of parsed header metadata
            'profiles': dict of sub-profiles if multi-axis (e.g. crossplane, inplane)
    """
    if isinstance(content, bytes):
        for enc in ("utf-8", "latin1", "cp1252", "iso-8859-1"):
            try:
                content = content.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

    # Check for PTW Mephysto / BeamScan MCC multi-scan format
    if "BEGIN_SCAN" in content.upper() and "BEGIN_DATA" in content.upper():
        mcc_res = _try_parse_ptw_mcc_scans(content)
        if mcc_res is not None:
            return mcc_res

    # Check for Varian Eclipse W2CAD format ($STOM ... %TYPE OPD / %TYPE OPP ... <X Y Z Dose>)
    if any(marker in content.upper() for marker in ("$STOM", "%TYPE OPD", "%TYPE OPP", "$NUMS")):
        w2cad_res = _try_parse_varian_w2cad(content)
        if w2cad_res is not None:
            return w2cad_res

    lines = [ln.strip() for ln in content.splitlines()]

    meta: Dict[str, Any] = {
        "data_source": "ASCII / CSV Export",
        "beam_type": "Flattened",
        "axis": "Crossplane (X)",
        "field_size_nominal": "10x10 cm",
        "energy": "6 MV",
    }

    # Harvest metadata key-value pairs from header comments
    in_data_block = False
    data_lines = []
    has_explicit_data_block = False

    for line in lines:
        if not line:
            continue

        # Detect block markers
        upper_line = line.upper()
        if any(marker in upper_line for marker in ("<BEGIN_DATA>", "BEGIN_DATA", "$DATA")):
            in_data_block = True
            has_explicit_data_block = True
            continue
        if any(marker in upper_line for marker in ("<END_DATA>", "END_DATA")):
            in_data_block = False
            continue

        # Extract metadata outside or inside headers
        if not in_data_block and (line.startswith(("#", "%", "!", "*", "//", "[")) or "=" in line or ":" in line):
            # Parse key-values
            for delim in ("=", ":"):
                if delim in line:
                    parts = line.lstrip("#%!*/[ ").rstrip("]").split(delim, 1)
                    if len(parts) == 2:
                        k = parts[0].strip().upper().replace(" ", "_")
                        v = parts[1].strip()
                        if "ENERGY" in k:
                            meta["energy"] = v
                            if "FFF" in v.upper():
                                meta["beam_type"] = "FFF"
                        elif "FIELD" in k or "FIELD_SIZE" in k:
                            meta["field_size_nominal"] = v
                        elif "AXIS" in k or "SCAN_AXIS" in k:
                            if "Y" in v.upper() or "INPLANE" in v.upper():
                                meta["axis"] = "Inplane (Y)"
                            else:
                                meta["axis"] = "Crossplane (X)"
                        elif "MACHINE" in k or "DEVICE" in k:
                            meta["unit_name"] = v
                        elif "UNIT" in k or "UNITS" in k:
                            meta["unit"] = v.strip().lower()
                        elif k in ("MM", "MILLIMETER", "MILLIMETERS"):
                            meta["unit"] = "mm"
                        elif k in ("CM", "CENTIMETER", "CENTIMETERS"):
                            meta["unit"] = "cm"
                        elif "DEPTH" in k:
                            meta["depth_mm"] = v
            continue

        # Collect data lines
        if has_explicit_data_block:
            if in_data_block:
                data_lines.append(line)
        else:
            # If line starts with a number or minus sign, treat as data candidate
            if re.match(r"^[-+]?[0-9]*\.?[0-9]+", line):
                data_lines.append(line)

    if not data_lines:
        raise ValueError("No numeric profile data found in file content.")

    # Parse numeric tokens per line
    raw_rows = []
    for ln in data_lines:
        tokens = re.split(r"[\t,;]+|\s+", ln.strip())
        numeric = []
        for t in tokens:
            try:
                numeric.append(float(t))
            except ValueError:
                pass
        if len(numeric) >= 2:
            raw_rows.append(numeric)

    if not raw_rows:
        raise ValueError("Failed to extract valid numeric coordinate-dose pairs.")

    arr = np.array(raw_rows)
    ncols = arr.shape[1]

    # Column mapping logic:
    # 2 columns: Col 0 = Position, Col 1 = Dose
    # 3 columns:
    #   Case A: Col 0 is 1-based index (1, 2, 3...) -> Col 1 is Position, Col 2 is Dose
    #   Case B: Col 0 is Position, Col 1 is Crossplane, Col 2 is Inplane
    # >= 4 columns: Col 0 is Position, Col 1 is Dose (or Col 1 is Pos, Col 2 is Dose)
    multi_profiles = {}
    if ncols == 2:
        x_raw = arr[:, 0]
        y_raw = arr[:, 1]
    elif ncols == 3:
        # Check if Col 0 is sequential integer index
        diffs = np.diff(arr[:, 0])
        is_index = np.all(np.abs(diffs - 1.0) < 1e-4) and (arr[0, 0] in (0.0, 1.0))
        if is_index:
            x_raw = arr[:, 1]
            y_raw = arr[:, 2]
        else:
            x_raw = arr[:, 0]
            y_raw = arr[:, 1]
            multi_profiles["crossplane"] = {"x_mm": x_raw, "values": arr[:, 1]}
            multi_profiles["inplane"] = {"x_mm": x_raw, "values": arr[:, 2]}
    else:
        # Check first column vs second column
        if arr[0, 0] in (0.0, 1.0) and np.all(np.abs(np.diff(arr[:, 0]) - 1.0) < 1e-4):
            x_raw = arr[:, 1]
            y_raw = arr[:, 2]
        else:
            x_raw = arr[:, 0]
            y_raw = arr[:, 1]

    # Coordinate unit check:
    # If spatial coordinate range is small (e.g. max(|x|) <= 25.0) and nominal field is cm-sized,
    # coordinates were exported in cm -> convert to mm.
    # If unit is explicitly declared as mm / millimeter, do NOT scale by 10.
    declared_unit = str(meta.get("unit", "")).lower().strip()
    is_declared_mm = ("mm" in declared_unit) or ("millimeter" in declared_unit)
    max_abs_x = float(np.max(np.abs(x_raw)))
    if not is_declared_mm and max_abs_x <= 25.0 and max_abs_x > 0.0:
        x_raw = x_raw * 10.0

    # Clean, sort by x ascending, and drop duplicates
    df = pd.DataFrame({"x": x_raw, "y": y_raw})
    df = df.dropna().drop_duplicates(subset=["x"]).sort_values("x")
    
    x_clean = df["x"].to_numpy()
    y_clean = df["y"].to_numpy()

    # Normalize multi-profiles if present
    cleaned_multi = {}
    for prof_name, pdata in multi_profiles.items():
        px = pdata["x_mm"]
        if not is_declared_mm and max_abs_x <= 25.0 and max_abs_x > 0.0:
            px = px * 10.0
        pdf = pd.DataFrame({"x": px, "y": pdata["values"]}).dropna().drop_duplicates(subset=["x"]).sort_values("x")
        cleaned_multi[prof_name] = {"x_mm": pdf["x"].to_numpy(), "values": pdf["y"].to_numpy()}

    return {
        "x_mm": x_clean,
        "values": y_clean,
        "meta": meta,
        "profiles": cleaned_multi if cleaned_multi else {"primary": {"x_mm": x_clean, "values": y_clean}},
    }


def parse_profile_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Parse profile from file path."""
    with open(file_path, "rb") as f:
        return parse_profile_text(f.read())


# ===========================================================================
# 3. DICOM EPID 2D IMAGE PARSER & AXIS PROFILER
# ===========================================================================

def parse_dicom_epid(dcm_path_or_bytes: Union[str, Path, bytes, io.BytesIO]) -> Dict[str, Any]:
    """
    Parse a DICOM RT Image / EPID file (.dcm), extracting 2D pixel matrices
    and central crossplane (horizontal) and inplane (vertical) profiles.

    Performs:
    1. Zero-PHI metadata extraction (no patient identifiers stored).
    2. Imager pixel spacing reading and SID/SAD magnification scaling to isocenter.
    3. Rescale slope and intercept calibration.
    4. EPID negative-image inversion check.
    5. Multi-pixel lateral strip averaging (5 mm wide strip) to suppress quantum noise.

    Returns:
        Dict:
            'crossplane': {'x_mm': np.ndarray, 'values': np.ndarray}
            'inplane': {'x_mm': np.ndarray, 'values': np.ndarray}
            'meta': dict of linac, energy, spacing metadata
            'image_2d': 2D np.ndarray (calibrated, oriented)
            'pixel_spacing_iso': (dy_iso, dx_iso) in mm
    """
    import pydicom

    if isinstance(dcm_path_or_bytes, (str, Path)):
        ds = pydicom.dcmread(str(dcm_path_or_bytes), force=True)
    elif isinstance(dcm_path_or_bytes, bytes):
        ds = pydicom.dcmread(io.BytesIO(dcm_path_or_bytes), force=True)
    else:
        ds = pydicom.dcmread(dcm_path_or_bytes, force=True)

    # Pixel Spacing at Imager Plane
    if hasattr(ds, "ImagePlanePixelSpacing"):
        imager_spacing = [float(s) for s in ds.ImagePlanePixelSpacing]
    elif hasattr(ds, "ImagerPixelSpacing"):
        imager_spacing = [float(s) for s in ds.ImagerPixelSpacing]
    elif hasattr(ds, "PixelSpacing"):
        imager_spacing = [float(s) for s in ds.PixelSpacing]
    else:
        imager_spacing = [0.392, 0.392]  # Default standard EPID pixel pitch ~0.392 mm

    dy_imager, dx_imager = imager_spacing[0], imager_spacing[1]

    # Magnification correction: SID / SAD
    sid = float(getattr(ds, "RTImageSID", 1500.0) or 1500.0)
    sad = float(getattr(ds, "RadiationMachineSAD", 1000.0) or 1000.0)
    mag = max(sid / max(sad, 1.0), 0.5)

    # Pixel spacing scaled to isocenter
    dx_iso = dx_imager / mag
    dy_iso = dy_imager / mag

    # Pixel array calibration
    pixel_arr = ds.pixel_array.astype(float)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    img_cal = pixel_arr * slope + intercept

    # Inversion check: if corners/edges have higher signal than central region, invert
    h, w = img_cal.shape
    corner_box = 10
    corner_mean = (
        np.mean(img_cal[:corner_box, :corner_box]) +
        np.mean(img_cal[:corner_box, -corner_box:]) +
        np.mean(img_cal[-corner_box:, :corner_box]) +
        np.mean(img_cal[-corner_box:, -corner_box:])
    ) / 4.0
    center_mean = np.mean(img_cal[h//4:3*h//4, w//4:3*w//4])

    if corner_mean > center_mean:
        img_cal = np.max(img_cal) - img_cal

    # Determine center of field
    # Initial estimate: array center
    cy_idx = h // 2
    cx_idx = w // 2

    # Multi-pixel lateral strip averaging: 5 mm strip width
    strip_half_x = max(1, int(round(2.5 / dy_iso)))
    strip_half_y = max(1, int(round(2.5 / dx_iso)))

    # Crossplane (Horizontal cut along row axis cy_idx)
    y_min_strip = max(0, cy_idx - strip_half_x)
    y_max_strip = min(h, cy_idx + strip_half_x + 1)
    crossplane_vals = np.mean(img_cal[y_min_strip:y_max_strip, :], axis=0)
    x_coords = (np.arange(w) - cx_idx) * dx_iso

    # Inplane (Vertical cut along col axis cx_idx)
    x_min_strip = max(0, cx_idx - strip_half_y)
    x_max_strip = min(w, cx_idx + strip_half_y + 1)
    inplane_vals = np.mean(img_cal[:, x_min_strip:x_max_strip], axis=1)
    y_coords = (np.arange(h) - cy_idx) * dy_iso

    # Normalize to CAX = 100%
    if crossplane_vals[cx_idx] > 0:
        crossplane_vals = (crossplane_vals / crossplane_vals[cx_idx]) * 100.0
    if inplane_vals[cy_idx] > 0:
        inplane_vals = (inplane_vals / inplane_vals[cy_idx]) * 100.0

    # Metadata extraction (Strictly ZERO-PHI: never record patient name or ID)
    beam_energy_val = getattr(ds, "NominalBeamEnergy", None)
    energy_str = f"{beam_energy_val} MV" if beam_energy_val else "6 MV"
    machine_name = str(getattr(ds, "RadiationMachineName", "") or "Varian TrueBeam")

    meta = {
        "data_source": "DICOM EPID (.dcm)",
        "unit_name": machine_name,
        "energy": energy_str,
        "beam_type": "FFF" if "FFF" in energy_str.upper() else "Flattened",
        "sid_mm": sid,
        "sad_mm": sad,
        "magnification": round(mag, 4),
        "pixel_spacing_iso_mm": (round(dy_iso, 4), round(dx_iso, 4)),
        "image_shape": (h, w),
    }

    return {
        "crossplane": {"x_mm": x_coords, "values": crossplane_vals},
        "inplane": {"x_mm": y_coords, "values": inplane_vals},
        "meta": meta,
        "image_2d": img_cal,
        "pixel_spacing_iso": (dy_iso, dx_iso),
    }


# ===========================================================================
# 4. FLATTENED BEAM PROFILE ANALYSIS (IEC 60976, Varian, Siemens)
# ===========================================================================

def analyze_flattened_profile(
    x_mm: Union[np.ndarray, list],
    values: Union[np.ndarray, list],
    protocol: str = "IEC",
    cax_pos: Optional[float] = None,
    fwhm_threshold: float = 50.0,
    penumbra_low: float = 20.0,
    penumbra_high: float = 80.0,
) -> Dict[str, Any]:
    """
    Rigorously evaluate a flattened photon beam profile according to IEC 60976,
    Varian, and Siemens standards.

    Parameters:
        x_mm: 1D array of spatial positions in mm.
        values: 1D array of dose values.
        protocol: 'IEC' (IEC 60976 Dmax/Dmin, PDQ) or 'Varian' (Difference Flatness, Point-Diff)
        cax_pos: Position of central axis in mm (defaults to 0.0 or closest point)
        fwhm_threshold: Percentage dose defining field edges (default 50.0%)
        penumbra_low: Lower penumbra boundary (default 20.0%)
        penumbra_high: Upper penumbra boundary (default 80.0%)

    Returns:
        Dict of results including:
        - fwhm_mm: Measured field width at 50%
        - cax_offset_mm: Center offset (x_L + x_R) / 2
        - penumbra_left_mm: 80-20% left penumbra
        - penumbra_right_mm: 80-20% right penumbra
        - penumbra_mean_mm: Average penumbra
        - flatness_pct: Protocol evaluated flatness
        - flatness_iec_pct: IEC 60976 Flatness (Dmax/Dmin * 100)
        - flatness_varian_pct: Varian Flatness ((Dmax - Dmin) / (Dmax + Dmin) * 100)
        - symmetry_pct: Protocol evaluated symmetry
        - symmetry_varian_pct: Maximum point-difference symmetry
        - symmetry_iec_pdq: IEC PDQ ratio symmetry
        - symmetry_area_pct: Siemens Area symmetry
        - status: 'OK', 'Tolerance', or 'Action'
    """
    x = np.asarray(x_mm, dtype=float)
    y = np.asarray(values, dtype=float)

    # Sort ascending and clean
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Normalize to central axis (x = 0 or cax_pos)
    if cax_pos is None:
        cax_pos = 0.0
    
    # Interpolate dose at CAX
    f_interp = interpolate.interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    d_cax = float(f_interp(cax_pos))
    if d_cax <= 0:
        d_cax = float(np.max(y))
    
    y_norm = (y / d_cax) * 100.0

    # Find 50% field edge crossings (x_L and x_R)
    # Left edge: crossing for x < cax_pos
    left_mask = x <= cax_pos
    right_mask = x >= cax_pos

    x_left_half = x[left_mask]
    y_left_half = y_norm[left_mask]
    x_right_half = x[right_mask]
    y_right_half = y_norm[right_mask]

    # Continuous inverse interpolation for exact crossings
    def find_crossing(x_arr, y_arr, target_val):
        # Monotonic search around target_val
        diffs = y_arr - target_val
        signs = np.sign(diffs)
        sign_changes = np.where(np.diff(signs) != 0)[0]
        if len(sign_changes) == 0:
            # Fallback to closest point
            return float(x_arr[np.argmin(np.abs(diffs))])
        idx = sign_changes[0]
        x0, x1 = x_arr[idx], x_arr[idx + 1]
        y0, y1 = y_arr[idx], y_arr[idx + 1]
        if y1 == y0:
            return float(x0)
        return float(x0 + (target_val - y0) * (x1 - x0) / (y1 - y0))

    # Field edges at 50%
    x_l_50 = find_crossing(x_left_half, y_left_half, fwhm_threshold)
    x_r_50 = find_crossing(x_right_half, y_right_half, fwhm_threshold)
    fwhm_mm = float(x_r_50 - x_l_50)
    cax_offset_mm = float((x_l_50 + x_r_50) / 2.0)

    # Penumbra crossings: 20% and 80%
    x_l_20 = find_crossing(x_left_half, y_left_half, penumbra_low)
    x_l_80 = find_crossing(x_left_half, y_left_half, penumbra_high)
    penumbra_left_mm = float(np.abs(x_l_80 - x_l_20))

    x_r_80 = find_crossing(x_right_half, y_right_half, penumbra_high)
    x_r_20 = find_crossing(x_right_half, y_right_half, penumbra_low)
    penumbra_right_mm = float(np.abs(x_r_20 - x_r_80))
    penumbra_mean_mm = float((penumbra_left_mm + penumbra_right_mm) / 2.0)

    # Flattened evaluation region: inner 80% of field width (|x - cax_offset| <= 0.4 * FWHM)
    eval_half_w = 0.40 * fwhm_mm
    eval_left = cax_offset_mm - eval_half_w
    eval_right = cax_offset_mm + eval_half_w

    eval_mask = (x >= eval_left) & (x <= eval_right)
    if np.sum(eval_mask) < 3:
        # Fallback to dense interpolated evaluation
        eval_x = np.linspace(eval_left, eval_right, 101)
        eval_y = f_interp(eval_x) / d_cax * 100.0
    else:
        eval_x = x[eval_mask]
        eval_y = y_norm[eval_mask]

    d_max = float(np.max(eval_y))
    d_min = float(np.min(eval_y))

    # 1. IEC 60976 Flatness: 100 * (Dmax / Dmin)
    flatness_iec_pct = float(100.0 * (d_max / max(d_min, 1e-4)))

    # 2. Varian Difference Flatness: 100 * (Dmax - Dmin) / (Dmax + Dmin)
    flatness_varian_pct = float(100.0 * (d_max - d_min) / max(d_max + d_min, 1e-4))

    # Symmetry calculations across [0, eval_half_w]
    sym_sample_xs = np.linspace(0.0, eval_half_w, 101)
    # Evaluate at +xs and -xs relative to cax_offset
    d_pos = f_interp(cax_offset_mm + sym_sample_xs) / d_cax * 100.0
    d_neg = f_interp(cax_offset_mm - sym_sample_xs) / d_cax * 100.0

    # Varian Point-Difference Symmetry: 100 * max(|D(x) - D(-x)|) / D_cax
    point_diffs = d_pos - d_neg
    max_idx = np.argmax(np.abs(point_diffs))
    symmetry_varian_pct = float(point_diffs[max_idx])

    # IEC 60976 PDQ Ratio Symmetry: max(D(x)/D(-x), D(-x)/D(x))
    ratios = np.maximum(d_pos / np.maximum(d_neg, 1e-4), d_neg / np.maximum(d_pos, 1e-4))
    symmetry_iec_pdq = float(np.max(ratios))

    # Siemens Area Symmetry: 100 * (Area_L - Area_R) / (Area_L + Area_R)
    # Area_L from eval_left to cax_offset, Area_R from cax_offset to eval_right
    mask_l = (eval_x <= cax_offset_mm)
    mask_r = (eval_x >= cax_offset_mm)
    if np.sum(mask_l) >= 2 and np.sum(mask_r) >= 2:
        area_l = float(integrate.trapezoid(eval_y[mask_l], eval_x[mask_l]))
        area_r = float(integrate.trapezoid(eval_y[mask_r], eval_x[mask_r]))
        symmetry_area_pct = float(100.0 * (area_l - area_r) / max(area_l + area_r, 1e-4))
    else:
        symmetry_area_pct = 0.0

    # Select primary metrics based on protocol
    proto_upper = str(protocol).upper()
    if "VARIAN" in proto_upper:
        flatness_pct = flatness_varian_pct
        symmetry_pct = symmetry_varian_pct
    else:  # IEC / Elekta default
        flatness_pct = flatness_iec_pct
        # For IEC symmetry report as ratio (e.g. 1.012 or (ratio-1)*100)
        symmetry_pct = symmetry_iec_pdq

    # Evaluate Clinical Status
    # Standard TG-142 / IEC limits:
    # Flatness (Varian): OK <= 1.5%, Tol <= 2.5%, Action > 2.5%
    # Flatness (IEC): OK <= 103.0%, Tol <= 104.0%, Action > 104.0%
    # Symmetry (Varian): |S| <= 1.5% OK, <= 2.5% Tol, > 2.5% Action
    # Symmetry (IEC): ratio <= 1.020 OK, <= 1.030 Tol, > 1.030 Action
    status = "OK"
    if np.isnan(fwhm_mm) or np.isnan(flatness_pct) or np.isnan(symmetry_pct):
        status = "Invalid"
    elif "VARIAN" in proto_upper:
        if abs(flatness_varian_pct) > 2.5 or abs(symmetry_varian_pct) > 2.5:
            status = "Action"
        elif abs(flatness_varian_pct) > 1.5 or abs(symmetry_varian_pct) > 1.5:
            status = "Tolerance"
    else:
        if flatness_iec_pct > 104.0 or symmetry_iec_pdq > 1.030:
            status = "Action"
        elif flatness_iec_pct > 103.0 or symmetry_iec_pdq > 1.020:
            status = "Tolerance"

    return {
        "protocol": protocol,
        "beam_type": "Flattened",
        "fwhm_mm": round(fwhm_mm, 3),
        "field_width_mm": round(fwhm_mm, 3),
        "cax_offset_mm": round(cax_offset_mm, 3),
        "penumbra_left_mm": round(penumbra_left_mm, 3),
        "penumbra_right_mm": round(penumbra_right_mm, 3),
        "penumbra_mean_mm": round(penumbra_mean_mm, 3),
        "flatness_pct": round(flatness_pct, 3),
        "flatness_iec_pct": round(flatness_iec_pct, 3),
        "flatness_varian_pct": round(flatness_varian_pct, 3),
        "symmetry_pct": round(symmetry_pct, 3),
        "symmetry_varian_pct": round(symmetry_varian_pct, 3),
        "symmetry_iec_pdq": round(symmetry_iec_pdq, 4),
        "symmetry_area_pct": round(symmetry_area_pct, 3),
        "d_max": round(d_max, 3),
        "d_min": round(d_min, 3),
        "status": status,
        "edges": {"left": round(x_l_50, 3), "right": round(x_r_50, 3)},
        "penumbra_points": {
            "left_20": round(x_l_20, 3),
            "left_80": round(x_l_80, 3),
            "right_80": round(x_r_80, 3),
            "right_20": round(x_r_20, 3),
        },
        "eval_region": {"left": round(eval_left, 3), "right": round(eval_right, 3)},
    }


# ===========================================================================
# 5. FFF (FLATTENING FILTER FREE) BEAM PROFILE ANALYSIS
# ===========================================================================

def analyze_fff_profile(
    x_mm: Union[np.ndarray, list],
    values: Union[np.ndarray, list],
    smoothing_radius_mm: float = 0.5,
    cax_pos: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Rigorously evaluate a Flattening Filter Free (FFF) unflattened beam profile.

    Features:
    1. Central axis peak normalization (D(0) = 100%).
    2. Inflection point derivative field width determination:
       - 1D Gaussian smoothing to suppress quantum/detector noise.
       - Numerical differentiation dD/dx.
       - Sub-pixel cubic spline extrema localization on left (|dD/dx| max) and right (-dD/dx max).
    3. Inflection-anchored 80%-20% penumbra determination (Fogliata / IEC FFF):
       - Local step height D_step = 2 * D_infl.
       - D_80 = 1.60 * D_infl, D_20 = 0.40 * D_infl.
    4. Off-axis point-difference symmetry and area symmetry across inner 80% inflection width.
    5. FFF lateral shoulder slope analysis (%/mm or %/cm).

    Returns:
        Dict with inflection_width_mm, penumbra_left_mm, penumbra_right_mm,
        symmetry_pct, shoulder slopes, status, etc.
    """
    x = np.asarray(x_mm, dtype=float)
    y = np.asarray(values, dtype=float)

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    dx_mean = float(np.mean(np.diff(x)))
    if dx_mean <= 0:
        dx_mean = 0.5

    # Interpolator
    f_interp = interpolate.interp1d(x, y, kind="cubic", bounds_error=False, fill_value="extrapolate")

    # Find CAX peak: use cax_pos if provided, otherwise locate around the peak of y
    if cax_pos is not None:
        center = float(cax_pos)
        search_mask = np.abs(x - center) <= 15.0
        if np.any(search_mask):
            peak_idx = int(np.argmax(y[search_mask]))
            peak_pos_mm = float(x[search_mask][peak_idx])
        else:
            peak_pos_mm = center
    else:
        if len(x) == 0:
            peak_pos_mm = 0.0
        else:
            # If coordinates span around 0.0, prioritize central search around 0.0
            if np.min(x) <= 0.0 <= np.max(x):
                central_mask = np.abs(x) <= 15.0
                if np.any(central_mask):
                    peak_idx = int(np.argmax(y[central_mask]))
                    peak_pos_mm = float(x[central_mask][peak_idx])
                else:
                    peak_pos_mm = float(x[np.argmax(y)])
            else:
                peak_pos_mm = float(x[np.argmax(y)])

    d_peak = float(f_interp(peak_pos_mm))
    if d_peak <= 0:
        d_peak = float(np.max(y))

    y_norm = (y / d_peak) * 100.0

    # Gaussian smoothing to prevent gradient noise amplification
    sigma_pts = max(1.0, smoothing_radius_mm / dx_mean)
    y_smooth = ndimage.gaussian_filter1d(y_norm, sigma=sigma_pts)

    # First derivative dD/dx
    d1 = np.gradient(y_smooth, x)

    # Split into left (x < peak_pos_mm) and right (x > peak_pos_mm) halves
    left_mask = x < peak_pos_mm
    right_mask = x > peak_pos_mm

    x_l, d1_l, y_l = x[left_mask], d1[left_mask], y_norm[left_mask]
    x_r, d1_r, y_r = x[right_mask], d1[right_mask], y_norm[right_mask]

    # Guard against empty slices
    if len(d1_l) == 0 or len(d1_r) == 0:
        raise ValueError("Insufficient data points on left or right half of profile to determine inflection points.")

    # Find inflection points:
    # On the left: dD/dx is positive, reaches maximum at the inflection point
    # On the right: dD/dx is negative, reaches minimum (max gradient downward) at inflection point
    idx_max_l = int(np.argmax(d1_l))
    idx_min_r = int(np.argmin(d1_r))

    # Sub-pixel spline optimization around discrete extrema
    # Left inflection
    window = max(2, int(round(5.0 / dx_mean)))
    w_start_l = max(0, idx_max_l - window)
    w_end_l = min(len(x_l), idx_max_l + window + 1)
    if (w_end_l - w_start_l) >= 4:
        spline_d1_l = interpolate.CubicSpline(x_l[w_start_l:w_end_l], d1_l[w_start_l:w_end_l])
        opt_res_l = optimize.minimize_scalar(
            lambda pos: -float(spline_d1_l(pos)),
            bounds=(x_l[w_start_l], x_l[w_end_l - 1]),
            method="bounded"
        )
        x_infl_l = float(opt_res_l.x)
    else:
        x_infl_l = float(x_l[idx_max_l])

    # Right inflection
    w_start_r = max(0, idx_min_r - window)
    w_end_r = min(len(x_r), idx_min_r + window + 1)
    if (w_end_r - w_start_r) >= 4:
        spline_d1_r = interpolate.CubicSpline(x_r[w_start_r:w_end_r], d1_r[w_start_r:w_end_r])
        opt_res_r = optimize.minimize_scalar(
            lambda pos: float(spline_d1_r(pos)),
            bounds=(x_r[w_start_r], x_r[w_end_r - 1]),
            method="bounded"
        )
        x_infl_r = float(opt_res_r.x)
    else:
        x_infl_r = float(x_r[idx_min_r])

    inflection_width_mm = float(x_infl_r - x_infl_l)
    cax_offset_mm = float((x_infl_l + x_infl_r) / 2.0)

    # Doses at inflection points
    f_smooth = interpolate.interp1d(x, y_smooth, kind="cubic", bounds_error=False, fill_value="extrapolate")
    d_infl_l = float(f_smooth(x_infl_l))
    d_infl_r = float(f_smooth(x_infl_r))

    # FFF Inflection-Anchored Penumbra (80%-20% of 2 * D_infl):
    # D_80 = 1.60 * D_infl
    # D_20 = 0.40 * D_infl
    d_80_l = 1.60 * d_infl_l
    d_20_l = 0.40 * d_infl_l

    d_80_r = 1.60 * d_infl_r
    d_20_r = 0.40 * d_infl_r

    def find_crossing(x_arr, y_arr, target_val):
        diffs = y_arr - target_val
        signs = np.sign(diffs)
        sc = np.where(np.diff(signs) != 0)[0]
        if len(sc) == 0:
            return float(x_arr[np.argmin(np.abs(diffs))])
        idx = sc[0]
        x0, x1 = x_arr[idx], x_arr[idx + 1]
        y0, y1 = y_arr[idx], y_arr[idx + 1]
        if y1 == y0:
            return float(x0)
        return float(x0 + (target_val - y0) * (x1 - x0) / (y1 - y0))

    x_l_20 = find_crossing(x_l, y_l, d_20_l)
    x_l_80 = find_crossing(x_l, y_l, d_80_l)
    penumbra_left_mm = float(np.abs(x_l_80 - x_l_20))

    x_r_80 = find_crossing(x_r, y_r, d_80_r)
    x_r_20 = find_crossing(x_r, y_r, d_20_r)
    penumbra_right_mm = float(np.abs(x_r_20 - x_r_80))
    penumbra_mean_mm = float((penumbra_left_mm + penumbra_right_mm) / 2.0)

    # FFF Evaluation region: inner 80% of inflection width
    eval_half_w = 0.40 * inflection_width_mm
    eval_left = cax_offset_mm - eval_half_w
    eval_right = cax_offset_mm + eval_half_w

    # FFF Symmetry: Point-difference across [0, eval_half_w]
    sym_xs = np.linspace(0.0, eval_half_w, 101)
    d_pos = f_smooth(cax_offset_mm + sym_xs)
    d_neg = f_smooth(cax_offset_mm - sym_xs)
    diffs = d_pos - d_neg
    max_idx = np.argmax(np.abs(diffs))
    symmetry_pct = float(diffs[max_idx])

    # FFF Area Symmetry
    eval_mask = (x >= eval_left) & (x <= eval_right)
    if np.sum(eval_mask) >= 6:
        mask_l = (x >= eval_left) & (x <= cax_offset_mm)
        mask_r = (x >= cax_offset_mm) & (x <= eval_right)
        area_l = float(integrate.trapezoid(y_norm[mask_l], x[mask_l]))
        area_r = float(integrate.trapezoid(y_norm[mask_r], x[mask_r]))
        symmetry_area_pct = float(100.0 * (area_l - area_r) / max(area_l + area_r, 1e-4))
    else:
        symmetry_area_pct = 0.0

    # FFF Shoulder Slopes: Left [-0.40 W_infl, -0.10 W_infl], Right [+0.10 W_infl, +0.40 W_infl]
    sh_mask_l = (x >= cax_offset_mm - 0.40 * inflection_width_mm) & (x <= cax_offset_mm - 0.10 * inflection_width_mm)
    sh_mask_r = (x >= cax_offset_mm + 0.10 * inflection_width_mm) & (x <= cax_offset_mm + 0.40 * inflection_width_mm)

    if np.sum(sh_mask_l) >= 2:
        p_l = np.polyfit(x[sh_mask_l], y_norm[sh_mask_l], 1)
        left_slope = float(p_l[0])  # %/mm
    else:
        left_slope = 0.0

    if np.sum(sh_mask_r) >= 2:
        p_r = np.polyfit(x[sh_mask_r], y_norm[sh_mask_r], 1)
        right_slope = float(p_r[0])  # %/mm
    else:
        right_slope = 0.0

    slope_diff = float(np.abs(left_slope) - np.abs(right_slope))

    # Flatness is N/A for FFF beams under international consensus
    flatness_status_str = "N/A (FFF Beam)"

    # Clinical status evaluation: FFF symmetry tolerance <= 1.5% OK, <= 2.5% Tol, > 2.5% Action
    status = "OK"
    if np.isnan(inflection_width_mm) or np.isnan(symmetry_pct):
        status = "Invalid"
    elif abs(symmetry_pct) > 2.5:
        status = "Action"
    elif abs(symmetry_pct) > 1.5:
        status = "Tolerance"

    return {
        "protocol": "FFF Inflection",
        "beam_type": "FFF",
        "inflection_width_mm": round(inflection_width_mm, 3),
        "field_width_mm": round(inflection_width_mm, 3),
        "cax_offset_mm": round(cax_offset_mm, 3),
        "peak_pos_mm": round(peak_pos_mm, 3),
        "d_peak": round(d_peak, 3),
        "d_infl_left": round(d_infl_l, 3),
        "d_infl_right": round(d_infl_r, 3),
        "penumbra_left_mm": round(penumbra_left_mm, 3),
        "penumbra_right_mm": round(penumbra_right_mm, 3),
        "penumbra_mean_mm": round(penumbra_mean_mm, 3),
        "flatness_pct": None,
        "flatness_display": flatness_status_str,
        "symmetry_pct": round(symmetry_pct, 3),
        "symmetry_area_pct": round(symmetry_area_pct, 3),
        "left_slope_pct_mm": round(left_slope, 4),
        "right_slope_pct_mm": round(right_slope, 4),
        "slope_diff_pct_mm": round(slope_diff, 4),
        "status": status,
        "edges": {"left": round(x_infl_l, 3), "right": round(x_infl_r, 3)},
        "penumbra_points": {
            "left_20": round(x_l_20, 3),
            "left_80": round(x_l_80, 3),
            "right_80": round(x_r_80, 3),
            "right_20": round(x_r_20, 3),
        },
        "eval_region": {"left": round(eval_left, 3), "right": round(eval_right, 3)},
    }


# ===========================================================================
# 6. UNIFIED CONTROLLER & HIGH-LEVEL INTERFACE
# ===========================================================================

def analyze_beam_profile(
    x_mm: Union[np.ndarray, list],
    values: Union[np.ndarray, list],
    beam_type: str = "Flattened",
    protocol: str = "IEC",
    cax_pos: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Unified entry-point for analyzing any beam profile.
    Automatically dispatches to analyze_flattened_profile or analyze_fff_profile.
    """
    is_fff = str(beam_type).upper() == "FFF" or "FFF" in str(beam_type).upper()
    if is_fff:
        return analyze_fff_profile(x_mm, values, cax_pos=cax_pos)
    else:
        return analyze_flattened_profile(x_mm, values, protocol=protocol, cax_pos=cax_pos)


def analyze_pdd_curve(
    z_mm: Union[np.ndarray, list],
    dose_vals: Union[np.ndarray, list],
    ssd_cm: float = 100.0,
) -> Dict[str, Any]:
    """
    Rigorously evaluate a Percentage Depth Dose (PDD) curve in water for high-energy photon beams.

    Calculates:
    1. d_max_mm: Depth of dose maximum (mm).
    2. pdd_10: PDD at reference depth 10 cm (100 mm) normalized to 100% at d_max.
    3. pdd_20: PDD at depth 20 cm (200 mm) normalized to 100% at d_max.
    4. pdd_20_10: Ratio PDD(20 cm) / PDD(10 cm).
    5. tpr20_10: IAEA TRS-398 Rev. 1 (2024, Eq. 12) Beam Quality Index:
       TPR20,10 = 1.2661 * (PDD20 / PDD10) - 0.0595
    6. Interpolated / smoothed curve arrays for plotting.
    """
    z = np.asarray(z_mm, dtype=float)
    dose = np.asarray(dose_vals, dtype=float)

    if len(z) < 5:
        raise ValueError("PDD analysis requires at least 5 depth points.")

    sort_idx = np.argsort(z)
    z = z[sort_idx]
    dose = dose[sort_idx]

    d_max_val = float(np.max(dose))
    d_max_idx = int(np.argmax(dose))
    d_max_mm = float(z[d_max_idx])

    # Normalize to 100% at dmax
    pdd = (dose / max(d_max_val, 1e-6)) * 100.0

    # Interpolate
    f_pdd = interpolate.interp1d(z, pdd, kind="cubic" if len(z) > 10 else "linear", fill_value="extrapolate")

    pdd_10 = float(f_pdd(100.0))
    pdd_20 = float(f_pdd(200.0))
    pdd_20_10 = float(pdd_20 / max(pdd_10, 1e-4))

    # IAEA TRS-398 Eq. 12
    tpr20_10 = float(1.2661 * pdd_20_10 - 0.0595)

    return {
        "d_max_mm": round(d_max_mm, 2),
        "pdd_10": round(pdd_10, 2),
        "pdd_20": round(pdd_20, 2),
        "pdd_20_10": round(pdd_20_10, 4),
        "tpr20_10": round(tpr20_10, 4),
        "z_mm": z,
        "pdd_pct": pdd,
        "status": "OK" if 0.50 <= tpr20_10 <= 0.85 else "Action",
    }



class BeamProfileAnalyzer:
    """
    Object-oriented wrapper for radiotherapy beam profile analysis.
    Provides complete state storage and serialization for UI & test verification.
    """

    def __init__(self, x_mm: Union[np.ndarray, list], values: Union[np.ndarray, list], meta: Optional[dict] = None):
        self.x_mm = np.asarray(x_mm, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.meta = meta or {}
        self.results: Dict[str, Any] = {}

    def analyze(self, beam_type: str = "Flattened", protocol: str = "IEC") -> Dict[str, Any]:
        """Run analysis on stored profile."""
        self.results = analyze_beam_profile(self.x_mm, self.values, beam_type=beam_type, protocol=protocol)
        return self.results

    def to_json(self) -> str:
        """Serialize analysis results and downsampled profile points for persistence."""
        # Downsample profile to ~100 points for database storage
        n_pts = len(self.x_mm)
        if n_pts > 100:
            indices = np.linspace(0, n_pts - 1, 100, dtype=int)
            x_samp = self.x_mm[indices].tolist()
            y_samp = self.values[indices].tolist()
        else:
            x_samp = self.x_mm.tolist()
            y_samp = self.values.tolist()

        payload = {
            "meta": self.meta,
            "results": self.results,
            "sampled_profile": {"x_mm": x_samp, "values": y_samp},
        }
        return json.dumps(payload)
