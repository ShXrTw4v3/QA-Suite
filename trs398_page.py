"""
trs398_page.py — Micro QA + IAEA TRS-398 (Rev. 1, 2024) Reference Dosimetry Suite
=============================================================================
Unified TRS-398 Rev. 1 (2024) Absorbed Dose Calibration module.
Supports:
  - High-Energy Photon Beams (Conventional Flattened & Flattening Filter-Free FFF)
    using IAEA TRS-398 Rev. 1 (2024) Table 16 / Eq. (34)
  - High-Energy Electron Beams using IAEA TRS-398 Rev. 1 (2024) Table 20 / Eqs. (37), (39), (98)-(99)
  - Direct integration with Micro-QA Linac Database (`units`)
  - Volume Averaging Correction (k_vol) per TRS-398 Rev. 1 Eq. (22) & Table 11
  - High Dose-per-Pulse (DPP) & Ion Recombination (k_s) monitoring per Table 10
  - Step-by-step mathematical derivations with live substitutions
  - Interactive SVG physical geometry schematic & live Plotly k_Q curves
  - Embedded Interactive Visual Suite & Clinical Certificate generator
  - SQLite persistence in `calibration_logs`
"""

from __future__ import annotations

import os
import math
import warnings
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime

from crud import (
    get_units, log_calibration, get_calibration_logs,
    get_reference_beam_data, get_reference_beam_by_id,
    get_ion_chambers, get_unit_energies,
)
import trs398_calculator as calc_engine
from clinical_pdf_generator import generate_trs398_pdf
import beam_profile_analyzer as bpa

# Try importing pylinac for optional cross-verification
try:
    from pylinac.calibration.trs398 import TRS398Photon, TRS398Electron, KQ_PHOTON_CHAMBERS, KQ_ELECTRON_CHAMBERS
    _HAS_PYLINAC = True
except ImportError:
    _HAS_PYLINAC = False
    KQ_PHOTON_CHAMBERS = calc_engine.KQ_PHOTON_CHAMBERS
    KQ_ELECTRON_CHAMBERS = {}


# ===========================================================================
# Helper — Metric Display Cards
# ===========================================================================

def _metric_card(label: str, value, fmt: str = ".4f", subtext: str = "") -> None:
    """Displays a formatted metric card."""
    if value is None:
        st.metric(label, "N/A")
    else:
        val_str = f"{value:{fmt}}" if isinstance(value, (int, float)) else str(value)
        st.metric(label, val_str, delta=subtext if subtext else None)


def _render_results_dashboard(res: dict, target_output: float = 1.000) -> None:
    """Renders the primary clinical results dashboard."""
    st.markdown("---")
    st.subheader("📊 IAEA TRS-398 Absolute Dosimetry Results")

    dose_zref_cgy = res["d_w_zref_cgy"]
    dose_dmax_cgy = res["d_w_zmax_cgy"]
    dose_per_mu_dmax = res["dose_per_mu_zmax_cgy"]
    dose_per_mu_zref = res["dose_per_mu_zref_cgy"]
    diff_pct = res["diff_pct"]
    trim_pct = res["recommended_mu_trim_pct"]

    # ── Status Announcement Box ───────────────────────────────────────────
    abs_diff = abs(diff_pct)
    if abs_diff <= 1.0:
        st.success(f"✅ **IN CLINICAL TOLERANCE (PASS):** Output at d_max is **{dose_per_mu_dmax:.4f} cGy/MU** (Δ = {diff_pct:+.2f}% from nominal target {target_output:.3f} cGy/MU).")
    elif abs_diff <= 2.0:
        st.warning(f"⚠️ **ACTION LEVEL WARNING:** Output at d_max is **{dose_per_mu_dmax:.4f} cGy/MU** (Δ = {diff_pct:+.2f}%). Review linac calibration.")
    else:
        st.error(f"🚨 **OUT OF TOLERANCE:** Output at d_max is **{dose_per_mu_dmax:.4f} cGy/MU** (Δ = {diff_pct:+.2f}%). Immediate machine adjustment required (Trim: {trim_pct:+.2f}%).")

    # ── KPI Row 1: Absorbed Doses ─────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _metric_card("Dose at d_max", f"{dose_dmax_cgy:.2f} cGy", subtext=f"{res['d_w_zmax_gy']:.4f} Gy")
    with k2:
        _metric_card("Linac Output (d_max)", f"{dose_per_mu_dmax:.4f} cGy/MU", subtext=f"Target: {target_output:.3f}")
    with k3:
        _metric_card("Output Deviation (Δ)", f"{diff_pct:+.2f}%", subtext="Tolerance: ±1.0%")
    with k4:
        _metric_card("Recommended MU Trim", f"{trim_pct:+.2f}%", subtext="Potentiometer adjustment")

    # ── KPI Row 2: Correction Factors ─────────────────────────────────────
    st.markdown("##### 🔍 Correction Factors Breakdown")
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    with f1:
        _metric_card("k_Q (Beam Quality)", res["k_q"])
    with f2:
        _metric_card("k_TP (Air Density)", res["k_tp"])
    with f3:
        _metric_card("k_pol (Polarity)", res["k_pol"])
    with f4:
        _metric_card("k_s (Recomb)", res["k_s"])
    with f5:
        _metric_card("k_vol (FFF Averaging)", res["k_vol"])
    with f6:
        _metric_card("Total k_tot", res["k_tot"])

    if res["is_fff"]:
        st.info(
            f"⚡ **FFF Mode Active:** Volume averaging factor **k_vol = {res['k_vol']:.4f}** applied. "
            f"Linac Dose Rate: **{res['dose_rate_mu_min']:.0f} MU/min** | Estimated Dose Per Pulse: **{res['dose_per_pulse_mgy']:.3f} mGy/pulse**."
        )

    if res.get("ks_warning"):
        st.warning(f"⚠️ {res['ks_warning']}")


# ===========================================================================
# Plotly Chart: k_Q vs TPR20,10
# ===========================================================================

def _build_kq_curve(chamber_key: str, active_tpr: float, active_kq: float) -> go.Figure:
    """Builds an interactive Plotly curve of k_Q vs TPR20,10 with crosshair marker."""
    tprs = calc_engine.KQ_PHOTON_TPRS
    resolved = calc_engine.find_chamber_key(chamber_key)
    active_kqs = calc_engine.KQ_PHOTON_CHAMBERS.get(resolved, calc_engine.KQ_PHOTON_CHAMBERS["PTW 30013 Farmer (0.60 cm³, waterproof)"])

    fig = go.Figure()

    # Draw other chambers lightly in background
    for name, kqs in calc_engine.KQ_PHOTON_CHAMBERS.items():
        if name != resolved:
            fig.add_trace(go.Scatter(
                x=list(tprs), y=list(kqs),
                mode="lines",
                name=name.split("(")[0].strip(),
                line=dict(color="#cbd5e1", width=1),
                hoverinfo="skip",
                showlegend=False,
            ))

    # Active chamber curve
    fig.add_trace(go.Scatter(
        x=list(tprs), y=list(active_kqs),
        mode="lines+markers",
        name=f"Selected: {resolved.split('(')[0].strip()}",
        line=dict(color="#0284c7", width=3),
        marker=dict(size=6, color="#0284c7"),
    ))

    # Active operating point crosshair
    fig.add_trace(go.Scatter(
        x=[active_tpr], y=[active_kq],
        mode="markers",
        name="Operating Point",
        marker=dict(color="#ef4444", size=14, symbol="circle", line=dict(color="#fff", width=2)),
        hovertemplate=f"<b>Current Setup</b><br>TPR₂₀,₁₀ = {active_tpr:.3f}<br>k_Q = {active_kq:.4f}<extra></extra>",
    ))

    fig.add_vline(x=active_tpr, line=dict(color="#ef4444", width=1.5, dash="dash"))
    fig.add_hline(y=active_kq, line=dict(color="#ef4444", width=1.5, dash="dash"))

    is_dark = st.session_state.get("app_theme_dark", True) if "app_theme_dark" in st.session_state else True
    plot_template = "plotly_dark" if is_dark else "plotly_white"
    grid_col = "rgba(255, 255, 255, 0.08)" if is_dark else "#e8edf3"

    fig.update_layout(
        template=plot_template,
        title=dict(text=f"<b>Beam Quality Factor k_Q vs TPR₂₀,₁₀</b> — {resolved.split('(')[0].strip()}", font=dict(size=15)),
        xaxis=dict(title="Beam Quality Index (TPR₂₀,₁₀)", gridcolor=grid_col),
        yaxis=dict(title="k_Q,Q₀ Factor", gridcolor=grid_col),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ===========================================================================
# 1. PHOTON BEAM TAB (With FFF & Micro-QA Database Link)
# ===========================================================================

def _photon_tab() -> None:
    """Render the enhanced TRS-398 Photon calibration module."""
    units_df = get_units()
    unit_names = units_df["name"].tolist() if not units_df.empty else ["Varian TrueBeam #1", "Varian TrueBeam #2"]

    chambers = list(calc_engine.KQ_PHOTON_CHAMBERS.keys())

    # ── Top Row: Preset Selection & Unit Linking ─────────────────────────
    st.markdown("#### ⚡ Clinical Setup & Machine Preset")
    preset_cols = st.columns([2, 2, 2])
    
    with preset_cols[0]:
        selected_unit = st.selectbox("Linac Machine (from Clinic DB)", unit_names, key="ph_unit_sel")

    # Dynamically fetch configured energies for the selected Linac
    unit_energies_df = get_unit_energies(unit_name=selected_unit)
    cfg_energies = unit_energies_df["energy_name"].tolist() if not unit_energies_df.empty else []

    with preset_cols[1]:
        preset_options = []
        if cfg_energies:
            preset_options.extend([f"Linac Config: {e}" for e in cfg_energies])
        preset_options.extend([
            "Custom Parameters",
            "User Setup (6 MV W, PTW 30013, SSD 100, z=10cm)",
            "Varian TrueBeam - 6 MV FFF (1400 MU/min)",
            "Varian TrueBeam - 10 MV (Flattened)",
            "Varian TrueBeam - 10 MV FFF (2400 MU/min)",
            "Elekta Versa HD - 6 MV FFF (1400 MU/min)",
        ])
        preset_choice = st.selectbox("Beam Energy / Preset", preset_options, key="ph_preset_sel")

    with preset_cols[2]:
        meas_date = st.date_input("Measurement Date", value=date.today(), key="ph_date")

    # Apply preset defaults if chosen
    default_energy = 6
    default_tpr = 0.670
    default_pdd = 67.0
    default_fff = False
    default_dose_rate = 600
    default_kvol = 1.0000

    if preset_choice.startswith("Linac Config: "):
        cfg_en_name = preset_choice.replace("Linac Config: ", "").strip()
        en_row = unit_energies_df[unit_energies_df["energy_name"] == cfg_en_name].iloc[0]
        default_energy = int(en_row["nominal_mv"])
        default_fff = "FFF" in cfg_en_name.upper() or "FFF" in str(en_row.get("beam_type", "")).upper()
        if pd.notnull(en_row.get("tpr20_10")) and float(en_row["tpr20_10"]) > 0:
            default_tpr = float(en_row["tpr20_10"])
        if pd.notnull(en_row.get("pdd_10")) and float(en_row["pdd_10"]) > 0:
            default_pdd = float(en_row["pdd_10"])
        if default_fff:
            default_dose_rate = 2400 if default_energy >= 10 else 1400
            default_kvol = 1.0034 if default_energy >= 10 else 1.0015
        else:
            default_dose_rate = 600
            default_kvol = 1.0000
    elif "User Setup" in preset_choice:
        default_energy = 6; default_tpr = 0.670; default_pdd = 67.0; default_fff = False; default_dose_rate = 600; default_kvol = 1.0000
    elif "6 MV FFF" in preset_choice:
        default_energy = 6; default_tpr = 0.635; default_pdd = 63.5; default_fff = True; default_dose_rate = 1400; default_kvol = 1.0015
    elif "10 MV (Flattened)" in preset_choice:
        default_energy = 10; default_tpr = 0.735; default_pdd = 73.5; default_fff = False; default_dose_rate = 600; default_kvol = 1.0000
    elif "10 MV FFF" in preset_choice:
        default_energy = 10; default_tpr = 0.705; default_pdd = 70.5; default_fff = True; default_dose_rate = 2400; default_kvol = 1.0034

    st.markdown("---")

    # ── Automated PDD & TPR20,10 Auto-Populate Expander ───────────────────
    with st.expander("📥 Auto-Determine BQI (TPR₂₀,₁₀) & PDD₁₀ from Tank Scan or Database Baseline", expanded=False):
        st.markdown(
            "According to **IAEA TRS-398 Rev. 1 (2024, Eq. 12)**: "
            "$$\\text{TPR}_{20,10} = 1.2661 \\times \\left(\\frac{\\text{PDD}_{20}}{\\text{PDD}_{10}}\\right) - 0.0595$$"
        )
        pdd_mode = st.radio("Select Beam Quality Source", ["From Stored Linac Baseline (Database)", "Upload Tank PDD Scan (.mcc, .csv, .tsv)"], horizontal=True, key="trs_pdd_source_mode")

        if pdd_mode == "From Stored Linac Baseline (Database)":
            db_pdds = get_reference_beam_data(curve_type="PDD")
            if db_pdds.empty:
                st.info("No reference PDD curves stored in database yet. Upload a PDD in the '📐 Beam Profile QA' -> '🏛️ Linac Reference Beam Data & PDD Manager' section.")
            else:
                sel_pdd_id = st.selectbox(
                    "Choose Machine Reference PDD",
                    db_pdds["id"].tolist(),
                    format_func=lambda i: f"{db_pdds.loc[db_pdds['id'] == i, 'unit_name'].values[0]} — {db_pdds.loc[db_pdds['id'] == i, 'energy'].values[0]} (TPR20,10={db_pdds.loc[db_pdds['id'] == i, 'tpr20_10'].values[0]:.4f}, PDD10={db_pdds.loc[db_pdds['id'] == i, 'pdd_10'].values[0]:.1f}%)",
                    key="trs_sel_pdd_db"
                )
                sel_pdd_rec = get_reference_beam_by_id(sel_pdd_id)
                if sel_pdd_rec and sel_pdd_rec.get("tpr20_10"):
                    cp1, cp2, cp3 = st.columns(3)
                    cp1.metric("Database TPR₂₀,₁₀", f"{sel_pdd_rec['tpr20_10']:.4f}")
                    cp2.metric("Database PDD₁₀", f"{sel_pdd_rec['pdd_10']:.1f}%")
                    cp3.metric("d_max", f"{sel_pdd_rec.get('d_max_mm', 14.0):.1f} mm")
                    if st.button("Apply Database TPR₂₀,₁₀ & PDD₁₀ to Calculation", key="trs_apply_db_pdd_btn"):
                        st.session_state["ph_tpr"] = float(sel_pdd_rec["tpr20_10"])
                        st.session_state["ph_pdd"] = float(sel_pdd_rec["pdd_10"])
                        st.success(f"Applied TPR₂₀,₁₀ = {sel_pdd_rec['tpr20_10']:.4f} and PDD₁₀ = {sel_pdd_rec['pdd_10']:.1f}% for {sel_pdd_rec['unit_name']} {sel_pdd_rec['energy']}.")
                        st.rerun()
        else:
            pdd_up_file = st.file_uploader("Upload Water Tank PDD Scan", type=["mcc", "csv", "tsv", "txt", "asc", "dat"], key="trs_pdd_file_up")
            if pdd_up_file is not None:
                try:
                    pdd_bytes = pdd_up_file.getvalue()
                    parsed = bpa.parse_profile_text(pdd_bytes)
                    pdd_target = None
                    if "profiles" in parsed and parsed["profiles"]:
                        for k, v in parsed["profiles"].items():
                            if "PDD" in k.upper():
                                pdd_target = v
                                break
                        if pdd_target is None:
                            pdd_target = list(parsed["profiles"].values())[0]
                    else:
                        pdd_target = parsed

                    p_res = bpa.analyze_pdd_curve(pdd_target["x_mm"], pdd_target["values"])
                    cp1, cp2, cp3 = st.columns(3)
                    cp1.metric("Calculated TPR₂₀,₁₀", f"{p_res['tpr20_10']:.4f}")
                    cp2.metric("Calculated PDD₁₀", f"{p_res['pdd_10']:.1f}%")
                    cp3.metric("d_max", f"{p_res['d_max_mm']:.1f} mm")
                    if st.button("Apply Calculated TPR₂₀,₁₀ & PDD₁₀ to Calculation", key="trs_apply_calc_pdd_btn"):
                        st.session_state["ph_tpr"] = float(p_res["tpr20_10"])
                        st.session_state["ph_pdd"] = float(p_res["pdd_10"])
                        st.success(f"Applied TPR₂₀,₁₀ = {p_res['tpr20_10']:.4f} and PDD₁₀ = {p_res['pdd_10']:.1f}%.")
                        st.rerun()
                except Exception as p_err:
                    st.error(f"Failed to analyze PDD scan: {p_err}")

    # ── Geometry & Chamber Controls ───────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    reg_chambers = get_ion_chambers(active_only=True)

    with c1:
        st.markdown("**1. Equipment & Chamber**")
        if not reg_chambers.empty:
            ch_mode = st.radio("Chamber Selection", ["From Certified Registry", "All TRS-398 Catalog"], horizontal=True, key="ph_ch_mode")
            if ch_mode == "From Certified Registry":
                ch_ids = reg_chambers["id"].tolist()
                def format_reg_ch(cid):
                    r = reg_chambers[reg_chambers["id"] == cid].iloc[0]
                    lbl = r["user_label"] or r["model"].split("(")[0].strip()
                    return f"{lbl} (S/N {r['serial_no']}) — N_D,w: {r['ndw_cgy']:.4f} cGy/nC"
                sel_cid = st.selectbox("Registered Chamber", ch_ids, format_func=format_reg_ch, key="ph_reg_ch_id")
                r_ch = reg_chambers[reg_chambers["id"] == sel_cid].iloc[0]
                matched_model = calc_engine.find_chamber_key(r_ch["model"])
                chamber = matched_model
                chamber_sn = str(r_ch["serial_no"])
                n_dw_cgy = float(r_ch["ndw_cgy"])
                st.caption(f"Certified: **{r_ch['cal_lab']}** ({r_ch['cal_date']})")
                k_elec = st.number_input("k_elec", value=1.0000, format="%.4f", step=0.0001, key="ph_kelec")
            else:
                chamber = st.selectbox("Ion Chamber (TRS-398 Rev. 1 Table 16)", chambers, index=0, key="ph_chamber")
                chamber_sn = st.text_input("Chamber Serial No", value="PTW-30013-10482", key="ph_sn")
                n_dw_cgy = st.number_input("N_D,w (cGy/nC)", value=5.385, format="%.4f", step=0.001, key="ph_ndw")
                k_elec = st.number_input("k_elec", value=1.0000, format="%.4f", step=0.0001, key="ph_kelec")
        else:
            chamber = st.selectbox("Ion Chamber (TRS-398 Rev. 1 Table 16)", chambers, index=0, key="ph_chamber")
            chamber_sn = st.text_input("Chamber Serial No", value="PTW-30013-10482", key="ph_sn")
            n_dw_cgy = st.number_input("N_D,w (cGy/nC)", value=5.385, format="%.4f", step=0.001, key="ph_ndw")
            k_elec = st.number_input("k_elec", value=1.0000, format="%.4f", step=0.0001, key="ph_kelec")

    with c2:
        st.markdown("**2. Beam Delivery & FFF Mode**")
        col_en, col_fff = st.columns(2)
        with col_en:
            energy_mv = st.number_input("Nominal Energy (MV)", min_value=1, max_value=25, value=default_energy, step=1, key="ph_energy")
        with col_fff:
            is_fff = st.checkbox("⚡ FFF Beam", value=default_fff, key="ph_fff")
        
        tpr2010 = st.number_input("TPR₂₀,₁₀ (BQI)", value=default_tpr, format="%.3f", step=0.001, key="ph_tpr")
        pdd_zref = st.number_input("PDD at z_ref (%)", value=default_pdd, format="%.1f", step=0.1, key="ph_pdd")

        if is_fff:
            auto_kvol = calc_engine.estimate_kvol_fff(chamber, True, tpr2010, sdd_cm=110.0)
            col_kv, col_dr = st.columns(2)
            with col_kv:
                k_vol = st.number_input("k_vol (Eq. 22 / Table 11)", value=float(auto_kvol), format="%.4f", step=0.0005, key="ph_kvol")
            with col_dr:
                dose_rate = st.number_input("Dose Rate (MU/min)", value=default_dose_rate, step=100, key="ph_drate")
        else:
            k_vol = 1.0000
            dose_rate = 600

    with c3:
        st.markdown("**3. Setup Geometry & Environment**")
        setup = st.selectbox("Setup Geometry", ["SSD (100 cm)", "SAD (100 cm)"], key="ph_setup")
        mu_delivered = st.number_input("Delivered MU", min_value=1, value=100, step=10, key="ph_mu")
        col_t, col_p = st.columns(2)
        with col_t:
            temp_c = st.number_input("Temp (°C)", value=20.0, format="%.1f", step=0.1, key="ph_temp")
        with col_p:
            press_kpa = st.number_input("Press (kPa)", value=101.325, format="%.3f", step=0.05, key="ph_press")

    st.markdown("---")

    # ── Voltages & Electrometer Readings ──────────────────────────────────
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.markdown("**Voltage Settings**")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            v1_val = st.number_input("Operating Voltage V₁ (V)", value=300.0, step=10.0, key="ph_v1")
        with col_v2:
            v2_val = st.number_input("Reduced Voltage V₂ (V)", value=100.0, step=10.0, key="ph_v2")

    with v_col2:
        st.markdown("**Polarity Readings (M_+, M_-)**")
        col_mp, col_mm = st.columns(2)
        with col_mp:
            m_plus = st.number_input("Reading M_+ (nC)", value=12.35, format="%.3f", step=0.01, key="ph_mplus")
        with col_mm:
            m_minus = st.number_input("Reading M_- (nC)", value=12.35, format="%.3f", step=0.01, key="ph_mminus")

    st.markdown("**Electrometer Irradiations at Normal V₁ (nC)**")
    r_cols = st.columns(4)
    with r_cols[0]:
        r1 = st.number_input("Reading #1", value=12.34, format="%.3f", step=0.01, key="ph_r1")
    with r_cols[1]:
        r2 = st.number_input("Reading #2", value=12.35, format="%.3f", step=0.01, key="ph_r2")
    with r_cols[2]:
        r3 = st.number_input("Reading #3", value=12.35, format="%.3f", step=0.01, key="ph_r3")
    with r_cols[3]:
        r_v2 = st.number_input("Reading at V₂ (for ks)", value=12.28, format="%.3f", step=0.01, key="ph_rv2")

    notes = st.text_area("Clinical Notes & Machine Log Comments", placeholder="e.g. Annual calibration after target replacement", key="ph_notes")

    # ── Calculation Action ────────────────────────────────────────────────
    st.markdown("")
    calc_button = st.button("🚀 Calculate IAEA TRS-398 (Rev. 1, 2024) Absorbed Dose", type="primary", use_container_width=True)

    if calc_button or "_trs398_photon_res" in st.session_state:
        # Convert cGy/nC to Gy/nC: 5.385 cGy/nC = 0.05385 Gy/nC
        ndw_gy = n_dw_cgy / 100.0
        sdd_val = 110.0 if "SSD" in setup else 100.0

        res = calc_engine.calc_trs398_dose(
            raw_readings=[r1, r2, r3],
            mu_delivered=mu_delivered,
            nd_w=ndw_gy,
            tpr2010=tpr2010,
            chamber_key=chamber,
            temp_c=temp_c,
            press_kpa=press_kpa,
            pdd_zref=pdd_zref,
            m_plus=m_plus,
            m_minus=m_minus,
            v1=v1_val,
            v2=v2_val,
            m1=r2,
            m2=r_v2,
            is_fff=is_fff,
            k_vol=k_vol,
            dose_rate_mu_min=dose_rate,
            sdd_cm=sdd_val,
        )

        st.session_state["_trs398_photon_res"] = res

        # Display primary results dashboard
        _render_results_dashboard(res)

        # Plotly Beam Quality Curve
        st.markdown("")
        fig = _build_kq_curve(chamber, tpr2010, res["kq"])
        st.plotly_chart(fig, use_container_width=True)

        # Step-by-Step Derivation Expander
        with st.expander("📐 Step-by-Step Mathematical Derivation Chain (IAEA TRS-398 Rev. 1, 2024)"):
            kvol_formula_str = r"\times k_{vol}" if is_fff else ""
            kvol_val_str = f"\\times {res['k_vol']:.4f}" if is_fff else ""

            st.markdown(f"""
            1. **Fully Corrected Electrometer Reading ($M_Q$, Section 4.4):**
               $$M_Q = \\bar{{M}}_{{raw}} \\times k_{{TP}} \\times k_{{pol}} \\times k_s \\times k_{{elec}}$$
               $$M_Q = {res['m_raw_avg_nc']:.3f}\\text{{ nC}} \\times {res['k_tp']:.4f} \\times {res['k_pol']:.4f} \\times {res['k_s']:.4f} \\times {res['k_elec']:.4f} = \\mathbf{{{res['m_q_nc']:.4f}\\text{{ nC}}}}$$

            2. **Beam Quality Factor ($k_Q$, Table 16 / Eq. 34):**
               Interpolated from IAEA TRS-398 Rev. 1 (2024) Table 16 at $\\text{{TPR}}_{{20,10}} = {tpr2010:.3f} \\implies \\mathbf{{{res['kq']:.4f}}}$ (Eq. 34 analytical: ${res['k_q_analytical']:.4f}$)

            3. **Absorbed Dose at Reference Depth ($z_{{ref}} = 10\\text{{ cm}}$, Section 6.3):**
               $$D_{{w,Q}}(z_{{ref}}) = M_Q \\times N_{{D,w,Q_0}} \\times k_Q {kvol_formula_str}$$
               $$D_{{w,Q}}(10\\text{{ cm}}) = {res['m_q_nc']:.4f}\\text{{ nC}} \\times {ndw_gy:.6f}\\text{{ Gy/nC}} \\times {res['kq']:.4f} {kvol_val_str} = \\mathbf{{{res['d_w_zref_gy']:.4f}\\text{{ Gy}}}} \\quad ({res['d_w_zref_cgy']:.2f}\\text{{ cGy}})$$

            4. **Absorbed Dose at Depth of Dose Max ($z_{{max}}$, Eq. 32):**
               $$D_{{w,Q}}(z_{{max}}) = D_{{w,Q}}(z_{{ref}}) \\times \\frac{{100}}{{\\text{{PDD}}(z_{{ref}})}} = {res['d_w_zref_gy']:.4f} \\times \\frac{{100}}{{{pdd_zref:.1f}}} = \\mathbf{{{res['d_w_zmax_gy']:.4f}\\text{{ Gy}}}} \\quad ({res['d_w_zmax_cgy']:.2f}\\text{{ cGy}})$$
               $$\\text{{Output}} = \\mathbf{{{res['dose_per_mu_zmax_cgy']:.4f}\\text{{ cGy/MU}}}}$$
            """)

        # Save to SQLite Database & Clinical PDF Export
        save_col1, save_col2 = st.columns([1, 1])
        with save_col1:
            if st.button("💾 Save Calibration to Clinic Database (qa_data.db)", key="ph_save_db", type="primary", use_container_width=True):
                log_calibration(
                    beam_type="photon" if not is_fff else "photon_fff",
                    unit_name=selected_unit,
                    institution="Radiation Oncology Physics",
                    physicist=st.session_state.get("ph_phys", "Medical Physicist"),
                    chamber=chamber,
                    energy=f"{energy_mv}MV{' FFF' if is_fff else ''}",
                    setup=setup.split()[0],
                    n_dw=float(n_dw_cgy),
                    kq=float(res["kq"]),
                    k_pol=float(res["k_pol"]),
                    k_s=float(res["k_s"]),
                    k_tp=float(res["k_tp"]),
                    k_vol_fff=float(res["k_vol"]),
                    m_corrected=float(res["m_q_nc"]),
                    dose_mu_zref=float(res["dose_per_mu_zref_cgy"]),
                    dose_mu_zmax=float(res["dose_per_mu_dmax_cgy"]),
                    notes=notes,
                    performed_at=meas_date.isoformat(),
                )
                st.success(f"✅ Calibration record successfully committed to SQLite database for **{selected_unit}**.")
        with save_col2:
            try:
                ph_pdf_meta = {
                    "unit_name": selected_unit,
                    "institution": "Radiation Oncology Physics",
                    "modality": "Medical Linear Accelerator (LINAC)",
                    "physicist": st.session_state.get("ph_phys", "Medical Physicist"),
                    "energy": f"{energy_mv} MV{' FFF' if is_fff else ''}",
                    "chamber_type": chamber,
                    "protocol_edition": "IAEA TRS-398 Rev. 1 (2024), Section 6",
                }
                ph_pdf_bytes = generate_trs398_pdf(res, ph_pdf_meta)
                st.download_button(
                    label="📄 Download Clinical PDF Certificate",
                    data=ph_pdf_bytes,
                    file_name=f"TRS398_Photon_Report_{selected_unit.replace(' ', '_')}_{meas_date.strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    key="ph_download_pdf",
                    use_container_width=True,
                )
            except Exception as pdf_err:
                st.caption(f"PDF generation error: {pdf_err}")


# ===========================================================================
# 2. INTERACTIVE VISUAL SUITE TAB
# ===========================================================================

def _interactive_visual_tab() -> None:
    """Renders the embedded interactive visual suite and SVG schematics."""
    st.markdown("#### 🌐 Interactive TRS-398 Rev. 1 Physical Simulation & Visual Suite")
    st.caption("Real-time physical geometry, forward-peaked FFF beam profiles, Table 16 inspector, and certificate printing.")

    # Check if index.html exists
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Embed custom CSS & JS directly into iframe
        css_path = os.path.join(os.path.dirname(__file__), "style.css")
        js_path = os.path.join(os.path.dirname(__file__), "app.js")

        css_str = ""
        js_str = ""
        if os.path.exists(css_path):
            with open(css_path, "r", encoding="utf-8") as fc:
                css_str = fc.read()
        if os.path.exists(js_path):
            with open(js_path, "r", encoding="utf-8") as fj:
                js_str = fj.read()

        # Inlined HTML for iframe
        inlined_html = html_content.replace('<link rel="stylesheet" href="style.css">', f"<style>{css_str}</style>")
        inlined_html = inlined_html.replace('<script src="app.js"></script>', f"<script>{js_str}</script>")

        st.components.v1.html(inlined_html, height=920, scrolling=True)
    else:
        st.info("Interactive visual suite files (index.html) found in workspace.")


# ===========================================================================
# 3. ELECTRON BEAM TAB (IAEA TRS-398 Rev. 1, 2024 Section 7 & Table 20)
# ===========================================================================

def _electron_tab() -> None:
    """Render the IAEA TRS-398 Rev. 1 (2024) Section 7 Electron calibration form."""
    el_chambers = list(calc_engine.KQ_ELECTRON_CHAMBERS.keys())
    st.markdown("#### ⚡ Electron Beam Reference Dosimetry (IAEA TRS-398 Rev. 1, Section 7 & Table 20)")

    units_df = get_units()
    unit_names = units_df["name"].tolist() if not units_df.empty else ["Varian TrueBeam #1"]

    e1, e2, e3 = st.columns(3)
    reg_chambers = get_ion_chambers(active_only=True)

    with e1:
        unit_name = st.selectbox("Unit / Machine", unit_names, key="el_unit_sel")
        if not reg_chambers.empty:
            el_ch_mode = st.radio("Chamber Selection", ["From Certified Registry", "All Table 20 Catalog"], horizontal=True, key="el_ch_mode")
            if el_ch_mode == "From Certified Registry":
                el_ch_ids = reg_chambers["id"].tolist()
                def format_el_ch(cid):
                    r = reg_chambers[reg_chambers["id"] == cid].iloc[0]
                    lbl = r["user_label"] or r["model"].split("(")[0].strip()
                    return f"{lbl} (S/N {r['serial_no']}) — N_D,w: {r['ndw_cgy']:.4f} cGy/nC"
                sel_el_cid = st.selectbox("Registered Chamber", el_ch_ids, format_func=format_el_ch, key="el_reg_ch_id")
                r_el = reg_chambers[reg_chambers["id"] == sel_el_cid].iloc[0]
                matched_el = r_el["model"]
                if matched_el not in el_chambers:
                    for ech in el_chambers:
                        if any(token.lower() in ech.lower() for token in r_el["model"].split() if len(token) > 3):
                            matched_el = ech
                            break
                    if matched_el not in el_chambers:
                        matched_el = el_chambers[0]
                chamber = matched_el
                n_dw_cgy = float(r_el["ndw_cgy"])
                st.caption(f"Certified: **{r_el['cal_lab']}** ({r_el['cal_date']})")
                k_elec = st.number_input("k_elec", value=1.0000, format="%.4f", step=0.0001, key="el_kelec")
            else:
                chamber = st.selectbox("Ion Chamber (TRS-398 Rev. 1 Table 20)", el_chambers, key="el_ch")
                n_dw_cgy = st.number_input("N_D,w,Q0 (cGy/nC)", value=5.385, format="%.4f", key="el_ndw")
                k_elec = st.number_input("k_elec", value=1.0000, format="%.4f", step=0.0001, key="el_kelec")
        else:
            chamber = st.selectbox("Ion Chamber (TRS-398 Rev. 1 Table 20)", el_chambers, key="el_ch")
            n_dw_cgy = st.number_input("N_D,w,Q0 (cGy/nC)", value=5.385, format="%.4f", key="el_ndw")
            k_elec = st.number_input("k_elec", value=1.0000, format="%.4f", step=0.0001, key="el_kelec")
    with e2:
        energy_str = st.selectbox("Nominal Energy", ["6 MeV", "9 MeV", "12 MeV", "15 MeV", "18 MeV", "20 MeV"], index=2, key="el_energy")
        cone = st.selectbox("Cone / Applicator (cm × cm)", ["10 × 10", "15 × 15", "20 × 20"], key="el_cone")
        i_50 = st.number_input("R₅₀,ion / I₅₀ (g/cm²)", value=5.00, min_value=1.00, max_value=9.50, format="%.2f", step=0.05, key="el_i50")
        pdd_zref = st.number_input("PDD(z_ref) (%)", value=98.5, min_value=50.0, max_value=105.0, format="%.1f", step=0.1, key="el_pdd")
    with e3:
        mu = st.number_input("Monitor Units", value=100, step=10, key="el_mu")
        temp = st.number_input("Temp (°C)", value=20.0, format="%.1f", key="el_temp")
        press = st.number_input("Press (kPa)", value=101.325, format="%.3f", key="el_press")
        el_date = st.date_input("Measurement Date", value=date.today(), key="el_date")

    st.markdown("---")
    ev1, ev2 = st.columns(2)
    with ev1:
        st.markdown("**Polarity Readings (M_+, M_-)**")
        ep1, ep2 = st.columns(2)
        with ep1:
            el_mplus = st.number_input("Reading M_+ (nC)", value=20.11, format="%.3f", key="el_mplus")
        with ep2:
            el_mminus = st.number_input("Reading M_- (nC)", value=20.11, format="%.3f", key="el_mminus")
    with ev2:
        st.markdown("**Two-Voltage Recombination (V₁, V₂, M₂)**")
        evc1, evc2, evc3 = st.columns(3)
        with evc1:
            el_v1 = st.number_input("V₁ (V)", value=300.0, step=10.0, key="el_v1")
        with evc2:
            el_v2 = st.number_input("V₂ (V)", value=100.0, step=10.0, key="el_v2")
        with evc3:
            el_rv2 = st.number_input("M₂ at V₂ (nC)", value=19.98, format="%.3f", key="el_rv2")

    st.markdown("**Electrometer Readings at V₁ (nC)**")
    er1, er2, er3 = st.columns(3)
    with er1:
        em1 = st.number_input("Reading 1", value=20.10, format="%.3f", key="el_r1")
    with er2:
        em2 = st.number_input("Reading 2", value=20.12, format="%.3f", key="el_r2")
    with er3:
        em3 = st.number_input("Reading 3", value=20.11, format="%.3f", key="el_r3")

    el_notes = st.text_input("Clinical Notes", value="", key="el_notes")

    if st.button("🚀 Calculate IAEA TRS-398 (Rev. 1) Electron Dose", type="primary", use_container_width=True):
        try:
            el_res = calc_engine.calc_trs398_electron_dose(
                raw_readings=[em1, em2, em3],
                mu_delivered=mu,
                nd_w=n_dw_cgy / 100.0,
                i50_cm=i_50,
                chamber_key=chamber,
                temp_c=temp,
                press_kpa=press,
                k_elec=k_elec,
                m_plus=el_mplus,
                m_minus=el_mminus,
                v1=el_v1,
                v2=el_v2,
                m1=em2,
                m2=el_rv2,
                pdd_zref=pdd_zref,
            )
            st.session_state["_trs398_electron_res"] = el_res
        except ValueError as err:
            st.error(f"⚠️ {err}")
            st.session_state.pop("_trs398_electron_res", None)

    if "_trs398_electron_res" in st.session_state:
        el_res = st.session_state["_trs398_electron_res"]
        st.success(
            f"✅ **Measured Electron Output at z_max:** **{el_res['dose_per_mu_zmax_cgy']:.4f} cGy/MU** "
            f"(Δ = {el_res['diff_pct']:+.2f}% from 1.000 cGy/MU) | "
            f"Dose at z_ref ({el_res['z_ref_cm']:.2f} g/cm²): **{el_res['d_w_zref_cgy']:.2f} cGy**"
        )
        ek1, ek2, ek3, ek4, ek5, ek6 = st.columns(6)
        with ek1:
            _metric_card("R₅₀ (g/cm²)", el_res["r50_cm"], fmt=".2f", subtext="Eq. (37)")
        with ek2:
            _metric_card("z_ref (g/cm²)", el_res["z_ref_cm"], fmt=".2f", subtext="Eq. (39)")
        with ek3:
            _metric_card("k_Q (Table 20)", el_res["k_q"], fmt=".4f", subtext=f"Eq. 98/99: {el_res['k_q_analytical']:.4f}")
        with ek4:
            _metric_card("k_TP (Eq. 10)", el_res["k_tp"], fmt=".4f")
        with ek5:
            _metric_card("k_pol (Eq. 11)", el_res["k_pol"], fmt=".4f")
        with ek6:
            _metric_card("k_s (Table 10)", el_res["k_s"], fmt=".4f")

        el_save_col1, el_save_col2 = st.columns([1, 1])
        with el_save_col1:
            if st.button("💾 Save Electron Calibration to Clinic Database (qa_data.db)", key="el_save_db", use_container_width=True):
                log_calibration(
                    beam_type="electron",
                    unit_name=unit_name,
                    institution="Radiation Oncology Physics",
                    physicist="Medical Physicist",
                    chamber=chamber,
                    energy=f"{energy_str} ({cone})",
                    setup="SSD",
                    n_dw=float(n_dw_cgy),
                    kq=float(el_res["k_q"]),
                    k_pol=float(el_res["k_pol"]),
                    k_s=float(el_res["k_s"]),
                    k_tp=float(el_res["k_tp"]),
                    k_vol_fff=1.0,
                    m_corrected=float(el_res["m_q_nc"]),
                    dose_mu_zref=float(el_res["dose_per_mu_zref_cgy"]),
                    dose_mu_zmax=float(el_res["dose_per_mu_zmax_cgy"]),
                    notes=el_notes,
                    performed_at=el_date.isoformat(),
                )
                st.success(f"✅ Electron calibration committed to `qa_data.db` for **{unit_name}**.")
        with el_save_col2:
            try:
                el_pdf_meta = {
                    "unit_name": unit_name,
                    "institution": "Radiation Oncology Physics",
                    "modality": "Medical Linear Accelerator (LINAC)",
                    "physicist": "Medical Physicist",
                    "energy": f"{energy_str} ({cone})",
                    "chamber_type": chamber,
                    "protocol_edition": "IAEA TRS-398 Rev. 1 (2024), Section 7",
                }
                el_pdf_bytes = generate_trs398_pdf(el_res, el_pdf_meta)
                st.download_button(
                    label="📄 Download Clinical PDF Certificate",
                    data=el_pdf_bytes,
                    file_name=f"TRS398_Electron_Report_{unit_name.replace(' ', '_')}_{el_date.strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    key="el_download_pdf",
                    use_container_width=True,
                )
            except Exception as pdf_err:
                st.caption(f"PDF generation error: {pdf_err}")


# ===========================================================================
# 4. CALIBRATION LOG HISTORY TAB
# ===========================================================================

def _history_tab() -> None:
    """Display all saved calibration results from SQLite."""
    st.subheader("📒 Local Clinic Calibration History (`qa_data.db`)")

    logs = get_calibration_logs()
    if logs.empty:
        st.info("No calibration records saved yet. Run a calibration and click 'Save to Clinic Database'.")
        return

    display_cols = [
        "performed_at", "unit_name", "beam_type", "energy", "chamber",
        "kq", "k_vol_fff", "k_tp", "k_s", "dose_mu_zmax", "notes"
    ]
    available = [c for c in display_cols if c in logs.columns]
    st.dataframe(logs[available], use_container_width=True, hide_index=True)

    # Plot trend over time
    if "dose_mu_zmax" in logs.columns and len(logs) > 1:
        st.markdown("##### 📈 Output Constancy Trend (cGy/MU)")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=logs["performed_at"], y=logs["dose_mu_zmax"],
            mode="lines+markers",
            name="Dose/MU at d_max",
            line=dict(color="#0284c7", width=2),
            marker=dict(size=8, color="#0284c7"),
        ))
        fig_trend.add_hline(y=1.000, line=dict(color="#10b981", width=2, dash="dash"), annotation_text="Nominal Target 1.000 cGy/MU")
        is_dark = st.session_state.get("app_theme_dark", True) if "app_theme_dark" in st.session_state else True
        fig_trend.update_layout(
            template="plotly_dark" if is_dark else "plotly_white",
            height=300,
            margin=dict(l=40, r=20, t=30, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    csv = logs.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full Calibration Log CSV",
        data=csv,
        file_name=f"trs398_hospital_calibration_log_{date.today()}.csv",
        mime="text/csv",
    )


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def page_trs398() -> None:
    """Main router for the TRS-398 Reference Dosimetry module."""
    st.markdown(
        '<div class="section-header">☢️ IAEA TRS-398 (Rev. 1, 2024) Reference Dosimetry & Calibration</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Absorbed Dose Determination in External Beam Radiotherapy (IAEA TRS-398 Rev. 1, 2024) — Flattened, FFF & Electron Beams. "
        "Directly integrated with the local clinic SQLite database (`qa_data.db`)."
    )

    tab_photon, tab_visual, tab_electron, tab_history = st.tabs([
        "☀️ Photon Beam (with FFF)",
        "🌐 Interactive Visual Suite",
        "⚡ Electron Beam",
        "📒 Calibration Log & Trends",
    ])

    with tab_photon:
        _photon_tab()

    with tab_visual:
        _interactive_visual_tab()

    with tab_electron:
        _electron_tab()

    with tab_history:
        _history_tab()
