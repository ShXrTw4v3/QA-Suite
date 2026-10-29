"""
beam_profile_page.py — Radiotherapy Beam Profile Flatness, Symmetry & Penumbra Analyzer
=======================================================================================
Streamlit interactive UI module for radiotherapy linear accelerator beam profile QA.

Features:
- Dual data input: Upload File (CSV, TSV, ASCII, or DICOM EPID .dcm) vs Built-in Synthetic Demos.
- Protocols supported: IEC 60976 and Varian for flattened beams; FFF automatic peak
  normalization, derivative inflection edge detection, and point-difference symmetry.
- Plotly Visualizations:
  * Interactive 1D profile with CAX vertical line, inflection/FWHM edge markers,
    80-20% penumbra shaded regions, and inner 80% evaluation zone.
  * Folded Profile Symmetry / Point-difference plot.
  * 2D Fluence heatmap when DICOM EPID image is loaded.
- Numerical metrics dashboard with pass/action/out-of-tolerance badges.
- One-click SQLite logging to `qa_data.db` (`profile_qa_logs` and `test_results`).
- Print-ready clinical PDF report download (`generate_beam_profile_pdf`).
- Namespaced Streamlit keys (`bp_*`).
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import beam_profile_analyzer as bpa
from clinical_pdf_generator import generate_beam_profile_pdf
import crud

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions: Metric Cards & Badges
# =============================================================================

def _render_metric_card(label: str, value: Any, subtext: str = "", delta_color: str = "normal") -> None:
    """Render a formatted metric card."""
    if value is None:
        st.metric(label, "N/A")
    elif isinstance(value, float):
        st.metric(label, f"{value:.2f}", delta=subtext if subtext else None, delta_color=delta_color)
    else:
        st.metric(label, str(value), delta=subtext if subtext else None, delta_color=delta_color)


def _render_status_badge(status: str) -> None:
    """Render an HTML status badge for the QA evaluation."""
    s_upper = str(status).upper()
    if "OK" in s_upper or "PASS" in s_upper:
        st.success("✅ **STATUS: IN CLINICAL TOLERANCE (PASS)** — All beam profile metrics within TG-142 / IEC limits.")
    elif "TOL" in s_upper or "ACTION" in s_upper:
        st.warning("⚠️ **STATUS: ACTION LEVEL (WARNING)** — One or more profile metrics approaching or at tolerance limits.")
    else:
        st.error("🚨 **STATUS: OUT OF TOLERANCE (FAIL)** — Beam profile exceeds allowable action limits. Calibration required.")


# =============================================================================
# Plotly Visualization Builders
# =============================================================================

def _build_1d_profile_figure(
    x_mm: np.ndarray,
    values: np.ndarray,
    analysis: Dict[str, Any],
    beam_title: str = "Beam Profile",
    baseline_x: Optional[np.ndarray] = None,
    baseline_y: Optional[np.ndarray] = None,
    baseline_name: str = "Golden Baseline",
) -> go.Figure:
    """
    Build interactive Plotly 1D profile with:
    - Measured profile curve
    - Optional Machine Golden Baseline curve
    - CAX vertical line
    - Inflection / FWHM edge markers
    - 80-20% penumbra shaded regions
    - Inner 80% evaluation zone shaded region
    """
    fig = go.Figure()

    # Base profile trace
    fig.add_trace(
        go.Scatter(
            x=x_mm,
            y=values,
            mode="lines",
            name="Measured Profile",
            line=dict(color="#0284c7", width=2.5),
            hovertemplate="Position: %{x:.2f} mm<br>Dose: %{y:.2f} %<extra></extra>",
        )
    )

    # Optional Golden Baseline trace
    if baseline_x is not None and baseline_y is not None:
        fig.add_trace(
            go.Scatter(
                x=baseline_x,
                y=baseline_y,
                mode="lines",
                name=baseline_name,
                line=dict(color="#eab308", width=2.2, dash="dash"),
                hovertemplate="Baseline Pos: %{x:.2f} mm<br>Baseline Dose: %{y:.2f} %<extra></extra>",
            )
        )

    cax_shift = float(analysis.get("cax_offset_mm", 0.0))
    fig.add_vline(
        x=cax_shift,
        line=dict(color="#64748b", width=1.5, dash="dash"),
        annotation_text="CAX",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#64748b"),
    )

    # Shaded Inner 80% Evaluation Zone
    eval_region = analysis.get("eval_region", {})
    x_eval_l = eval_region.get("left")
    x_eval_r = eval_region.get("right")
    if x_eval_l is not None and x_eval_r is not None:
        fig.add_vrect(
            x0=x_eval_l,
            x1=x_eval_r,
            fillcolor="rgba(34, 197, 94, 0.08)",
            line=dict(width=1, color="rgba(34, 197, 94, 0.4)", dash="dot"),
            annotation_text="Inner 80% Evaluation Zone",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color="#15803d"),
        )

    # Shaded Penumbra Zones (80% to 20%)
    pen_pts = analysis.get("penumbra_points", {})
    p_l20 = pen_pts.get("left_20")
    p_l80 = pen_pts.get("left_80")
    if p_l20 is not None and p_l80 is not None:
        p_min, p_max = min(p_l20, p_l80), max(p_l20, p_l80)
        fig.add_vrect(
            x0=p_min,
            x1=p_max,
            fillcolor="rgba(249, 115, 22, 0.15)",
            line=dict(width=1, color="rgba(249, 115, 22, 0.6)"),
            annotation_text=f"Left Pen: {analysis.get('penumbra_left_mm', 0):.2f} mm",
            annotation_position="top left",
            annotation_font=dict(size=9, color="#c2410c"),
        )

    p_r80 = pen_pts.get("right_80")
    p_r20 = pen_pts.get("right_20")
    if p_r80 is not None and p_r20 is not None:
        p_min, p_max = min(p_r80, p_r20), max(p_r80, p_r20)
        fig.add_vrect(
            x0=p_min,
            x1=p_max,
            fillcolor="rgba(249, 115, 22, 0.15)",
            line=dict(width=1, color="rgba(249, 115, 22, 0.6)"),
            annotation_text=f"Right Pen: {analysis.get('penumbra_right_mm', 0):.2f} mm",
            annotation_position="top right",
            annotation_font=dict(size=9, color="#c2410c"),
        )

    # Field Edges (50% or Inflection points)
    edges = analysis.get("edges", {})
    x_edge_l = edges.get("left")
    x_edge_r = edges.get("right")
    edge_label = "Inflection" if analysis.get("beam_type") == "FFF" else "FWHM 50%"

    if x_edge_l is not None:
        fig.add_vline(
            x=x_edge_l,
            line=dict(color="#e11d48", width=1.5, dash="dot"),
            annotation_text=f"{edge_label} L ({x_edge_l:.1f} mm)",
            annotation_position="bottom left",
            annotation_font=dict(size=10, color="#be123c"),
        )
    if x_edge_r is not None:
        fig.add_vline(
            x=x_edge_r,
            line=dict(color="#e11d48", width=1.5, dash="dot"),
            annotation_text=f"{edge_label} R ({x_edge_r:.1f} mm)",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color="#be123c"),
        )

    is_dark = st.session_state.get("app_theme_dark", True) if "app_theme_dark" in st.session_state else True
    plot_template = "plotly_dark" if is_dark else "plotly_white"
    grid_col = "rgba(255, 255, 255, 0.08)" if is_dark else "#e2e8f0"
    zero_col = "rgba(255, 255, 255, 0.2)" if is_dark else "#94a3b8"

    fig.update_layout(
        template=plot_template,
        title=f"<b>{beam_title}</b> — Quantitative Overlay",
        xaxis=dict(title="Cross-Beam Distance from Central Axis (mm)", gridcolor=grid_col, zeroline=True, zerolinecolor=zero_col),
        yaxis=dict(title="Relative Dose (%)", gridcolor=grid_col, range=[0, max(115.0, float(np.max(values)) + 5)]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=40, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def _build_folded_symmetry_figure(
    x_mm: np.ndarray,
    values: np.ndarray,
    analysis: Dict[str, Any],
) -> go.Figure:
    """
    Build Folded Profile and Point-Difference Symmetry visualization.
    Folds the left half of the profile onto the right half across the central axis.
    """
    cax_pos = float(analysis.get("cax_offset_mm", 0.0))
    x_rel = x_mm - cax_pos

    # Mask positive and negative sides
    pos_mask = x_rel >= 0
    neg_mask = x_rel <= 0

    x_pos = x_rel[pos_mask]
    y_pos = values[pos_mask]

    x_neg = -x_rel[neg_mask]  # Invert to align with positive side
    y_neg = values[neg_mask]

    # Sort negative side ascending by distance from CAX
    neg_sort = np.argsort(x_neg)
    x_neg = x_neg[neg_sort]
    y_neg = y_neg[neg_sort]

    # Common evaluation range
    max_d = min(float(np.max(x_pos)), float(np.max(x_neg)))
    d_eval = np.linspace(0.5, max_d, 120)

    # Interpolate both halves onto common distance grid
    y_pos_interp = np.interp(d_eval, x_pos, y_pos)
    y_neg_interp = np.interp(d_eval, x_neg, y_neg)

    point_diff_pct = y_pos_interp - y_neg_interp

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("<b>Folded Half-Profiles</b>: Right Side vs Left Side (Mirrored)", "<b>Point-Difference Symmetry</b>: Δ(x) = D(+|x|) − D(−|x|) (%)"),
        row_heights=[0.6, 0.4],
    )

    # Top plot: Folded half-profiles
    fig.add_trace(
        go.Scatter(
            x=d_eval,
            y=y_pos_interp,
            mode="lines",
            name="Right Half (+X)",
            line=dict(color="#0284c7", width=2),
            hovertemplate="Dist: %{x:.2f} mm<br>Dose (+X): %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=d_eval,
            y=y_neg_interp,
            mode="lines",
            name="Left Half (−X folded)",
            line=dict(color="#e11d48", width=2, dash="dash"),
            hovertemplate="Dist: %{x:.2f} mm<br>Dose (−X): %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Bottom plot: Point difference
    fig.add_trace(
        go.Scatter(
            x=d_eval,
            y=point_diff_pct,
            mode="lines",
            name="Δ(x) Difference",
            line=dict(color="#7c3aed", width=1.8),
            fill="tozeroy",
            fillcolor="rgba(124, 58, 237, 0.12)",
            hovertemplate="Dist: %{x:.2f} mm<br>Δ: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Tolerance bands on point difference
    fig.add_hline(y=1.5, line=dict(color="#f59e0b", width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=-1.5, line=dict(color="#f59e0b", width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=0.0, line=dict(color="#94a3b8", width=1), row=2, col=1)

    is_dark = st.session_state.get("app_theme_dark", True) if "app_theme_dark" in st.session_state else True
    plot_template = "plotly_dark" if is_dark else "plotly_white"
    grid_col = "rgba(255, 255, 255, 0.08)" if is_dark else "#e2e8f0"

    fig.update_layout(
        template=plot_template,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Off-Axis Radial Distance |x| (mm)", row=2, col=1, gridcolor=grid_col)
    fig.update_yaxes(title_text="Relative Dose (%)", row=1, col=1, gridcolor=grid_col)
    fig.update_yaxes(title_text="Dose Diff (%)", row=2, col=1, gridcolor=grid_col, range=[-4.0, 4.0])

    return fig


def _build_2d_fluence_figure(image_2d: np.ndarray, pixel_spacing_iso: Tuple[float, float]) -> go.Figure:
    """Build interactive 2D EPID Fluence heatmap with scaled physical isocenter millimeter axes."""
    dy_iso, dx_iso = pixel_spacing_iso
    h, w = image_2d.shape

    cy_idx = h // 2
    cx_idx = w // 2

    x_iso = (np.arange(w) - cx_idx) * dx_iso
    y_iso = (np.arange(h) - cy_idx) * dy_iso

    fig = go.Figure(
        data=go.Heatmap(
            z=image_2d,
            x=x_iso,
            y=y_iso,
            colorscale="Viridis",
            colorbar=dict(title="Fluence"),
            hovertemplate="X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>Signal: %{z:.1f}<extra></extra>",
        )
    )

    # Crosshair at CAX
    fig.add_vline(x=0.0, line=dict(color="rgba(255, 255, 255, 0.7)", width=1, dash="dash"))
    fig.add_hline(y=0.0, line=dict(color="rgba(255, 255, 255, 0.7)", width=1, dash="dash"))

    is_dark = st.session_state.get("app_theme_dark", True) if "app_theme_dark" in st.session_state else True
    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        title="<b>2D Calibrated EPID Fluence Map</b> (Scaled to Isocenter)",
        xaxis=dict(title="Crossplane X (mm)", zeroline=False),
        yaxis=dict(title="Inplane Y (mm)", scaleanchor="x", scaleratio=1, zeroline=False),
        margin=dict(l=60, r=40, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# =============================================================================
# Main Streamlit Page Renderer
# =============================================================================

def render_beam_profile_page() -> None:
    """Main rendering entry-point for the Beam Profile QA Streamlit module."""
    st.markdown('<div class="section-header">📐 Beam Profile Flatness, Symmetry & Penumbra Analyzer</div>', unsafe_allow_html=True)
    st.caption(
        "Quantitative profile QA for conventional flattened and Flattening Filter Free (FFF) photon beams. "
        "Supports 1D/2D Water Tank scans, IC Profiler text files, and DICOM EPID (.dcm) portal dosimetry."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 0. Reference Beam Data & Baselines Manager
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("🏛️ Linac Reference Beam Data & PDD Manager (Golden Baselines)", expanded=False):
        st.markdown(
            "Upload and store measured golden reference profiles and PDD curves for specific machines "
            "(**TrueBeam, Edge, Halcyon**) into the database. Stored PDDs automatically calculate **$TPR_{20,10}$** "
            "for IAEA TRS-398 reference dosimetry."
        )

        ref_tab_upload, ref_tab_view = st.tabs(["📤 Upload Reference Baseline", "📋 View Stored Baselines"])

        with ref_tab_upload:
            u_col1, u_col2, u_col3 = st.columns(3)
            with u_col1:
                # Load available units from DB
                db_units = crud.get_units()
                unit_names = db_units["name"].tolist() if not db_units.empty else ["TrueBeam", "Edge", "Halcyon"]
                ref_unit = st.selectbox("Linac / Unit", unit_names, key="ref_upload_unit")
            with u_col2:
                ref_energy = st.selectbox("Beam Energy", ["6 MV", "6 MV FFF", "10 MV", "10 MV FFF", "15 MV", "18 MV"], key="ref_upload_energy")
            with u_col3:
                ref_field = st.selectbox("Field Size", ["10x10 cm", "28x28 cm", "40x40 cm", "5x5 cm", "30x30 cm"], key="ref_upload_field")

            u_col4, u_col5 = st.columns(2)
            with u_col4:
                ref_curve_type = st.selectbox("Curve Type", ["Auto-Detect from File", "PDD (Depth Dose)", "Crossplane Profile (X)", "Inplane Profile (Y)"], key="ref_upload_curve_type")
            with u_col5:
                ref_depth_val = st.number_input("Measurement Depth (mm)", value=100.0, step=10.0, key="ref_upload_depth")

            ref_file = st.file_uploader(
                "Upload Reference Scan (Varian Eclipse W2CAD, PTW BeamScan .mcc, CSV, TSV, ASCII, or Excel .xlsx)",
                type=None,
                help="Accepts files with or without extensions.",
                key="ref_beam_uploader",
            )

            if ref_file is not None:
                try:
                    ref_bytes = ref_file.getvalue()
                    ref_name = ref_file.name
                    parsed_ref = bpa.parse_profile_text(ref_bytes)

                    # Determine curve candidates
                    candidate_profiles = {}
                    if "profiles" in parsed_ref and parsed_ref["profiles"]:
                        candidate_profiles = parsed_ref["profiles"]
                    else:
                        candidate_profiles = {"Uploaded Curve": parsed_ref}

                    sel_candidate_key = st.selectbox("Select Curve from File to Store", list(candidate_profiles.keys()), key="ref_sel_candidate")
                    target_curve = candidate_profiles[sel_candidate_key]
                    c_x = np.asarray(target_curve["x_mm"], dtype=float)
                    c_y = np.asarray(target_curve["values"], dtype=float)

                    # Determine if PDD or Profile
                    is_pdd = "PDD" in sel_candidate_key.upper() or "PDD" in ref_curve_type.upper() or (c_x.min() >= 0.0 and c_x.max() > 150.0 and "PROFILE" not in sel_candidate_key.upper())
                    is_fff = "FFF" in ref_energy.upper() or target_curve.get("meta", {}).get("is_fff", False)

                    st.markdown("---")
                    st.markdown(f"**Previewing: `{sel_candidate_key}` ({len(c_x)} points)**")

                    # Preview plot
                    prev_fig = go.Figure()
                    prev_fig.add_trace(go.Scatter(x=c_x, y=c_y, mode="lines", name=sel_candidate_key, line=dict(color="#0284c7", width=2)))
                    prev_fig.update_layout(
                        title=f"{ref_unit} {ref_energy} — {sel_candidate_key}",
                        xaxis_title="Depth (mm)" if is_pdd else "Off-Axis Position (mm)",
                        yaxis_title="Dose / Reading (%)",
                        height=280,
                        margin=dict(l=40, r=20, t=35, b=35),
                    )
                    st.plotly_chart(prev_fig, use_container_width=True)

                    pdd_metrics = {}
                    prof_metrics = {}

                    if is_pdd:
                        pdd_analysis = bpa.analyze_pdd_curve(c_x, c_y)
                        pdd_metrics = pdd_analysis
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("d_max", f"{pdd_analysis['d_max_mm']:.1f} mm")
                        m2.metric("PDD(10 cm)", f"{pdd_analysis['pdd_10']:.2f}%")
                        m3.metric("PDD(20 cm)", f"{pdd_analysis['pdd_20']:.2f}%")
                        m4.metric("Calculated TPR₂₀,₁₀", f"{pdd_analysis['tpr20_10']:.4f}")
                        st.info(f"💡 This PDD curve will automatically supply $TPR_{{20,10}} = {pdd_analysis['tpr20_10']:.4f}$ and $PDD_{{10}} = {pdd_analysis['pdd_10']:.1f}\\%$ to the IAEA TRS-398 module for **{ref_unit} {ref_energy}**.")
                    else:
                        prof_analysis = bpa.analyze_beam_profile(c_x, c_y, beam_type="FFF" if is_fff else "Flattened")
                        prof_metrics = prof_analysis
                        p1, p2, p3, p4 = st.columns(4)
                        p1.metric("Field Width", f"{prof_analysis['field_width_mm']:.2f} mm")
                        p2.metric("CAX Offset", f"{prof_analysis['cax_offset_mm']:.2f} mm")
                        p3.metric("Penumbra (L/R)", f"{prof_analysis['penumbra_left_mm']:.1f} / {prof_analysis['penumbra_right_mm']:.1f} mm")
                        p4.metric("Symmetry", f"{prof_analysis['symmetry_pct']:.2f}%")

                    ref_notes = st.text_input("Baseline Notes / Protocol Reference", value=f"Golden baseline commissioned for {ref_unit}", key="ref_notes_input")

                    if st.button("💾 Save as Machine Golden Baseline in Database", type="primary", key="ref_save_btn"):
                        curve_payload = json.dumps({"x_mm": [round(float(v), 2) for v in c_x], "values": [round(float(v), 4) for v in c_y]})
                        c_type_label = "PDD" if is_pdd else ("Crossplane Profile" if "Cross" in sel_candidate_key else "Inplane Profile")
                        crud.save_reference_beam_data(
                            unit_name=ref_unit,
                            energy=ref_energy,
                            beam_type="FFF" if is_fff else "Flattened",
                            curve_type=c_type_label,
                            field_size_nominal=ref_field,
                            curve_data_json=curve_payload,
                            depth_mm=ref_depth_val,
                            tpr20_10=pdd_metrics.get("tpr20_10"),
                            pdd_10=pdd_metrics.get("pdd_10"),
                            pdd_20=pdd_metrics.get("pdd_20"),
                            d_max_mm=pdd_metrics.get("d_max_mm"),
                            field_width_mm=prof_metrics.get("field_width_mm"),
                            cax_offset_mm=prof_metrics.get("cax_offset_mm"),
                            penumbra_left_mm=prof_metrics.get("penumbra_left_mm"),
                            penumbra_right_mm=prof_metrics.get("penumbra_right_mm"),
                            flatness_pct=prof_metrics.get("flatness_pct"),
                            symmetry_pct=prof_metrics.get("symmetry_pct"),
                            source_filename=ref_name,
                            notes=ref_notes,
                        )
                        st.success(f"✅ Successfully saved reference baseline `{c_type_label}` for **{ref_unit} {ref_energy} ({ref_field})** to database!")
                        st.rerun()

                except Exception as exc:
                    st.error(f"Failed to process reference scan: {exc}")

        with ref_tab_view:
            ref_df = crud.get_reference_beam_data()
            if ref_df.empty:
                st.info("No reference baselines have been stored yet. Upload a curve above to save your first golden baseline.")
            else:
                st.dataframe(ref_df[["id", "unit_name", "energy", "beam_type", "curve_type", "field_size_nominal", "tpr20_10", "pdd_10", "field_width_mm", "uploaded_at"]], use_container_width=True)
                del_c1, del_c2 = st.columns([3, 1])
                with del_c1:
                    del_id = st.selectbox("Select Baseline ID to Remove", ref_df["id"].tolist(), format_func=lambda i: f"ID {i} - {ref_df.loc[ref_df['id'] == i, 'unit_name'].values[0]} {ref_df.loc[ref_df['id'] == i, 'energy'].values[0]} ({ref_df.loc[ref_df['id'] == i, 'curve_type'].values[0]})", key="ref_del_sel")
                with del_c2:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ Delete Baseline", key="ref_del_btn"):
                        crud.delete_reference_beam_data(del_id)
                        st.warning(f"Deleted baseline ID {del_id}.")
                        st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Data Source Selection
    # ─────────────────────────────────────────────────────────────────────────
    source_col1, source_col2 = st.columns([1.5, 2.5])
    with source_col1:
        data_mode = st.radio(
            "Select Profile Source",
            [
                "Built-in Synthetic Demos",
                "Upload File (MCC, CSV, TSV, ASCII, or DICOM .dcm)",
                "🏛️ Stored Linac Baseline (Database)",
            ],
            key="bp_data_source_mode",
        )

    profile_dict: Optional[Dict[str, Any]] = None
    multi_profiles: Dict[str, Any] = {}
    dicom_payload: Optional[Dict[str, Any]] = None

    if data_mode == "Built-in Synthetic Demos":
        with source_col2:
            demo_choice = st.selectbox(
                "Choose Benchmark Demo Profile",
                [
                    "6 MV Flattened (10x10 cm, perfect)",
                    "6 MV Flattened (10x10 cm, 2.5% horns)",
                    "6 MV FFF (10x10 cm)",
                    "10 MV FFF (10x10 cm)",
                ],
                key="bp_demo_preset",
            )

        if demo_choice == "6 MV Flattened (10x10 cm, perfect)":
            profile_dict = bpa.generate_synthetic_profile("Flattened", energy_mv=6.0, field_size_mm=100.0, horn_ratio=0.0)
        elif demo_choice == "6 MV Flattened (10x10 cm, 2.5% horns)":
            profile_dict = bpa.generate_synthetic_profile("Flattened", energy_mv=6.0, field_size_mm=100.0, horn_ratio=0.025)
        elif demo_choice == "6 MV FFF (10x10 cm)":
            profile_dict = bpa.generate_synthetic_profile("FFF", energy_mv=6.0, field_size_mm=100.0)
        elif demo_choice == "10 MV FFF (10x10 cm)":
            profile_dict = bpa.generate_synthetic_profile("FFF", energy_mv=10.0, field_size_mm=100.0)

    elif data_mode == "🏛️ Stored Linac Baseline (Database)":
        with source_col2:
            all_refs = crud.get_reference_beam_data()
            prof_refs = all_refs[all_refs["curve_type"].str.contains("Profile", case=False, na=False)] if not all_refs.empty else pd.DataFrame()
            if prof_refs.empty:
                st.warning("⚠️ No reference profiles stored in database yet. Use the '🏛️ Linac Reference Beam Data & PDD Manager' above to upload one.")
                return
            ref_choice_id = st.selectbox(
                "Choose Stored Linac Baseline",
                prof_refs["id"].tolist(),
                format_func=lambda i: f"{prof_refs.loc[prof_refs['id'] == i, 'unit_name'].values[0]} — {prof_refs.loc[prof_refs['id'] == i, 'energy'].values[0]} ({prof_refs.loc[prof_refs['id'] == i, 'field_size_nominal'].values[0]} {prof_refs.loc[prof_refs['id'] == i, 'curve_type'].values[0]})",
                key="bp_ref_db_choice",
            )
            raw_rec = crud.get_reference_beam_by_id(ref_choice_id)
            if raw_rec and raw_rec.get("curve_data_json"):
                parsed_c = json.loads(raw_rec["curve_data_json"])
                profile_dict = {
                    "x_mm": np.array(parsed_c["x_mm"]),
                    "values": np.array(parsed_c["values"]),
                    "meta": {
                        "unit_name": raw_rec["unit_name"],
                        "energy": raw_rec["energy"],
                        "beam_type": raw_rec["beam_type"],
                        "field_size_nominal": raw_rec["field_size_nominal"],
                        "axis": raw_rec["curve_type"],
                        "data_source": "Database Reference Baseline",
                    },
                }

    else:
        # File Upload Mode
        with source_col2:
            uploaded_file = st.file_uploader(
                "Upload Profile (Varian W2CAD, PTW BeamScan .mcc, CSV, TSV, ASCII, or DICOM .dcm)",
                type=None,
                help="Accepts files with or without extensions.",
                key="bp_file_uploader",
            )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name

            # Check if DICOM
            is_dcm = file_name.lower().endswith(".dcm") or (len(file_bytes) > 132 and file_bytes[128:132] == b"DICM")
            if is_dcm:
                try:
                    dicom_payload = bpa.parse_dicom_epid(file_bytes)
                    multi_profiles = {
                        "Crossplane (X)": dicom_payload["crossplane"],
                        "Inplane (Y)": dicom_payload["inplane"],
                    }
                    sel_axis = st.radio("Extracted EPID Axis", ["Crossplane (X)", "Inplane (Y)"], horizontal=True, key="bp_dicom_axis")
                    chosen_prof = multi_profiles[sel_axis]
                    profile_dict = {
                        "x_mm": chosen_prof["x_mm"],
                        "values": chosen_prof["values"],
                        "meta": dicom_payload["meta"],
                    }
                    profile_dict["meta"]["axis"] = sel_axis
                    st.success(f"✅ DICOM EPID parsed: {dicom_payload['meta']['image_shape'][1]}×{dicom_payload['meta']['image_shape'][0]} pixels at isocenter.")
                except Exception as exc:
                    st.error(f"Failed to parse DICOM EPID: {exc}")
                    return
            else:
                try:
                    parsed_txt = bpa.parse_profile_text(file_bytes)
                    if "profiles" in parsed_txt and parsed_txt["profiles"]:
                        axis_keys = list(parsed_txt["profiles"].keys())
                        sel_axis = st.radio("Select Profile Axis / Scan", axis_keys, horizontal=True, key="bp_text_axis")
                        profile_dict = parsed_txt["profiles"][sel_axis]
                    else:
                        profile_dict = parsed_txt
                    st.success(f"✅ Parsed {len(profile_dict['x_mm'])} coordinate-dose points from `{file_name}`.")
                except Exception as exc:
                    st.error(f"Failed to parse ASCII/CSV profile text: {exc}")
                    return
        else:
            st.info("👆 Please upload a profile file, select a 'Stored Linac Baseline', or select 'Built-in Synthetic Demos' to begin.")
            return

    if profile_dict is None:
        return

    x_mm = np.asarray(profile_dict["x_mm"], dtype=float)
    values = np.asarray(profile_dict["values"], dtype=float)
    meta = profile_dict.get("meta", {})

    # Detect or override beam mode and protocol
    detected_beam_type = meta.get("beam_type", "Flattened")
    is_fff_detected = "FFF" in str(detected_beam_type).upper() or "FFF" in str(meta.get("energy", "")).upper()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Protocol Configuration Toolbar
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    cfg_c1, cfg_c2, cfg_c3, cfg_c4 = st.columns(4)

    with cfg_c1:
        beam_type_choice = st.selectbox(
            "Beam Delivery Mode",
            ["Flattened", "FFF (Flattening Filter Free)"],
            index=1 if is_fff_detected else 0,
            key="bp_beam_type_choice",
        )
    is_fff = beam_type_choice.startswith("FFF")

    with cfg_c2:
        if is_fff:
            protocol_choice = st.selectbox(
                "Evaluation Protocol",
                ["FFF Inflection & Peak"],
                key="bp_protocol_choice_fff",
                disabled=True,
            )
            protocol = "FFF Inflection"
        else:
            protocol_choice = st.selectbox(
                "Evaluation Protocol",
                ["IEC 60976 (Dmax/Dmin)", "Varian (Diff Flatness & Symmetry)"],
                key="bp_protocol_choice_flat",
            )
            protocol = "Varian" if "Varian" in protocol_choice else "IEC"

    with cfg_c3:
        energy_input = st.text_input("Nominal Energy", value=meta.get("energy", "6 MV"), key="bp_energy_input")

    with cfg_c4:
        axis_input = st.text_input("Profile Axis", value=meta.get("axis", "Crossplane (X)"), key="bp_axis_input")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Execution of Scientific Numerical Analysis Engine
    # ─────────────────────────────────────────────────────────────────────────
    try:
        if is_fff:
            res = bpa.analyze_fff_profile(x_mm, values)
        else:
            res = bpa.analyze_flattened_profile(x_mm, values, protocol=protocol)
    except Exception as exc:
        st.error(f"Error during beam profile analysis: {exc}")
        return

    st.session_state["bp_last_analysis"] = res

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Numerical Metrics Display Dashboard
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Quantitative Beam Profile Metrics")
    _render_status_badge(res.get("status", "OK"))

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        _render_metric_card("Field Width", f"{res.get('field_width_mm', 0):.2f} mm", subtext="FWHM / Inflection")
    with m2:
        _render_metric_card("Left Penumbra (80-20)", f"{res.get('penumbra_left_mm', 0):.2f} mm", subtext="Tol: ≤ 7.0 mm")
    with m3:
        _render_metric_card("Right Penumbra (80-20)", f"{res.get('penumbra_right_mm', 0):.2f} mm", subtext="Tol: ≤ 7.0 mm")
    with m4:
        if is_fff:
            _render_metric_card("Flatness", "N/A (FFF)", subtext="Unflattened beam")
        else:
            flat_val = res.get("flatness_pct", 0.0)
            flat_label = "Flatness (IEC)" if protocol == "IEC" else "Flatness (Varian)"
            _render_metric_card(flat_label, f"{flat_val:.2f} %", subtext="Tol: ≤ 2.0%")
    with m5:
        sym_val = res.get("symmetry_pct", 0.0)
        _render_metric_card("Symmetry", f"{sym_val:.2f} %", subtext="Tol: ≤ 2.0%")
    with m6:
        cax_off = res.get("cax_offset_mm", 0.0)
        _render_metric_card("CAX Center Shift", f"{cax_off:+.2f} mm", subtext="Tol: ±0.5 mm")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Interactive Visual Suite Tabs
    # ─────────────────────────────────────────────────────────────────────────
    tab_titles = ["📈 1D Profile Overlay", "🔄 Folded Symmetry", "📜 Protocol Summary"]
    if dicom_payload is not None:
        tab_titles.insert(2, "🗺️ 2D Fluence Heatmap")

    tabs = st.tabs(tab_titles)

    # Tab: 1D Profile
    with tabs[0]:
        title_str = f"{energy_input} {axis_input} Profile ({protocol})"
        all_refs = crud.get_reference_beam_data()
        prof_refs = all_refs[all_refs["curve_type"].str.contains("Profile", case=False, na=False)] if not all_refs.empty else pd.DataFrame()
        baseline_x, baseline_y, baseline_name = None, None, "Golden Baseline"
        if not prof_refs.empty:
            c_base1, c_base2 = st.columns([1, 2])
            with c_base1:
                use_overlay = st.checkbox("🔍 Overlay Golden Baseline", key="bp_use_baseline_overlay")
            with c_base2:
                if use_overlay:
                    base_sel_id = st.selectbox(
                        "Select Golden Baseline to Compare",
                        prof_refs["id"].tolist(),
                        format_func=lambda i: f"{prof_refs.loc[prof_refs['id'] == i, 'unit_name'].values[0]} {prof_refs.loc[prof_refs['id'] == i, 'energy'].values[0]} ({prof_refs.loc[prof_refs['id'] == i, 'curve_type'].values[0]})",
                        key="bp_overlay_choice_id"
                    )
                    b_rec = crud.get_reference_beam_by_id(base_sel_id)
                    if b_rec and b_rec.get("curve_data_json"):
                        b_parsed = json.loads(b_rec["curve_data_json"])
                        baseline_x = np.array(b_parsed["x_mm"])
                        baseline_y = np.array(b_parsed["values"])
                        baseline_name = f"Baseline ({b_rec['unit_name']} {b_rec['energy']})"

        fig_1d = _build_1d_profile_figure(
            x_mm, values, res, beam_title=title_str,
            baseline_x=baseline_x, baseline_y=baseline_y, baseline_name=baseline_name
        )
        st.plotly_chart(fig_1d, use_container_width=True)

    # Tab: Folded Symmetry
    with tabs[1]:
        fig_folded = _build_folded_symmetry_figure(x_mm, values, res)
        st.plotly_chart(fig_folded, use_container_width=True)

    # Tab: 2D Fluence (if DICOM)
    tab_idx = 2
    if dicom_payload is not None:
        with tabs[tab_idx]:
            fig_2d = _build_2d_fluence_figure(dicom_payload["image_2d"], dicom_payload["pixel_spacing_iso"])
            st.plotly_chart(fig_2d, use_container_width=True)
        tab_idx += 1

    # Tab: Detailed Protocol Summary
    with tabs[tab_idx]:
        st.markdown("#### Full Algorithmic Output Breakdown")
        df_summary = pd.DataFrame(
            [
                {"Metric": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                for k, v in res.items()
                if not isinstance(v, dict)
            ]
        )
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. One-Click SQLite Logging & Clinical PDF Report Download
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Documentation, Database Logging & Clinical Certification")

    log_col1, log_col2 = st.columns([1.5, 1])

    # Available LINAC units from SQLite DB
    units_df = crud.get_units()
    linac_units = units_df[units_df["modality"] == "LINAC"] if not units_df.empty else pd.DataFrame()
    unit_names = linac_units["name"].tolist() if not linac_units.empty else (units_df["name"].tolist() if not units_df.empty else ["Varian TrueBeam #1"])

    with log_col1:
        st.markdown("##### 🗄️ Save Result to Clinic Database (`qa_data.db`)")
        with st.form("bp_db_log_form"):
            f_c1, f_c2 = st.columns(2)
            with f_c1:
                sel_unit = st.selectbox("Treatment Unit / Linac", unit_names, key="bp_sel_unit")
                physicist = st.text_input("Medical Physicist", value="Dr. Nemanja Golubovac, PhD", key="bp_physicist_name")
            with f_c2:
                meas_date = st.date_input("QA Measurement Date", value=date.today(), key="bp_meas_date")
                log_notes = st.text_input("Clinical Notes / Comments", value="Routine beam profile constancy check.", key="bp_log_notes")

            submit_log = st.form_submit_button("💾 Save Profile QA to qa_data.db", type="primary", use_container_width=True)

        if submit_log:
            try:
                # Resolve unit_id
                matched_rows = units_df.loc[units_df["name"] == sel_unit, "id"].values if not units_df.empty else []
                uid = int(matched_rows[0]) if len(matched_rows) > 0 else 1

                # 1. Log to comprehensive profile_qa_logs table
                analyzer_obj = bpa.BeamProfileAnalyzer(x_mm, values, meta=meta)
                analyzer_obj.results = res
                raw_json = analyzer_obj.to_json()

                field_w = float(res.get("field_width_mm", 100.0))
                cax_off = float(res.get("cax_offset_mm", 0.0))
                pen_l = float(res.get("penumbra_left_mm", 6.0))
                pen_r = float(res.get("penumbra_right_mm", 6.0))
                flat_val = float(res.get("flatness_pct")) if res.get("flatness_pct") is not None else None
                sym_val = float(res.get("symmetry_pct", 0.0))
                sym_area = float(res.get("symmetry_area_pct", 0.0))
                l_slope = float(res.get("left_slope_pct_mm")) if res.get("left_slope_pct_mm") is not None else None
                r_slope = float(res.get("right_slope_pct_mm")) if res.get("right_slope_pct_mm") is not None else None

                log_id = crud.save_profile_qa_log(
                    unit_name=sel_unit,
                    unit_id=uid,
                    institution="Department of Radiation Oncology",
                    physicist=physicist,
                    beam_energy=energy_input,
                    beam_type="FFF" if is_fff else "Flattened",
                    protocol=protocol,
                    data_source=data_mode,
                    axis=axis_input,
                    field_size_nominal=meta.get("field_size_nominal", "10x10 cm"),
                    field_width_x_mm=field_w if "X" in axis_input else None,
                    field_width_y_mm=field_w if "Y" in axis_input else None,
                    cax_offset_x_mm=cax_off if "X" in axis_input else None,
                    cax_offset_y_mm=cax_off if "Y" in axis_input else None,
                    penumbra_left_x_mm=pen_l if "X" in axis_input else None,
                    penumbra_right_x_mm=pen_r if "X" in axis_input else None,
                    penumbra_top_y_mm=pen_l if "Y" in axis_input else None,
                    penumbra_bottom_y_mm=pen_r if "Y" in axis_input else None,
                    flatness_x=flat_val if "X" in axis_input else None,
                    flatness_y=flat_val if "Y" in axis_input else None,
                    symmetry_x=sym_val if "X" in axis_input else None,
                    symmetry_y=sym_val if "Y" in axis_input else None,
                    symmetry_area_x=sym_area if "X" in axis_input else None,
                    symmetry_area_y=sym_area if "Y" in axis_input else None,
                    left_slope_x=l_slope,
                    right_slope_x=r_slope,
                    status=res.get("status", "OK"),
                    notes=log_notes,
                    raw_summary_json=raw_json,
                    performed_at=meas_date.isoformat(),
                )

                # 2. Also log individual control chart test results if corresponding tests exist
                tests_to_log = [
                    ("Beam Profile Flatness", flat_val),
                    ("Beam Profile Symmetry", sym_val),
                    ("Beam Profile Penumbra", float(res.get("penumbra_mean_mm", (pen_l + pen_r) / 2.0))),
                ]
                for tname, val in tests_to_log:
                    if val is not None:
                        t = crud.get_test_by_name(tname)
                        if t is not None:
                            tid = int(t["id"])
                            crud.assign_test_to_unit(uid, tid)
                            if hasattr(crud, "save_test_result"):
                                crud.save_test_result(uid, tid, val, performed_by=physicist, comment=log_notes)
                            else:
                                crud.log_result(uid, tid, val, performed_by=physicist, comment=log_notes)

                st.success(f"✅ Successfully committed Beam Profile QA Log #{log_id} to `qa_data.db` for **{sel_unit}**.")
            except Exception as db_err:
                st.error(f"Failed to save to database: {db_err}")

    with log_col2:
        st.markdown("##### 📄 Print-Ready Clinical PDF Report")
        st.caption("Generate an IAEA / AAPM compliant clinical PDF QA certificate with embedded 300 DPI high-resolution figures.")

        # Prepare payload for clinical PDF generator
        pdf_profile_results = {
            "is_fff": is_fff,
            "beam_energy": energy_input,
            "field_width_mm": float(res.get("field_width_mm", 100.0)),
            "flatness_pct": float(res.get("flatness_pct")) if res.get("flatness_pct") is not None else 0.0,
            "symmetry_pct": float(res.get("symmetry_pct", 0.0)),
            "penumbra_left_mm": float(res.get("penumbra_left_mm", 6.0)),
            "penumbra_right_mm": float(res.get("penumbra_right_mm", 6.0)),
            "cax_shift_mm": float(res.get("cax_offset_mm", 0.0)),
            "x_mm": x_mm,
            "values": values,
            "axis": axis_input,
            "data_source": data_mode,
            "protocol": protocol,
        }

        unit_meta = {
            "institution": "University Clinical Medical Physics Center",
            "unit_name": sel_unit,
            "modality": "Medical Linear Accelerator (LINAC)",
            "physicist": physicist,
            "reviewer": "Chief Clinical Medical Physicist",
            "energy": energy_input,
            "protocol_edition": f"IEC 60976 / Varian ({protocol})",
        }

        try:
            pdf_bytes = generate_beam_profile_pdf(pdf_profile_results, unit_meta)
            st.download_button(
                label="📄 Download Clinical PDF Certificate",
                data=pdf_bytes,
                file_name=f"Beam_Profile_QA_{sel_unit.replace(' ', '_')}_{meas_date.strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                key="bp_download_pdf",
                use_container_width=True,
            )
        except Exception as pdf_err:
            st.warning(f"Could not prepare PDF download: {pdf_err}")


# Standalone alias
page_beam_profile = render_beam_profile_page
