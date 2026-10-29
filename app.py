"""
app.py — Micro QA
============================================================
Main Streamlit application.

Navigation (sidebar):
  1. 🏥 Dashboard
  2. 📋 QA Data Entry
  3. 📈 Trending & Reports
  4. 🔧 Service & Downtime Log
  5. ⚙️ Admin / Settings

Run with:
    streamlit run app.py
"""

import os
import io
import re
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mpl_cache"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date

import json
import numpy as np

# ── Initialise the database on first run ─────────────────────────────────────
from db_setup import init_db, seed_db
from crud import (
    get_units, get_unit_by_id, get_unit_by_name,
    get_tests, get_tests_for_unit, get_test_by_id,
    add_unit, update_unit, add_test, assign_test_to_unit,
    log_result, get_historical_data, get_recent_results,
    add_service_log, get_service_logs,
    get_status_summary,
    add_ion_chamber, get_ion_chambers, delete_ion_chamber,
    add_unit_energy, get_unit_energies, delete_unit_energy,
    save_reference_beam_data, get_reference_beam_data, delete_reference_beam_data, get_reference_beam_by_id,
)

# Chamber & Certificate Parser
import chamber_manager as cm
import beam_profile_analyzer as bpa

# TRS-398 Reference Calibration (Enhanced Dual Engine: Pure-Python + Pylinac)
from trs398_page import page_trs398

# Automated Pylinac QA Suite Modules (Trajectory Logs, Picket Fence, Winston-Lutz, CBCT CatPhan)
from linac_qa_modules import (
    page_trajectory_logs,
    page_picket_fence,
    page_winston_lutz,
    page_cbct_catphan,
)

# Beam Profile Flatness, Symmetry & Penumbra Analyzer
from beam_profile_page import render_beam_profile_page

init_db()
seed_db()


# ===========================================================================
# Page Configuration
# ===========================================================================
st.set_page_config(
    page_title="Micro QA",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# Sidebar Navigation & Theme Control
# ===========================================================================
with st.sidebar:
    st.markdown("## ⚕️ Micro QA Suite")

    # Dark / Light Mode Toggle
    is_dark = st.toggle("🌙 Dark Mode", value=True, key="app_theme_dark")

    st.markdown("---")
    page = st.radio(
        "Navigation",
        options=[
            "🏥 Dashboard",
            "📋 QA Data Entry",
            "📈 Trending & Reports",
            "☢️ TRS-398 Calibration",
            "📐 Beam Profile QA",
            "📊 Trajectory Logs",
            "🔲 Picket Fence QA",
            "🎯 Winston-Lutz QA",
            "🔬 CBCT (CatPhan) QA",
            "🔧 Service & Downtime Log",
            "⚙️ Equipment & Admin",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("v2.2.0 — Full Linac & TRS-398 QA Suite")


# ===========================================================================
# Dynamic Theme CSS (Dark / Light)
# ===========================================================================
if is_dark:
    theme_css = """
    <style>
        /* ── Global Typography & Dark Base ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp {
            background-color: #0b1329 !important;
            color: #f1f5f9 !important;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #070d1d 0%, #0f172a 100%) !important;
            border-right: 1px solid #1e293b !important;
        }
        [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
        [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; padding: 4px 0; }

        /* ── Metric Cards ── */
        [data-testid="metric-container"] {
            background: #131f37 !important;
            border: 1px solid #233554 !important;
            border-left: 4px solid #38bdf8 !important;
            border-radius: 10px;
            padding: 16px 20px;
        }
        [data-testid="stMetricValue"] {
            color: #f8fafc !important;
        }
        [data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
        }

        /* ── Status Badges ── */
        .badge-ok       { color:#fff; background:#059669; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }
        .badge-tol      { color:#fff; background:#d97706; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }
        .badge-action   { color:#fff; background:#dc2626; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }

        /* ── Section Headers ── */
        .section-header {
            font-size: 1.5rem; font-weight: 700; color: #38bdf8 !important;
            margin-bottom: 1rem; border-bottom: 2px solid #233554 !important; padding-bottom: 6px;
        }

        /* ── Data Tables ── */
        .stDataFrame { border-radius: 8px; overflow: hidden; }

        /* ── Form Inputs & Textareas ── */
        textarea, input[type="text"], input[type="number"] {
            border-radius: 6px !important;
            background-color: #131f37 !important;
            border: 1px solid #233554 !important;
            color: #f8fafc !important;
        }

        /* ── Status Alert Boxes ── */
        .status-box {
            padding: 12px 18px; border-radius: 8px; font-size: 1rem;
            font-weight: 600; margin-top: 12px;
        }
        .status-ok     { background:#064e3b !important; color:#a7f3d0 !important; border:1px solid #059669 !important; }
        .status-tol    { background:#78350f !important; color:#fde68a !important; border:1px solid #d97706 !important; }
        .status-action { background:#7f1d1d !important; color:#fecaca !important; border:1px solid #dc2626 !important; }

        /* ── Expanders ── */
        div[data-testid="stExpander"] {
            background-color: #131f37 !important;
            border: 1px solid #233554 !important;
            border-radius: 8px !important;
        }

        /* ── Tabs ── */
        button[data-baseweb="tab"] {
            color: #94a3b8 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #38bdf8 !important;
            border-bottom-color: #38bdf8 !important;
        }
    </style>
    """
else:
    theme_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 100%);
        }
        [data-testid="stSidebar"] * { color: #e0e6f0 !important; }
        [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; padding: 4px 0; }

        [data-testid="metric-container"] {
            background: #f0f4f8;
            border-radius: 10px;
            padding: 16px 20px;
            border-left: 4px solid #415a77;
        }

        .badge-ok       { color:#fff; background:#2d9e5f; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }
        .badge-tol      { color:#fff; background:#e58b00; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }
        .badge-action   { color:#fff; background:#c0392b; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }

        .section-header {
            font-size: 1.5rem; font-weight: 700; color: #1b263b;
            margin-bottom: 1rem; border-bottom: 2px solid #415a77; padding-bottom: 6px;
        }

        .stDataFrame { border-radius: 8px; overflow: hidden; }
        textarea, input[type="text"], input[type="number"] { border-radius: 6px !important; }

        .status-box {
            padding: 12px 18px; border-radius: 8px; font-size: 1rem;
            font-weight: 600; margin-top: 12px;
        }
        .status-ok     { background:#d4edda; color:#155724; border:1px solid #c3e6cb; }
        .status-tol    { background:#fff3cd; color:#856404; border:1px solid #ffc107; }
        .status-action { background:#f8d7da; color:#721c24; border:1px solid #f5c6cb; }
    </style>
    """
st.markdown(theme_css, unsafe_allow_html=True)



# ===========================================================================
# Helpers
# ===========================================================================

def status_badge_html(status: str) -> str:
    """Return an HTML badge string for a given status."""
    cls = {"OK": "badge-ok", "Tolerance": "badge-tol", "Action": "badge-action"}.get(status, "badge-ok")
    return f'<span class="{cls}">{status}</span>'


def status_box_html(status: str, value: float, unit: str) -> str:
    """Return a coloured status announcement box."""
    messages = {
        "OK":        f"✅ Value {value:.3f} {unit} is within tolerance. All clear.",
        "Tolerance": f"⚠️  Value {value:.3f} {unit} is outside tolerance — review required.",
        "Action":    f"🚨  Value {value:.3f} {unit} has exceeded the ACTION level! Immediate action required.",
    }
    css_class = {"OK": "status-ok", "Tolerance": "status-tol", "Action": "status-action"}.get(status, "status-ok")
    return f'<div class="status-box {css_class}">{messages.get(status, "")}</div>'


def build_control_chart(df: pd.DataFrame, test: dict) -> go.Figure:
    """
    Build a Plotly control chart for a float-type test.
    Plots measured values with colour-coded markers and overlays
    the Reference, Tolerance, and Action limit lines.
    """
    # Colour-map each point by status
    color_map = {"OK": "#2d9e5f", "Tolerance": "#e58b00", "Action": "#c0392b"}
    marker_colors = df["status"].map(color_map).fillna("#415a77")

    fig = go.Figure()

    # ── Measured values ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["performed_at"],
        y=df["value_float"],
        mode="lines+markers",
        name="Measured",
        line=dict(color="#415a77", width=1.5),
        marker=dict(color=marker_colors.tolist(), size=8, line=dict(width=1, color="#fff")),
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Value: %{y:.3f}<extra></extra>",
    ))

    # ── Reference line ────────────────────────────────────────────────────
    ref = test.get("reference_value")
    if ref is not None:
        fig.add_hline(
            y=ref, line=dict(color="#2d9e5f", width=2, dash="dash"),
            annotation_text=f"Reference ({ref})", annotation_position="top left",
            annotation_font_color="#2d9e5f",
        )

    # ── Tolerance band ────────────────────────────────────────────────────
    tol_lo = test.get("tolerance_low")
    tol_hi = test.get("tolerance_high")
    if tol_lo is not None and tol_hi is not None:
        fig.add_hrect(
            y0=tol_lo, y1=tol_hi,
            fillcolor="rgba(45,158,95,0.08)", line_width=0,
            annotation_text="Tolerance Zone", annotation_position="top right",
            annotation_font_color="#2d9e5f",
        )
        fig.add_hline(y=tol_lo, line=dict(color="#e58b00", width=1.5, dash="dot"),
                      annotation_text=f"Tol Low ({tol_lo})", annotation_position="bottom left",
                      annotation_font_color="#e58b00")
        fig.add_hline(y=tol_hi, line=dict(color="#e58b00", width=1.5, dash="dot"),
                      annotation_text=f"Tol High ({tol_hi})", annotation_position="top left",
                      annotation_font_color="#e58b00")

    # ── Action limit lines ────────────────────────────────────────────────
    act_lo = test.get("action_low")
    act_hi = test.get("action_high")
    if act_lo is not None:
        fig.add_hline(y=act_lo, line=dict(color="#c0392b", width=1.5, dash="dashdot"),
                      annotation_text=f"Action Low ({act_lo})", annotation_position="bottom left",
                      annotation_font_color="#c0392b")
    if act_hi is not None:
        fig.add_hline(y=act_hi, line=dict(color="#c0392b", width=1.5, dash="dashdot"),
                      annotation_text=f"Action High ({act_hi})", annotation_position="top left",
                      annotation_font_color="#c0392b")

    unit_label = test.get("unit_of_measure") or ""
    is_dark = st.session_state.get("app_theme_dark", True)
    plot_template = "plotly_dark" if is_dark else "plotly_white"
    grid_col = "rgba(255, 255, 255, 0.08)" if is_dark else "#e8edf3"

    fig.update_layout(
        template=plot_template,
        title=dict(text=f"<b>{test['name']}</b> — Control Chart", font=dict(size=18)),
        xaxis_title="Date / Time",
        yaxis_title=f"Value ({unit_label})" if unit_label else "Value",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=80, b=60),
        height=480,
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid_col)
    fig.update_yaxes(showgrid=True, gridcolor=grid_col)

    return fig


# ===========================================================================
# Page: Dashboard
# ===========================================================================
def page_dashboard():
    st.markdown('<div class="section-header">🏥 Today\'s QA Dashboard</div>', unsafe_allow_html=True)
    today = date.today().strftime("%B %d, %Y")
    st.caption(f"Date: {today}")

    units_df = get_units()
    if units_df.empty:
        st.info("No units found. Add a unit in Admin / Settings.")
        return

    summary_df = get_status_summary()

    # ── Top-level status metrics ──────────────────────────────────────────
    total_ok  = int(summary_df.loc[summary_df["status"] == "OK",        "count"].sum()) if not summary_df.empty else 0
    total_tol = int(summary_df.loc[summary_df["status"] == "Tolerance", "count"].sum()) if not summary_df.empty else 0
    total_act = int(summary_df.loc[summary_df["status"] == "Action",    "count"].sum()) if not summary_df.empty else 0
    total_all = total_ok + total_tol + total_act

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Results Today",  total_all)
    c2.metric("✅ OK",                 total_ok,  delta=None)
    c3.metric("⚠️ Tolerance",          total_tol, delta=None)
    c4.metric("🚨 Action Level",       total_act, delta=None)

    st.markdown("---")

    # ── Per-unit summary table ────────────────────────────────────────────
    st.subheader("Per-Unit Status Breakdown (Today)")
    if summary_df.empty:
        st.info("No QA data has been recorded today yet.")
    else:
        pivot = summary_df.pivot_table(index="unit_name", columns="status", values="count", fill_value=0).reset_index()
        st.dataframe(pivot, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Equipment overview ────────────────────────────────────────────────
    st.subheader("Equipment Overview")
    st.dataframe(
        units_df[["name", "modality", "manufacturer", "model", "location"]],
        use_container_width=True,
        hide_index=True,
    )


# ===========================================================================
# Page: QA Data Entry
# ===========================================================================
def page_qa_entry():
    st.markdown('<div class="section-header">📋 QA Data Entry</div>', unsafe_allow_html=True)

    units_df = get_units()
    if units_df.empty:
        st.warning("No units registered. Please add a unit in ⚙️ Admin / Settings first.")
        return

    # ── Selectors ──────────────────────────────────────────────────────────
    col_unit, col_test = st.columns(2)

    with col_unit:
        unit_names = units_df["name"].tolist()
        selected_unit_name = st.selectbox("Select Unit", unit_names, key="entry_unit")
        selected_unit_id   = int(units_df.loc[units_df["name"] == selected_unit_name, "id"].values[0])

    tests_for_unit = get_tests_for_unit(selected_unit_id)

    with col_test:
        if tests_for_unit.empty:
            st.warning("No tests assigned to this unit. Assign tests in ⚙️ Admin / Settings.")
            return
        test_names       = tests_for_unit["name"].tolist()
        selected_test_name = st.selectbox("Select Test", test_names, key="entry_test")
        selected_test_id   = int(tests_for_unit.loc[tests_for_unit["name"] == selected_test_name, "id"].values[0])

    test = get_test_by_id(selected_test_id)

    # ── Test metadata info box ─────────────────────────────────────────────
    with st.expander("ℹ️ Test Definition Details", expanded=False):
        info_cols = st.columns(4)
        info_cols[0].metric("Frequency",  test.get("frequency") or "—")
        info_cols[1].metric("Reference",  f"{test.get('reference_value')} {test.get('unit_of_measure') or ''}".strip() if test.get('reference_value') is not None else "—")
        info_cols[2].metric("Tolerance",  f"{test.get('tolerance_low')} / {test.get('tolerance_high')}" if test.get('tolerance_low') is not None else "—")
        info_cols[3].metric("Action",     f"{test.get('action_low')} / {test.get('action_high')}"     if test.get('action_low')     is not None else "—")
        if test.get("description"):
            st.caption(test["description"])

    st.markdown("---")

    # ── Data-entry form ────────────────────────────────────────────────────
    with st.form("qa_entry_form", clear_on_submit=True):
        st.subheader("Enter Result")

        data_type = test.get("data_type", "float")
        value_float   = None
        value_bool    = None
        value_choice  = None

        if data_type == "float":
            ref = test.get("reference_value") or 0.0
            value_float = st.number_input(
                f"Measured Value ({test.get('unit_of_measure') or 'units'})",
                value=float(ref),
                format="%.4f",
                step=0.01,
            )

        elif data_type == "boolean":
            value_bool = st.selectbox("Result", options=["Pass", "Fail"]) == "Pass"

        elif data_type == "choice":
            choices_raw = test.get("choices") or ""
            options = [c.strip() for c in choices_raw.split(",") if c.strip()]
            if options:
                value_choice = st.selectbox("Result", options=options)
            else:
                st.warning("No choices defined for this test. Edit via ⚙️ Admin / Settings.")

        fcol1, fcol2 = st.columns(2)
        with fcol1:
            performed_by = st.text_input("Performed By", placeholder="Your name or initials")
        with fcol2:
            performed_at_date = st.date_input("Date of Measurement", value=date.today())
            performed_at_time = st.time_input("Time of Measurement", value=datetime.now().time())

        comment = st.text_area("Comment / Notes", placeholder="Optional observations…", height=80)

        submitted = st.form_submit_button("💾 Submit Result", use_container_width=True, type="primary")

    # ── Process submission ─────────────────────────────────────────────────
    if submitted:
        performed_at_dt = datetime.combine(performed_at_date, performed_at_time).isoformat(timespec="seconds")

        status = log_result(
            unit_id       = selected_unit_id,
            test_id       = selected_test_id,
            value_float   = value_float,
            value_boolean = value_bool,
            value_choice  = value_choice,
            performed_by  = performed_by,
            comment       = comment,
            performed_at  = performed_at_dt,
        )

        # ── Visual feedback based on status ───────────────────────────────
        unit_label = test.get("unit_of_measure") or ""

        if data_type == "float" and value_float is not None:
            st.markdown(status_box_html(status, value_float, unit_label), unsafe_allow_html=True)
        else:
            # Boolean / choice always shows success
            st.success("✅ Result recorded successfully.")

        st.balloons() if status == "OK" else None

    # ── Recent results for this unit ───────────────────────────────────────
    st.markdown("---")
    st.subheader(f"Recent Entries — {selected_unit_name}")
    recent = get_recent_results(selected_unit_id, limit=10)
    if recent.empty:
        st.info("No results recorded yet for this unit.")
    else:
        # Colour the status column
        st.dataframe(recent, use_container_width=True, hide_index=True)


# ===========================================================================
# Page: Trending & Reports
# ===========================================================================
def page_trending():
    st.markdown('<div class="section-header">📈 Trending & Reports</div>', unsafe_allow_html=True)

    units_df = get_units()
    if units_df.empty:
        st.info("No units available.")
        return

    col_u, col_t, col_n = st.columns([2, 2, 1])

    with col_u:
        unit_names       = units_df["name"].tolist()
        sel_unit_name    = st.selectbox("Unit", unit_names, key="trend_unit")
        sel_unit_id      = int(units_df.loc[units_df["name"] == sel_unit_name, "id"].values[0])

    tests_df = get_tests_for_unit(sel_unit_id)

    with col_t:
        if tests_df.empty:
            st.warning("No tests for this unit.")
            return
        # Allow only float tests for the chart (boolean/choice don't make sense on a control chart)
        float_tests = tests_df[tests_df["data_type"] == "float"]
        if float_tests.empty:
            st.warning("No numeric (float) tests assigned to this unit.")
            return
        sel_test_name = st.selectbox("Test", float_tests["name"].tolist(), key="trend_test")
        sel_test_id   = int(float_tests.loc[float_tests["name"] == sel_test_name, "id"].values[0])

    with col_n:
        n_points = st.number_input("Max Points", min_value=10, max_value=500, value=90, step=10,
                                   help="Limit the number of historical data points displayed.")

    df = get_historical_data(sel_unit_id, sel_test_id)

    if df.empty:
        st.info("No historical data recorded for this unit/test combination yet.")
        return

    # Limit to most recent N points
    if len(df) > n_points:
        df = df.tail(int(n_points))

    test = get_test_by_id(sel_test_id)

    # ── Summary statistics ─────────────────────────────────────────────────
    valid = df["value_float"].dropna()
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("N Points",  len(valid))
    s2.metric("Mean",      f"{valid.mean():.3f}")
    s3.metric("Std Dev",   f"{valid.std():.3f}")
    s4.metric("Min",       f"{valid.min():.3f}")
    s5.metric("Max",       f"{valid.max():.3f}")

    # ── Status distribution ────────────────────────────────────────────────
    status_counts = df["status"].value_counts()
    pct_ok  = status_counts.get("OK", 0) / max(len(df), 1) * 100
    pct_tol = status_counts.get("Tolerance", 0) / max(len(df), 1) * 100
    pct_act = status_counts.get("Action", 0) / max(len(df), 1) * 100

    b1, b2, b3 = st.columns(3)
    b1.metric("% OK",        f"{pct_ok:.1f}%")
    b2.metric("% Tolerance", f"{pct_tol:.1f}%", delta=None)
    b3.metric("% Action",    f"{pct_act:.1f}%", delta=None)

    st.markdown("---")

    # ── Control chart ─────────────────────────────────────────────────────
    fig = build_control_chart(df, test)
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw data table (toggleable) ────────────────────────────────────────
    with st.expander("📄 Raw Data Table", expanded=False):
        display_df = df[["performed_at", "value_float", "status", "performed_by", "comment"]].copy()
        display_df.columns = ["Date/Time", "Value", "Status", "Performed By", "Comment"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Download CSV ───────────────────────────────────────────────────────
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv_data,
        file_name=f"{sel_unit_name}_{sel_test_name}_{date.today()}.csv".replace(" ", "_"),
        mime="text/csv",
    )


# ===========================================================================
# Page: Service & Downtime Log
# ===========================================================================
def page_service_log():
    st.markdown('<div class="section-header">🔧 Service & Downtime Log</div>', unsafe_allow_html=True)

    units_df = get_units()
    if units_df.empty:
        st.warning("No units registered. Add a unit in ⚙️ Admin / Settings first.")
        return

    # ── Log new event ──────────────────────────────────────────────────────
    st.subheader("Log New Service Event")

    with st.form("service_form", clear_on_submit=True):
        sc1, sc2 = st.columns(2)

        with sc1:
            unit_names   = units_df["name"].tolist()
            sel_unit     = st.selectbox("Unit", unit_names, key="svc_unit")
            sel_unit_id  = int(units_df.loc[units_df["name"] == sel_unit, "id"].values[0])
            event_type   = st.selectbox(
                "Event Type",
                ["Downtime", "Preventive Maintenance", "Repair", "Inspection", "Note"],
            )
            performed_by = st.text_input("Performed By", placeholder="Engineer / Physicist name")

        with sc2:
            start_date = st.date_input("Start Date", value=date.today())
            start_time = st.time_input("Start Time", value=datetime.now().replace(minute=0, second=0).time())
            end_date   = st.date_input("End Date (optional)", value=date.today())
            end_time   = st.time_input("End Time (optional)", value=datetime.now().replace(minute=0, second=0).time())
            still_ongoing = st.checkbox("Machine still down / event ongoing")

        description = st.text_area("Description / Notes", placeholder="Details of the event…", height=100)

        submitted_svc = st.form_submit_button("📝 Log Event", use_container_width=True, type="primary")

    if submitted_svc:
        start_dt = datetime.combine(start_date, start_time).isoformat(timespec="seconds")
        end_dt   = None if still_ongoing else datetime.combine(end_date, end_time).isoformat(timespec="seconds")

        add_service_log(
            unit_id       = sel_unit_id,
            event_type    = event_type,
            start_datetime = start_dt,
            end_datetime   = end_dt,
            performed_by  = performed_by,
            description   = description,
        )
        st.success(f"✅ Service event logged for **{sel_unit}**.")

    st.markdown("---")

    # ── Filter & display logs ──────────────────────────────────────────────
    st.subheader("Service Log Records")

    fc1, fc2 = st.columns(2)
    with fc1:
        filter_unit = st.selectbox(
            "Filter by Unit",
            options=["All Units"] + units_df["name"].tolist(),
            key="svc_filter",
        )
    with fc2:
        filter_type = st.selectbox(
            "Filter by Event Type",
            options=["All Types", "Downtime", "Preventive Maintenance", "Repair", "Inspection", "Note"],
            key="svc_type_filter",
        )

    # Fetch logs
    if filter_unit == "All Units":
        logs_df = get_service_logs()
    else:
        uid = int(units_df.loc[units_df["name"] == filter_unit, "id"].values[0])
        logs_df = get_service_logs(unit_id=uid)

    if filter_type != "All Types" and not logs_df.empty:
        logs_df = logs_df[logs_df["event_type"] == filter_type]

    if logs_df.empty:
        st.info("No service log entries found.")
    else:
        # Format for display
        disp = logs_df[["unit_name", "event_type", "start_datetime", "end_datetime", "performed_by", "description"]].copy()
        disp.columns = ["Unit", "Event Type", "Start", "End", "Performed By", "Description"]
        disp["End"] = disp["End"].fillna("Ongoing ⚠️")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # ── Download ───────────────────────────────────────────────────────
        csv = logs_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Log CSV", data=csv,
                           file_name=f"service_log_{date.today()}.csv", mime="text/csv")


# ===========================================================================
# Page: Admin / Settings & Equipment
# ===========================================================================
def page_admin():
    st.markdown('<div class="section-header">⚙️ Equipment, Linacs & System Admin</div>', unsafe_allow_html=True)

    tab_linacs, tab_chambers, tab_tests, tab_assign = st.tabs([
        "🏭 Linacs & Golden Beam Data",
        "☢️ Ion Chambers & Certificates",
        "🧪 Manage Tests",
        "🔗 Assign Tests to Units",
    ])

    # ── Tab 1: Linacs & Golden Beam Data ──────────────────────────────────
    with tab_linacs:
        st.subheader("Linac Machines & Golden Reference Beam Data")
        st.caption(
            "Configure clinical treatment units, assign beam energies, and upload Golden Reference Beam Data "
            "(water tank PDDs and profiles). Golden baselines automatically populate beam quality metrics (TPR₂₀,₁₀, PDD₁₀) "
            "for IAEA TRS-398 absolute reference dosimetry and provide overlay baselines in Beam Profile QA."
        )

        units_df = get_units()
        if units_df.empty:
            st.warning("No treatment units registered in database yet. Add your first unit below.")
        else:
            col_sel, col_info = st.columns([1, 2])
            with col_sel:
                sel_unit_name = st.selectbox("Select Linac Unit", units_df["name"].tolist(), key="admin_sel_unit")
                sel_unit_row = units_df[units_df["name"] == sel_unit_name].iloc[0]
                sel_unit_id = int(sel_unit_row["id"])
            with col_info:
                st.info(
                    f"**{sel_unit_row['name']}** ({sel_unit_row['modality']}) — "
                    f"{sel_unit_row['manufacturer'] or 'Varian'} {sel_unit_row['model'] or ''} "
                    f"| S/N: {sel_unit_row['serial_no'] or 'N/A'} | Location: {sel_unit_row['location'] or 'Main Vault'}"
                )

            # Clinical non-matching advisory
            if "EDGE" in sel_unit_name.upper():
                st.warning("ℹ️ **Clinical Note:** Edge is independent and NOT beam-matched to TrueBeam. All golden baselines and energy parameters are stored independently.")

            st.markdown("---")

            # ── 1. Configured Energies for this Linac ──
            st.subheader(f"⚡ Configured Beam Energies for {sel_unit_name}")
            energies_df = get_unit_energies(unit_id=sel_unit_id)

            if energies_df.empty:
                st.info(f"No beam energies configured for {sel_unit_name} yet. Upload a PDD scan below or add an energy.")
            else:
                disp_en = energies_df[["energy_name", "beam_type", "nominal_mv", "tpr20_10", "pdd_10", "d_max_mm"]].copy()
                disp_en.columns = ["Energy Name", "Beam Type", "Nominal (MV)", "Ref TPR₂₀,₁₀", "Ref PDD₁₀ (%)", "d_max (mm)"]
                disp_en["Ref TPR₂₀,₁₀"] = disp_en["Ref TPR₂₀,₁₀"].apply(lambda v: f"{v:.4f}" if pd.notnull(v) else "—")
                disp_en["Ref PDD₁₀ (%)"] = disp_en["Ref PDD₁₀ (%)"].apply(lambda v: f"{v:.1f}%" if pd.notnull(v) else "—")
                disp_en["d_max (mm)"] = disp_en["d_max (mm)"].apply(lambda v: f"{v:.1f} mm" if pd.notnull(v) else "—")
                st.dataframe(disp_en, use_container_width=True, hide_index=True)

            with st.expander("➕ Configure / Add Beam Energy Manually", expanded=False):
                with st.form(f"add_energy_form_{sel_unit_id}"):
                    ce1, ce2, ce3 = st.columns(3)
                    with ce1:
                        new_en_name = st.text_input("Energy Name", placeholder="e.g. 6 MV, 6 MV FFF, 10 MV, 15 MV")
                    with ce2:
                        new_en_type = st.selectbox("Beam Type", ["Photon Flattened", "Photon FFF", "Electron"])
                        new_nom_mv = st.number_input("Nominal Energy (MV/MeV)", value=6.0, step=1.0)
                    with ce3:
                        new_tpr = st.number_input("Reference TPR₂₀,₁₀ (optional)", value=0.670, format="%.4f")
                        new_pdd10 = st.number_input("Reference PDD₁₀ % (optional)", value=67.0, format="%.1f")
                    submit_en = st.form_submit_button("Add / Update Energy", type="primary")
                    if submit_en and new_en_name.strip():
                        add_unit_energy(
                            unit_id=sel_unit_id,
                            energy_name=new_en_name.strip(),
                            beam_type=new_en_type,
                            nominal_mv=float(new_nom_mv),
                            tpr20_10=float(new_tpr) if new_tpr > 0 else None,
                            pdd_10=float(new_pdd10) if new_pdd10 > 0 else None,
                        )
                        st.success(f"Added energy {new_en_name} to {sel_unit_name}.")
                        st.rerun()

            st.markdown("---")

            # ── 2. Golden Reference Beam Data Uploader ──
            st.subheader(f"🌟 Upload Golden Reference Beam Data for {sel_unit_name}")
            st.markdown(
                "Upload water tank scans (**PTW BeamScan `.mcc`**, IBA OmniPro, CSV, TSV, or Excel). "
                "The system parses multi-scan files, calculates clinical beam parameters ($TPR_{20,10}$, $PDD_{10}$, $d_{max}$, FWHM, penumbra, flatness, symmetry), "
                "and stores them as the official golden baseline in `qa_data.db`."
            )

            ref_file = st.file_uploader(
                f"Choose beam data file for {sel_unit_name}",
                type=["mcc", "csv", "tsv", "txt", "asc", "dat", "xlsx", "xls"],
                key=f"ref_uploader_{sel_unit_id}",
            )

            if ref_file is not None:
                file_bytes = ref_file.getvalue()
                filename = ref_file.name

                parsed_result = None
                try:
                    if filename.lower().endswith((".xlsx", ".xls")):
                        excel_dfs = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
                        first_df = list(excel_dfs.values())[0]
                        numeric_cols = first_df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) >= 2:
                            x_col = numeric_cols[0]
                            y_col = numeric_cols[1]
                            parsed_result = {
                                "x_mm": first_df[x_col].values,
                                "values": first_df[y_col].values,
                                "meta": {
                                    "data_source": f"Excel: {filename}",
                                    "axis": "Profile / PDD",
                                    "field_size_nominal": "10x10 cm",
                                    "energy": "6 MV",
                                    "beam_type": "Flattened",
                                },
                            }
                    else:
                        parsed_result = bpa.parse_profile_text(file_bytes)
                except Exception as parse_err:
                    st.error(f"Error parsing file: {parse_err}")

                if parsed_result is not None:
                    active_scan = None
                    if "profiles" in parsed_result and parsed_result["profiles"]:
                        scan_keys = list(parsed_result["profiles"].keys())
                        st.success(f"Detected **{len(scan_keys)}** scans in file `{filename}`.")
                        selected_scan_key = st.selectbox("Select scan to inspect & register as baseline:", scan_keys, key="sel_scan_mcc")
                        active_scan = parsed_result["profiles"][selected_scan_key]
                    else:
                        active_scan = parsed_result

                    x_arr = np.asarray(active_scan["x_mm"], dtype=float)
                    y_arr = np.asarray(active_scan["values"], dtype=float)
                    meta_dict = active_scan.get("meta", {})

                    detected_axis = meta_dict.get("axis", "Crossplane")
                    is_pdd = "PDD" in detected_axis.upper() or "DEPTH" in detected_axis.upper() or "PDD" in filename.upper()
                    is_fff = meta_dict.get("is_fff", False) or "FFF" in str(meta_dict.get("energy", "")).upper() or "FFF" in filename.upper()

                    col_plot, col_analysis = st.columns([3, 2])

                    with col_analysis:
                        st.markdown("##### 🔬 Scan Analysis")
                        if is_pdd:
                            pdd_res = bpa.analyze_pdd_curve(x_arr, y_arr)
                            st.metric("Computed TPR₂₀,₁₀", f"{pdd_res['tpr20_10']:.4f}", help="IAEA TRS-398 Eq. 12")
                            st.metric("PDD at 10 cm", f"{pdd_res['pdd_10']:.1f}%")
                            st.metric("PDD at 20 cm", f"{pdd_res['pdd_20']:.1f}%")
                            st.metric("d_max", f"{pdd_res['d_max_mm']:.1f} mm")
                            st.caption(f"PDD₂₀ / PDD₁₀ = {pdd_res['pdd_20_10']:.4f}")
                        else:
                            b_res = bpa.analyze_beam_profile(x_arr, y_arr, beam_type="FFF" if is_fff else "Flattened")
                            st.metric("Field Width", f"{b_res.get('field_width_mm', 0):.1f} mm")
                            st.metric("Penumbra (80-20%)", f"{b_res.get('penumbra_mean_mm', 0):.2f} mm")
                            if not is_fff:
                                st.metric("Flatness (IEC)", f"{b_res.get('flatness_iec_pct', 0):.2f}%")
                            st.metric("Symmetry", f"{b_res.get('symmetry_pct', 0):.2f}%")

                    with col_plot:
                        fig_ref = go.Figure()
                        if is_pdd:
                            fig_ref.add_trace(go.Scatter(
                                x=x_arr, y=y_arr, mode="lines", name="PDD Curve",
                                line=dict(color="#0284c7", width=2.5),
                            ))
                            fig_ref.add_vline(x=100.0, line=dict(color="#e58b00", dash="dash"), annotation_text="10 cm (z_ref)")
                            fig_ref.add_vline(x=200.0, line=dict(color="#e58b00", dash="dash"), annotation_text="20 cm")
                            fig_ref.update_layout(
                                title=f"PDD Baseline: {meta_dict.get('energy', 'Photon')}",
                                xaxis_title="Depth in Water (mm)", yaxis_title="Dose (%)",
                                height=340, margin=dict(l=40, r=20, t=40, b=40),
                            )
                        else:
                            fig_ref.add_trace(go.Scatter(
                                x=x_arr, y=y_arr, mode="lines", name="Profile Curve",
                                line=dict(color="#0284c7", width=2.5),
                            ))
                            fig_ref.add_vline(x=0.0, line=dict(color="#94a3b8", dash="dash"), annotation_text="CAX")
                            fig_ref.update_layout(
                                title=f"Profile Baseline: {meta_dict.get('energy', 'Photon')} ({detected_axis})",
                                xaxis_title="Off-Axis Position (mm)", yaxis_title="Dose (%)",
                                height=340, margin=dict(l=40, r=20, t=40, b=40),
                            )
                        st.plotly_chart(fig_ref, use_container_width=True)

                    with st.form(f"save_golden_form_{sel_unit_id}"):
                        st.markdown("##### 💾 Commit Golden Baseline & Register Energy")
                        gf1, gf2, gf3 = st.columns(3)
                        with gf1:
                            target_energy = st.text_input("Energy Name *", value=meta_dict.get("energy", "6 MV").strip())
                            curve_type_sel = st.selectbox("Curve Type", ["PDD", "Crossplane Profile", "Inplane Profile"], index=0 if is_pdd else 1)
                        with gf2:
                            beam_type_sel = st.selectbox("Beam Quality Type", ["Photon Flattened", "Photon FFF", "Electron"], index=1 if is_fff else 0)
                            fs_sel = st.text_input("Field Size", value=meta_dict.get("field_size_nominal", "10x10 cm"))
                        with gf3:
                            nom_depth = st.number_input("Measurement Depth (mm)", value=float(meta_dict.get("depth_mm", 100.0) if not is_pdd else 0.0), step=10.0)
                            phys_name = st.text_input("Physicist / Uploader", value="Medical Physicist")

                        save_golden_btn = st.form_submit_button("🌟 Save as Golden Baseline & Register Energy", type="primary", use_container_width=True)

                    if save_golden_btn:
                        curve_data_json = json.dumps({"x_mm": x_arr.tolist(), "values": y_arr.tolist()})

                        tpr_val = None
                        pdd10_val = None
                        pdd20_val = None
                        dmax_val = None
                        fwidth_val = None
                        pen_l = None
                        pen_r = None
                        flat_pct = None
                        sym_pct = None

                        if is_pdd:
                            pdd_res = bpa.analyze_pdd_curve(x_arr, y_arr)
                            tpr_val = float(pdd_res["tpr20_10"])
                            pdd10_val = float(pdd_res["pdd_10"])
                            pdd20_val = float(pdd_res["pdd_20"])
                            dmax_val = float(pdd_res["d_max_mm"])
                        else:
                            b_res = bpa.analyze_beam_profile(x_arr, y_arr, beam_type="FFF" if "FFF" in beam_type_sel else "Flattened")
                            fwidth_val = float(b_res.get("field_width_mm") or 0.0)
                            pen_l = float(b_res.get("penumbra_left_mm") or 0.0)
                            pen_r = float(b_res.get("penumbra_right_mm") or 0.0)
                            flat_pct = float(b_res.get("flatness_iec_pct") or 0.0) if not is_fff else None
                            sym_pct = float(b_res.get("symmetry_pct") or 0.0)

                        save_reference_beam_data(
                            unit_name=sel_unit_name,
                            energy=target_energy.strip(),
                            beam_type="FFF" if "FFF" in beam_type_sel else "Flattened",
                            curve_type=curve_type_sel,
                            field_size_nominal=fs_sel,
                            depth_mm=nom_depth,
                            ssd_cm=100.0,
                            curve_data_json=curve_data_json,
                            tpr20_10=tpr_val,
                            pdd_10=pdd10_val,
                            pdd_20=pdd20_val,
                            d_max_mm=dmax_val,
                            field_width_mm=fwidth_val,
                            penumbra_left_mm=pen_l,
                            penumbra_right_mm=pen_r,
                            flatness_pct=flat_pct,
                            symmetry_pct=sym_pct,
                            source_filename=filename,
                            uploaded_by=phys_name,
                        )

                        mv_match = re.search(r"(\d+)", target_energy)
                        nom_mv = float(mv_match.group(1)) if mv_match else 6.0
                        add_unit_energy(
                            unit_id=sel_unit_id,
                            energy_name=target_energy.strip(),
                            beam_type=beam_type_sel,
                            nominal_mv=nom_mv,
                            d_max_mm=dmax_val,
                            tpr20_10=tpr_val,
                            pdd_10=pdd10_val,
                        )

                        st.success(
                            f"✅ Golden Baseline successfully stored for **{sel_unit_name}** — **{target_energy}** ({curve_type_sel})!\n\n"
                            f"Energy **{target_energy}** registered in unit energies with TPR₂₀,₁₀ = {tpr_val if tpr_val else 'N/A'}. "
                            f"Available immediately across TRS-398 Reference Dosimetry and Beam Profile QA."
                        )
                        st.rerun()

            # ── 3. Stored Baselines Table for this Linac ──
            st.markdown("---")
            st.subheader(f"📚 Stored Reference Baselines for {sel_unit_name}")
            unit_refs = get_reference_beam_data(unit_name=sel_unit_name)

            if unit_refs.empty:
                st.info(f"No reference baselines stored for {sel_unit_name} yet.")
            else:
                disp_refs = unit_refs[["id", "energy", "curve_type", "beam_type", "field_size_nominal", "tpr20_10", "pdd_10", "uploaded_at"]].copy()
                disp_refs.columns = ["ID", "Energy", "Curve Type", "Beam Quality", "Field Size", "TPR₂₀,₁₀", "PDD₁₀ (%)", "Uploaded At"]
                disp_refs["TPR₂₀,₁₀"] = disp_refs["TPR₂₀,₁₀"].apply(lambda v: f"{v:.4f}" if pd.notnull(v) else "—")
                disp_refs["PDD₁₀ (%)"] = disp_refs["PDD₁₀ (%)"].apply(lambda v: f"{v:.1f}%" if pd.notnull(v) else "—")
                st.dataframe(disp_refs, use_container_width=True, hide_index=True)

                col_del_id, col_del_btn = st.columns([2, 1])
                with col_del_id:
                    del_ref_id = st.selectbox("Select Baseline ID to Remove", unit_refs["id"].tolist(), key="del_ref_sel")
                with col_del_btn:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ Delete Baseline", type="secondary", key="del_ref_btn"):
                        delete_reference_beam_data(int(del_ref_id))
                        st.success("Deleted baseline record.")
                        st.rerun()

            # ── 4. Unit Details Edit / Add ──
            st.markdown("---")
            with st.expander("✏️ Edit Unit Details or ➕ Register New Linac", expanded=False):
                # Edit Unit
                st.markdown("##### ✏️ Edit Existing Unit Details")
                edit_unit_row = units_df[units_df["name"] == sel_unit_name].iloc[0]
                MODALITIES = ["LINAC", "CT", "MRI", "PET/CT", "Cobalt", "kV Imaging", "Other"]
                current_modality = edit_unit_row["modality"] if edit_unit_row["modality"] in MODALITIES else "Other"

                with st.form("edit_unit_form_unified"):
                    eu1, eu2 = st.columns(2)
                    with eu1:
                        e_name = st.text_input("Unit Name *", value=edit_unit_row["name"])
                        e_modality = st.selectbox("Modality *", MODALITIES, index=MODALITIES.index(current_modality))
                        e_mfr = st.text_input("Manufacturer", value=edit_unit_row["manufacturer"] or "")
                        e_model = st.text_input("Model", value=edit_unit_row["model"] or "")
                    with eu2:
                        e_serial = st.text_input("Serial Number", value=edit_unit_row["serial_no"] or "")
                        e_location = st.text_input("Location", value=edit_unit_row["location"] or "")
                        e_notes = st.text_area("Notes", value=edit_unit_row["notes"] or "", height=80)
                    save_edit = st.form_submit_button("💾 Save Unit Changes", use_container_width=True, type="primary")

                if save_edit:
                    if not e_name.strip():
                        st.error("Unit Name cannot be empty.")
                    else:
                        update_unit(sel_unit_id, e_name.strip(), e_modality, e_mfr, e_model, e_serial, e_location, e_notes)
                        st.success(f"✅ **{e_name}** updated.")
                        st.rerun()

                # Add Unit
                st.markdown("##### ➕ Register New Treatment Unit")
                with st.form("add_unit_form_unified", clear_on_submit=True):
                    u1, u2 = st.columns(2)
                    with u1:
                        new_name = st.text_input("Unit Name *", placeholder="e.g. Varian Edge #2")
                        new_modality = st.selectbox("Modality *", ["LINAC", "CT", "MRI", "PET/CT", "Cobalt", "kV Imaging", "Other"])
                        new_mfr = st.text_input("Manufacturer", placeholder="e.g. Varian / Elekta")
                        new_model = st.text_input("Model", placeholder="e.g. TrueBeam / Edge / Halcyon")
                    with u2:
                        new_serial = st.text_input("Serial Number")
                        new_location = st.text_input("Location", placeholder="e.g. Vault 3")
                        new_notes = st.text_area("Notes", height=80)
                    submit_unit = st.form_submit_button("➕ Add Unit", use_container_width=True, type="primary")

                if submit_unit:
                    if not new_name.strip():
                        st.error("Unit Name is required.")
                    else:
                        add_unit(new_name.strip(), new_modality, new_mfr, new_model, new_serial, new_location, new_notes)
                        st.success(f"✅ Unit **{new_name}** added successfully.")
                        st.rerun()

    # ── Tab 2: Ion Chambers & Certificates ────────────────────────────────
    with tab_chambers:
        st.subheader("Ionization Chamber Registry & Calibration Certificates")
        st.caption(
            "Upload official calibration certificates (PDF or Image) from accredited calibration laboratories "
            "(PTW-Freiburg SSDL, IAEA Dosimetry Laboratory, IBA, Standard Imaging ADCL, NIST). "
            "Calibration factors (N_D,w) are automatically parsed, normalized to cGy/nC, and registered for IAEA TRS-398 reference dosimetry."
        )

        cert_file = st.file_uploader(
            "Upload Chamber Calibration Certificate (PDF or Image)",
            type=["pdf", "png", "jpg", "jpeg"],
            key="admin_chamber_cert_file",
        )

        def_model = "PTW 30013 Farmer (0.60 cm³, waterproof)"
        def_sn = ""
        def_ndw_cgy = 5.3850
        def_lab = "PTW-Freiburg Calibration Laboratory (SSDL)"
        def_date = str(date.today())
        def_cert_id = ""
        def_notes = ""

        if cert_file is not None:
            file_bytes = cert_file.getvalue()
            filename = cert_file.name

            if filename.lower().endswith(".pdf"):
                try:
                    parsed_cert = cm.parse_chamber_certificate_pdf(file_bytes)
                    def_model = parsed_cert.get("model", def_model)
                    def_sn = parsed_cert.get("serial_no", def_sn)
                    def_ndw_cgy = float(parsed_cert.get("ndw_cgy", def_ndw_cgy))
                    def_lab = parsed_cert.get("cal_lab", def_lab)
                    def_date = parsed_cert.get("cal_date", def_date)
                    def_cert_id = parsed_cert.get("cert_id", def_cert_id)

                    st.success(
                        f"✅ **PDF Calibration Certificate Parsed Successfully:**\n\n"
                        f"- **Chamber Model:** `{def_model}`\n"
                        f"- **Serial Number:** `{def_sn}`\n"
                        f"- **Calibration Factor $N_{{D,w}}$:** `{def_ndw_cgy:.4f} cGy/nC` ({parsed_cert.get('ndw_gy', 0.05385):.6f} Gy/nC)\n"
                        f"- **Laboratory:** `{def_lab}` | **Date:** `{def_date}` | **Certificate:** `{def_cert_id}`"
                    )

                    with st.expander("📄 Extracted Text Excerpt", expanded=False):
                        st.text(parsed_cert.get("raw_text_excerpt", ""))
                except Exception as pdf_err:
                    st.error(f"Error parsing PDF certificate: {pdf_err}")
            else:
                st.info("📷 Image certificate uploaded. Preview displayed below for manual verification.")
                st.image(file_bytes, caption=f"Uploaded Certificate: {filename}", use_column_width=True)

        st.markdown("#### 📝 Register / Verify Certified Chamber")
        with st.form("register_chamber_form", clear_on_submit=False):
            rc1, rc2 = st.columns(2)
            with rc1:
                model_idx = 0
                if def_model in cm.STANDARD_CHAMBER_MODELS:
                    model_idx = cm.STANDARD_CHAMBER_MODELS.index(def_model)
                reg_model = st.selectbox("Chamber Model (IAEA TRS-398 Table 16 / 20) *", cm.STANDARD_CHAMBER_MODELS, index=model_idx)
                reg_sn = st.text_input("Chamber Serial Number *", value=def_sn, placeholder="e.g. 010482")
                reg_ndw = st.number_input("N_D,w Calibration Factor (cGy/nC) *", value=float(def_ndw_cgy), format="%.4f", step=0.001)
                reg_label = st.text_input("User Friendly Label", placeholder="e.g. Clinical Farmer #1 (Primary Standard)")

            with rc2:
                reg_lab = st.text_input("Calibration Laboratory", value=def_lab)
                reg_date = st.text_input("Calibration Date", value=def_date)
                reg_cert_id = st.text_input("Certificate ID / Number", value=def_cert_id)
                reg_quality = st.selectbox("Reference Radiation Quality", ["Co-60", "High-Energy Photon"])
                reg_notes = st.text_area("Clinical Notes", value=def_notes, height=70)

            submit_ch = st.form_submit_button("💾 Register Certified Chamber", type="primary", use_container_width=True)

        if submit_ch:
            if not reg_sn.strip():
                st.error("Chamber Serial Number is required.")
            else:
                add_ion_chamber(
                    model=reg_model,
                    serial_no=reg_sn.strip(),
                    ndw_cgy=float(reg_ndw),
                    user_label=reg_label.strip(),
                    cal_date=reg_date.strip(),
                    cal_lab=reg_lab.strip(),
                    cert_id=reg_cert_id.strip(),
                    ref_quality=reg_quality,
                    notes=reg_notes.strip(),
                )
                st.success(f"✅ Ion Chamber **{reg_model}** (S/N {reg_sn}) successfully registered! N_D,w = {reg_ndw:.4f} cGy/nC.")
                st.rerun()

        st.markdown("---")
        st.subheader("🏛️ Hospital Certified Chamber Registry")
        chambers_df = get_ion_chambers(active_only=False)

        if chambers_df.empty:
            st.info("No ionization chambers registered yet.")
        else:
            disp_ch = chambers_df[["id", "model", "serial_no", "ndw_cgy", "ndw_gy", "user_label", "cal_lab", "cal_date", "cert_id"]].copy()
            disp_ch.columns = ["ID", "Model", "Serial No", "N_D,w (cGy/nC)", "N_D,w (Gy/nC)", "Label", "Laboratory", "Calibration Date", "Certificate ID"]
            disp_ch["N_D,w (cGy/nC)"] = disp_ch["N_D,w (cGy/nC)"].apply(lambda v: f"{v:.4f}")
            disp_ch["N_D,w (Gy/nC)"] = disp_ch["N_D,w (Gy/nC)"].apply(lambda v: f"{v:.6f}")
            st.dataframe(disp_ch, use_container_width=True, hide_index=True)

            del_ch_col1, del_ch_col2 = st.columns([2, 1])
            with del_ch_col1:
                del_ch_id = st.selectbox(
                    "Select Chamber to Delete",
                    chambers_df["id"].tolist(),
                    format_func=lambda i: f"ID {i}: {chambers_df.loc[chambers_df['id'] == i, 'model'].values[0]} (S/N {chambers_df.loc[chambers_df['id'] == i, 'serial_no'].values[0]})",
                    key="del_ch_sel",
                )
            with del_ch_col2:
                st.write("")
                st.write("")
                if st.button("🗑️ Delete Chamber", type="secondary", key="del_ch_btn"):
                    delete_ion_chamber(int(del_ch_id))
                    st.success("Deleted chamber record.")
                    st.rerun()

    # ── Tab 2: Tests ──────────────────────────────────────────────────────
    with tab_tests:
        st.subheader("Define New Test")

        with st.form("add_test_form", clear_on_submit=True):
            t1, t2 = st.columns(2)

            with t1:
                t_name     = st.text_input("Test Name *",   placeholder="e.g. Daily Output Constancy")
                t_desc     = st.text_area("Description",    height=70, placeholder="What does this test measure?")
                t_dtype    = st.selectbox("Data Type *",    ["float", "boolean", "choice"])
                t_unit_msr = st.text_input("Unit of Measure", placeholder="e.g. %, mm, cGy (leave blank if N/A)")
                t_freq     = st.selectbox("Frequency",      ["Daily", "Weekly", "Monthly", "Annual", "Ad-hoc"])

            with t2:
                t_ref      = st.number_input("Reference Value",  value=0.0, format="%.4f",
                                             help="Expected / baseline value (float tests only)")
                t_tol_lo   = st.number_input("Tolerance Low",    value=-3.0, format="%.4f",
                                             help="Value below this triggers a Tolerance warning")
                t_tol_hi   = st.number_input("Tolerance High",   value=3.0,  format="%.4f",
                                             help="Value above this triggers a Tolerance warning")
                t_act_lo   = st.number_input("Action Low",       value=-5.0, format="%.4f",
                                             help="Value below this triggers an Action alert")
                t_act_hi   = st.number_input("Action High",      value=5.0,  format="%.4f",
                                             help="Value above this triggers an Action alert")
                t_choices  = st.text_input("Choices (CSV)",     placeholder="Option A, Option B, Option C",
                                           help="For 'choice' type tests only — comma-separated list")

            submit_test = st.form_submit_button("➕ Add Test", use_container_width=True, type="primary")

        if submit_test:
            if not t_name.strip():
                st.error("Test Name is required.")
            else:
                try:
                    ref_val = float(t_ref)   if t_dtype == "float" else None
                    tl      = float(t_tol_lo) if t_dtype == "float" else None
                    th      = float(t_tol_hi) if t_dtype == "float" else None
                    al      = float(t_act_lo) if t_dtype == "float" else None
                    ah      = float(t_act_hi) if t_dtype == "float" else None
                    choices = t_choices.strip() if t_dtype == "choice" else None

                    add_test(
                        name=t_name.strip(), data_type=t_dtype, description=t_desc,
                        unit_of_measure=t_unit_msr, reference_value=ref_val,
                        tolerance_low=tl, tolerance_high=th,
                        action_low=al, action_high=ah,
                        choices=choices, frequency=t_freq,
                    )
                    st.success(f"✅ Test **{t_name}** added successfully.")
                except Exception as e:
                    st.error(f"Error adding test: {e}")

        st.markdown("---")
        st.subheader("Existing Tests")
        tests_df = get_tests()
        if not tests_df.empty:
            st.dataframe(
                tests_df[["id", "name", "data_type", "unit_of_measure", "reference_value",
                           "tolerance_low", "tolerance_high", "action_low", "action_high", "frequency"]],
                use_container_width=True, hide_index=True,
            )

    # ── Tab 3: Assign Tests to Units ──────────────────────────────────────
    with tab_assign:
        st.subheader("Assign a Test to a Unit")
        st.caption("Creates a link so the test appears in the QA Entry form for that unit.")

        units_df = get_units()
        tests_df = get_tests()

        if units_df.empty or tests_df.empty:
            st.warning("You need at least one unit and one test before you can create assignments.")
        else:
            with st.form("assign_form", clear_on_submit=True):
                a1, a2 = st.columns(2)
                with a1:
                    sel_unit_name = st.selectbox("Unit", units_df["name"].tolist(), key="assign_unit")
                    sel_unit_id   = int(units_df.loc[units_df["name"] == sel_unit_name, "id"].values[0])
                with a2:
                    sel_test_name = st.selectbox("Test", tests_df["name"].tolist(), key="assign_test")
                    sel_test_id   = int(tests_df.loc[tests_df["name"] == sel_test_name, "id"].values[0])

                submit_assign = st.form_submit_button("🔗 Assign", use_container_width=True, type="primary")

            if submit_assign:
                try:
                    assign_test_to_unit(sel_unit_id, sel_test_id)
                    st.success(f"✅ **{sel_test_name}** is now assigned to **{sel_unit_name}**.")
                except Exception as e:
                    st.error(f"Error creating assignment: {e}")


# ===========================================================================
# Router
# ===========================================================================
if page == "🏥 Dashboard":
    page_dashboard()
elif page == "📋 QA Data Entry":
    page_qa_entry()
elif page == "📈 Trending & Reports":
    page_trending()
elif page == "☢️ TRS-398 Calibration":
    page_trs398()
elif page == "📐 Beam Profile QA":
    render_beam_profile_page()
elif page == "📊 Trajectory Logs":
    page_trajectory_logs()
elif page == "🔲 Picket Fence QA":
    page_picket_fence()
elif page == "🎯 Winston-Lutz QA":
    page_winston_lutz()
elif page == "🔬 CBCT (CatPhan) QA":
    page_cbct_catphan()
elif page == "🔧 Service & Downtime Log":
    page_service_log()
elif page in ("⚙️ Equipment & Admin", "⚙️ Admin / Settings"):
    page_admin()
