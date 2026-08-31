from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.science.actions import ExperimentActionType
from src.science.discovery_engine import AutonomousDiscoveryEngine

st.set_page_config(
    page_title="AIcoScientist Discovery Console",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    :root {
        --bg-main: #0e1117;
        --card-bg: #1a1f2c;
        --card-border: #2d3748;
        --accent-emerald: #10b981;
        --accent-blue: #3b82f6;
        --accent-amber: #f59e0b;
        --accent-purple: #8b5cf6;
    }
    .metric-card {
        background: #1a1f2c;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .rec-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15);
    }
    .agent-card {
        background: #1e293b;
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .falsification-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 12px;
        margin-top: 12px;
    }
    .roadmap-badge {
        background: #312e81;
        color: #c7d2fe;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine() -> AutonomousDiscoveryEngine:
    engine = AutonomousDiscoveryEngine(seed=42)
    engine.initialize_curated_scenario(n_init_prop=6, n_init_xrd=4, seed=42)
    return engine


if "engine" not in st.session_state:
    st.session_state.engine = get_engine()
    st.session_state.last_outcome = None
    st.session_state.last_rec = None
    st.session_state.last_perspectives = []

engine: AutonomousDiscoveryEngine = st.session_state.engine

# Sidebar Configuration & Controls
with st.sidebar:
    st.title("🔬 AIcoScientist Controls")
    st.caption("Multimodal Autonomous Discovery System")

    st.subheader("Campaign Lifecycle")
    col_reset, col_load = st.columns(2)
    with col_reset:
        if st.button("🔄 Reset Demo", use_container_width=True):
            engine.reset()
            st.session_state.last_outcome = None
            st.session_state.last_rec = None
            st.session_state.last_perspectives = []
            st.rerun()

    with col_load:
        if st.button("📥 Load Scenario", use_container_width=True):
            engine.initialize_curated_scenario(n_init_prop=6, n_init_xrd=4, seed=42)
            st.session_state.last_outcome = None
            st.session_state.last_rec = None
            st.session_state.last_perspectives = []
            st.rerun()

    st.divider()
    st.subheader("Policy Value Weights")
    w_info = st.slider("Scientific Info Weight (w_info)", 0.0, 3.0, 1.0, 0.1)
    w_disc = st.slider("Discovery Weight (w_disc)", 0.0, 3.0, 1.0, 0.1)
    w_cost = st.slider("Cost Weight (w_cost)", 0.0, 1.0, 0.15, 0.05)

    engine.policy.w_info = w_info
    engine.policy.w_disc = w_disc
    engine.policy.w_cost = w_cost

    st.divider()
    st.subheader("Normalized Action Costs")
    cost_xrd = st.number_input("XRD Characterization Cost", value=1.0, step=0.5)
    cost_prop = st.number_input("SECCM Property Test Cost", value=5.0, step=1.0)
    engine.policy.cost_xrd = cost_xrd
    engine.policy.cost_property = cost_prop

    st.caption("Illustrative normalized demo cost units.")

# Main Header & Metrics
st.markdown("## 🧪 AIcoScientist: Multimodal Next-Best-Experiment Discovery Console")
st.markdown(
    "*Deciding which material to investigate, which experiment to run, and what scientific hypothesis is tested.*"
)

revealed_summary = engine.oracle.get_revealed_state_summary()
best_k0 = revealed_summary["best_observed_k0"]
best_k0_str = f"{best_k0:.5f} cm/s" if best_k0 is not None else "None"
budget_spent = engine.total_budget_spent

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Budget Consumed</div>
            <div class="metric-value">{budget_spent:.1f} <span style="font-size:0.9rem;color:#64748b">/ 100 Units</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">XRD Characterizations</div>
            <div class="metric-value" style="color:#10b981">{revealed_summary['num_xrd_observed']} <span style="font-size:0.9rem;color:#64748b">/ 966 Samples</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">SECCM Property Tests</div>
            <div class="metric-value" style="color:#3b82f6">{revealed_summary['num_property_observed']} <span style="font-size:0.9rem;color:#64748b">/ 966 Samples</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Best Observed k0</div>
            <div class="metric-value" style="color:#f59e0b">{best_k0_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# 3-Column Core Layout
col_hypo, col_map, col_action = st.columns([1.1, 1.6, 1.3])

# LEFT COLUMN: Scientific Hypotheses
with col_hypo:
    st.subheader("Scientific Hypotheses")
    n_events = len(engine.hypothesis_engine.evidence_events)
    if n_events == 0:
        st.caption("Prior / No sequential evidence yet")
    else:
        st.caption(f"Evidence weight after {n_events} adaptive experiment(s)")

    hypotheses = engine.hypothesis_engine.hypotheses
    for hid, h in hypotheses.items():
        status_color = "#10b981" if h.status.value == "SUPPORTED" else ("#ef4444" if h.status.value == "WEAKENED" else "#3b82f6")
        st.markdown(
            f"""
            <div style="background:#1a1f2c;border:1px solid #2d3748;border-radius:10px;padding:14px;margin-bottom:12px">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-weight:700;color:#f8fafc;font-size:1.05rem">{h.title}</span>
                    <span style="background:{status_color}22;color:{status_color};border:1px solid {status_color};border-radius:4px;padding:2px 8px;font-size:0.75rem;font-weight:700">{h.status.value}</span>
                </div>
                <div style="margin-top:8px;font-size:0.88rem;color:#cbd5e1;line-height:1.4">{h.statement}</div>
                <div style="margin-top:12px">
                    <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#94a3b8;margin-bottom:4px">
                        <span>Evidence Weight</span>
                        <span style="font-weight:700;color:#f8fafc">{h.belief_score*100:.1f}%</span>
                    </div>
                    <div style="background:#334155;border-radius:4px;height:8px;overflow:hidden">
                        <div style="background:{status_color};width:{h.belief_score*100}%;height:100%"></div>
                    </div>
                </div>
                <div style="display:flex;gap:12px;margin-top:10px;font-size:0.75rem;color:#94a3b8">
                    <span>Supporting: <b style="color:#10b981">{h.supporting_evidence_count}</b></span>
                    <span>Contradicting: <b style="color:#ef4444">{h.contradicting_evidence_count}</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# CENTER COLUMN: Materials Landscape
with col_map:
    st.subheader("Materials Composition Landscape")
    st.caption("966 Measured Au-Ir-Rh Physical Candidates")

    landscape_df = engine.get_landscape_dataframe()
    last_rec = st.session_state.last_rec

    color_discrete_map = {
        "Unobserved": "#475569",
        "XRD Characterized": "#10b981",
        "Property Tested": "#3b82f6",
        "Both XRD & Property": "#a855f7",
    }

    fig = px.scatter(
        landscape_df,
        x="Au",
        y="Ir",
        color="status",
        color_discrete_map=color_discrete_map,
        hover_data=["candidate_id", "Library", "Area", "Au", "Ir", "Rh", "predicted_k0", "measured_k0"],
        labels={"Au": "Au (at.%)", "Ir": "Ir (at.%)"},
        title="Ternary Composition Space (Au vs Ir)",
    )
    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#cbd5e1",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        height=430,
    )

    if last_rec is not None:
        rec_cid = last_rec.action.candidate_id
        rec_row = landscape_df[landscape_df["candidate_id"] == rec_cid]
        if not rec_row.empty:
            fig.add_trace(
                go.Scatter(
                    x=rec_row["Au"],
                    y=rec_row["Ir"],
                    mode="markers+text",
                    marker=dict(
                        size=18,
                        color="#f59e0b",
                        symbol="star",
                        line=dict(width=2, color="#ffffff"),
                    ),
                    text=[f" ⭐ Recommended: {rec_cid}"],
                    textposition="top center",
                    name="Recommended Action",
                )
            )

    st.plotly_chart(fig, use_container_width=True)

# RIGHT COLUMN: Next Experiment Recommendation Card
with col_action:
    st.subheader("Next Best Experiment")
    st.caption("Autonomous Scientific Recommendation")

    col_btn_ask, col_btn_run = st.columns(2)
    with col_btn_ask:
        if st.button("💡 Ask AI Scientist", use_container_width=True, type="primary"):
            rec, perspectives = engine.propose_next_experiment()
            st.session_state.last_rec = rec
            st.session_state.last_perspectives = perspectives
            st.rerun()

    with col_btn_run:
        can_run = st.session_state.last_rec is not None
        if st.button("⚡ Run Experiment", use_container_width=True, disabled=not can_run):
            if st.session_state.last_rec:
                outcome_summary = engine.execute_experiment(st.session_state.last_rec.action)
                st.session_state.last_outcome = outcome_summary
                st.session_state.last_rec = None
                st.session_state.last_perspectives = []
                st.rerun()

    rec = st.session_state.last_rec
    if rec is not None:
        act_type = rec.action.action_type.value
        act_color = "#10b981" if act_type == "XRD" else "#3b82f6"

        st.markdown(
            f"""
            <div class="rec-card">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-size:0.8rem;letter-spacing:0.05em;color:#94a3b8">RECOMMENDED ACTION</span>
                    <span style="background:{act_color}22;color:{act_color};border:1px solid {act_color};border-radius:4px;padding:3px 10px;font-weight:700;font-size:0.85rem">{act_type}</span>
                </div>
                <div style="font-size:1.4rem;font-weight:700;color:#f8fafc;margin-top:6px">{rec.action.candidate_id}</div>
                <div style="font-size:0.88rem;color:#cbd5e1;margin-top:8px;line-height:1.4">{rec.rationale}</div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:14px">
                    <div style="background:#0f172a;padding:8px;border-radius:6px;text-align:center">
                        <div style="font-size:0.7rem;color:#94a3b8">TOTAL VALUE</div>
                        <div style="font-weight:700;color:#f8fafc;font-size:1.1rem">{rec.total_value:.3f}</div>
                    </div>
                    <div style="background:#0f172a;padding:8px;border-radius:6px;text-align:center">
                        <div style="font-size:0.7rem;color:#10b981">INFO VALUE</div>
                        <div style="font-weight:700;color:#10b981;font-size:1.1rem">{rec.scientific_information_value:.3f}</div>
                    </div>
                    <div style="background:#0f172a;padding:8px;border-radius:6px;text-align:center">
                        <div style="font-size:0.7rem;color:#3b82f6">DISCOVERY</div>
                        <div style="font-weight:700;color:#3b82f6;font-size:1.1rem">{rec.discovery_value:.3f}</div>
                    </div>
                </div>
                <div class="falsification-box">
                    <div style="font-size:0.75rem;font-weight:700;color:#ef4444;text-transform:uppercase">Falsification Criterion ({rec.hypothesis_id})</div>
                    <div style="font-size:0.82rem;color:#fca5a5;margin-top:4px">{rec.falsification_criterion}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("⚖️ Why Not Another Candidate? (Counterfactuals)", expanded=False):
            for alt in rec.alternatives:
                st.markdown(
                    f"**{alt.action_type.value} on {alt.candidate_id}** (Net Score: `{alt.total_value:.2f}`): {alt.contrastive_rationale}"
                )

        with st.expander("👥 Multi-Agent Scientific Reasoning", expanded=False):
            for p in st.session_state.last_perspectives:
                st.markdown(f"**{p.role_name}**: *{p.headline}*")
                st.caption(p.body)
                for pt in p.key_points:
                    st.markdown(f"- {pt}")
                st.divider()

    else:
        st.info("Click **💡 Ask AI Scientist** to generate the optimal next experiment proposal.")

st.divider()

# Post-Experiment Reveal Modal / Area
last_outcome = st.session_state.last_outcome
if last_outcome is not None:
    st.subheader("📊 Latest Experimental Observation Revealed")
    out_act = last_outcome["action"]
    out_data = last_outcome["outcome"]["revealed_data"]

    if out_act["action_type"] == "XRD":
        r1, r2 = st.columns([1.8, 1.2])
        with r1:
            tt = out_data.get("two_theta", [])
            inte = out_data.get("intensity", [])
            fig_xrd = go.Figure()
            fig_xrd.add_trace(
                go.Scatter(
                    x=tt,
                    y=inte,
                    mode="lines",
                    line=dict(color="#10b981", width=1.5),
                    name="Measured Diffractogram",
                )
            )
            fig_xrd.update_layout(
                title=f"Measured Real XRD Diffractogram: {out_act['candidate_id']}",
                xaxis_title="2θ (degrees)",
                yaxis_title="Intensity (counts)",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="#cbd5e1",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_xrd, use_container_width=True)

        with r2:
            st.markdown("#### What Changed in Scientific Belief?")
            deltas = last_outcome.get("belief_deltas", {})
            for hid, delta in deltas.items():
                d_color = "#10b981" if delta > 0 else ("#ef4444" if delta < 0 else "#94a3b8")
                d_symbol = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
                st.markdown(
                    f"**{hid}**: `{last_outcome['before_beliefs'][hid]*100:.1f}%` → `{last_outcome['after_beliefs'][hid]*100:.1f}%` "
                    f"(<span style='color:{d_color};font-weight:700'>{d_symbol} {abs(delta)*100:.1f}%</span>)",
                    unsafe_allow_html=True,
                )
            st.success("✅ Structural evidence assimilated into ephemerally fitted surrogate and hypothesis engine.")

    elif out_act["action_type"] == "PROPERTY":
        p1, p2 = st.columns([1, 1])
        with p1:
            st.metric("Measured k0 [cm/s]", f"{out_data.get('k0', 0.0):.6f}")
            st.caption(f"Candidate: {out_act['candidate_id']} | Library: {last_outcome['outcome']['provenance'].get('library', '')}")
        with p2:
            st.metric("Campaign Best Observed k0", f"{last_outcome.get('best_observed_k0', 0.0):.6f}")
            st.info("Performance data recorded in tamper-evident ledger.")

st.divider()

# Timeline & Roadmap Bottom Section
btm_left, btm_right = st.columns([1.6, 1.1])

with btm_left:
    st.subheader("📜 Experiment Action Timeline")
    if engine.timeline:
        t_df = pd.DataFrame(engine.timeline)
        st.dataframe(t_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No experiments recorded yet in active campaign.")

with btm_right:
    st.subheader("🔋 Battery Materials R&D Roadmap")
    st.markdown(
        """
        <div style="background:#1a1f2c;border:1px solid #312e81;border-radius:10px;padding:16px">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-weight:700;color:#c7d2fe">CURRENT VALIDATION</span>
                <span class="roadmap-badge">Active Au-Ir-Rh MVP</span>
            </div>
            <p style="font-size:0.85rem;color:#cbd5e1;margin-top:6px">
                Au-Ir-Rh real multimodal dataset: Composition + real XRD + SECCM electrochemical kinetics.
            </p>
            <hr style="border-color:#312e81;margin:10px 0">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-weight:700;color:#f59e0b">FUTURE TARGET APPLICATION</span>
                <span class="roadmap-badge" style="background:#78350f;color:#fde68a">Future Work</span>
            </div>
            <p style="font-size:0.85rem;color:#cbd5e1;margin-top:6px">
                <b>Battery Materials Discovery:</b> Extending multimodal actions to:
                <br>• In-situ XRD / PDF phase tracking
                <br>• SEM morphology & FIB-SEM 3D reconstruction
                <br>• XPS chemical state & SEI passivation
                <br>• EIS impedance spectroscopy
                <br>• Fast-charging cycling lifecycle testing
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
