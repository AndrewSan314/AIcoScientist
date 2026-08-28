import json
import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sem_analysis import analyze_sem_image, image_from_bytes


MASTER = ROOT / "data" / "processed" / "master_dataset.csv"
METRICS = ROOT / "outputs" / "model_metrics.json"
IMPORTANCE = ROOT / "outputs" / "feature_importance.csv"
RECS = ROOT / "outputs" / "recommendations.csv"
SEM_IMAGES = ROOT / "data" / "raw" / "sem_images"
SAM_CHECKPOINT = ROOT / "models" / "sam_vit_b_01ec64.pth"


@st.cache_resource(show_spinner="Loading SAM model (first time only)...")
def _cached_sam_generator(checkpoint_path: str):
    from src.sem_analysis import load_sam_generator

    return load_sam_generator(checkpoint_path, "vit_b")


@st.cache_data(show_spinner="Analyzing SEM image...")
def _analyze_sem_cached(image_bytes: bytes, enhance: bool, use_sam: bool, checkpoint_path: str):
    image = image_from_bytes(image_bytes)
    generator = None
    checkpoint = None
    if use_sam and checkpoint_path and Path(checkpoint_path).exists():
        generator = _cached_sam_generator(checkpoint_path)
        if generator is not None:
            checkpoint = checkpoint_path
    return analyze_sem_image(
        image,
        enhance=enhance,
        sam_checkpoint=checkpoint,
        sam_generator=generator,
    )


st.set_page_config(page_title="Battery AI Co-Scientist", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #f6f7f5;
        --panel: #ffffff;
        --ink: #17201c;
        --muted: #66736c;
        --line: #dde3df;
        --accent: #147a58;
        --accent-soft: #e6f2ed;
        --amber-soft: #f7efe0;
    }

    .stApp {
        background: var(--bg);
        color: var(--ink);
    }

    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 1320px;
        padding: 2.2rem 2rem 3rem;
    }

    h1, h2, h3, p {
        letter-spacing: 0;
    }

    .hero {
        display: grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(320px, .8fr);
        gap: 28px;
        align-items: stretch;
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        padding: 28px 0;
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: clamp(2.3rem, 4vw, 4.6rem);
        line-height: .96;
        font-weight: 760;
        letter-spacing: -0.045em;
        margin: 0 0 18px;
        max-width: 980px;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.65;
        max-width: 760px;
        margin: 0;
    }

    .hero-panel,
    .soft-panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        box-shadow: 0 18px 44px -34px rgba(20, 34, 28, .45);
    }

    .hero-panel {
        padding: 22px;
        display: grid;
        gap: 14px;
    }

    .label {
        color: var(--muted);
        font-size: .75rem;
        font-weight: 720;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .panel-value {
        font-size: 2rem;
        line-height: 1;
        font-weight: 760;
        letter-spacing: -0.035em;
    }

    .accent {
        color: var(--accent);
    }

    .status-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 12px 0 26px;
    }

    .metric-tile {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px;
    }

    .metric-tile .value {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 1.6rem;
        font-weight: 740;
        letter-spacing: -0.04em;
        margin-top: 3px;
    }

    .soft-panel {
        padding: 22px;
        margin-bottom: 18px;
    }

    .section-head {
        display: flex;
        justify-content: space-between;
        align-items: end;
        gap: 16px;
        margin-bottom: 16px;
    }

    .section-head h2 {
        font-size: 1.45rem;
        line-height: 1.1;
        margin: 0;
        letter-spacing: -0.03em;
    }

    .section-head p {
        color: var(--muted);
        margin: 0;
        max-width: 520px;
        line-height: 1.45;
    }

    .recipe-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 8px 0 18px;
    }

    .recipe {
        background: linear-gradient(180deg, #ffffff 0%, #f9fbfa 100%);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px;
    }

    .recipe.lead {
        border-color: rgba(20, 122, 88, .42);
        box-shadow: inset 0 3px 0 var(--accent);
    }

    .rank {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        color: var(--accent);
        font-size: .82rem;
        font-weight: 760;
    }

    .retention {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 2rem;
        font-weight: 780;
        letter-spacing: -0.05em;
        margin: 8px 0 12px;
    }

    .recipe dl {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px 12px;
        margin: 0 0 14px;
    }

    .recipe dt {
        color: var(--muted);
        font-size: .73rem;
    }

    .recipe dd {
        margin: 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: .88rem;
        color: var(--ink);
    }

    .reason {
        color: var(--muted);
        min-height: 46px;
        line-height: 1.45;
        margin: 0;
    }

    .confidence {
        display: inline-flex;
        align-items: center;
        border: 1px solid rgba(20, 122, 88, .28);
        background: var(--accent-soft);
        color: var(--accent);
        border-radius: 999px;
        padding: 4px 10px;
        font-size: .76rem;
        font-weight: 720;
        margin-top: 12px;
    }

    .empty {
        border: 1px dashed #b9c4bd;
        border-radius: 14px;
        padding: 28px;
        background: #ffffffb8;
        color: var(--muted);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--line);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 16px;
        color: var(--muted);
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent);
        background: #ffffff;
        border: 1px solid var(--line);
        border-bottom-color: #ffffff;
    }

    .stButton > button {
        border-radius: 999px;
        border: 1px solid var(--line);
        background: #ffffff;
        color: var(--ink);
        transition: transform .18s ease, border-color .18s ease, color .18s ease;
    }

    .stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }

    .stButton > button:active {
        transform: translateY(1px) scale(.99);
    }

    @media (max-width: 900px) {
        [data-testid="stAppViewContainer"] > .main .block-container {
            padding: 1.2rem 1rem 2rem;
        }
        .hero,
        .status-row,
        .recipe-grid {
            grid-template-columns: 1fr;
        }
        .section-head {
            display: block;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


df = load_csv(MASTER)
metrics = load_json(METRICS)
importance = load_csv(IMPORTANCE)
recs = load_csv(RECS)

sample_count = 0 if df is None else len(df)
target = "retention_100" if metrics is None else metrics.get("target", "retention_100")
top_retention = None if recs is None or recs.empty else float(recs.iloc[0]["predicted_retention"])
top_retention_display = "Missing" if top_retention is None else f"{top_retention:.2f}%"
top_recipe = None if recs is None or recs.empty else recs.iloc[0]

st.markdown(
    f"""
    <section class="hero">
        <div>
            <div class="label">AI Co-Scientist MVP</div>
            <h1 class="hero-title">Si/MXene electrode optimization, ready for the next lab round.</h1>
            <p class="hero-copy">
                Fabrication parameters, SEM features, EDX composition, and electrochemical results are merged into one
                model-ready view. The model ranks candidate recipes for capacity retention while keeping the scientist in control.
            </p>
        </div>
        <aside class="hero-panel">
            <div>
                <div class="label">Best predicted retention</div>
                <div class="panel-value accent">{top_retention_display}</div>
            </div>
            <div>
                <div class="label">Primary target</div>
                <div class="panel-value">{escape(target)}</div>
            </div>
            <div>
                <div class="label">Current recommendation</div>
                <p class="hero-copy">
                    {escape("Run the pipeline first." if top_recipe is None else f"Si {top_recipe['si_content']}%, MXene {top_recipe['mxene_content']}%, alginate {top_recipe['alginate_content']}%")}
                </p>
            </div>
        </aside>
    </section>
    """,
    unsafe_allow_html=True,
)

mae = "Missing" if metrics is None else f"{metrics['mae']:.2f}"
rmse = "Missing" if metrics is None else f"{metrics['rmse']:.2f}"
r2 = "Missing" if metrics is None else f"{metrics['r2']:.2f}"

st.markdown(
    f"""
    <div class="status-row">
        <div class="metric-tile"><div class="label">Samples</div><div class="value">{sample_count}</div></div>
        <div class="metric-tile"><div class="label">MAE</div><div class="value">{mae}</div></div>
        <div class="metric-tile"><div class="label">RMSE</div><div class="value">{rmse}</div></div>
        <div class="metric-tile"><div class="label">R2</div><div class="value">{r2}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Recommendations", "Dataset", "Model", "Feature importance", "SEM imaging"])

with tabs[0]:
    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="label">Decision queue</div>
                <h2>Top fabrication candidates</h2>
            </div>
            <p>Recipes are ranked with a lightweight GP/UCB Bayesian Optimization score over the discrete recipe grid. This is a decision aid, not final scientific validation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if recs is None or recs.empty:
        st.markdown(
            '<div class="empty">No recommendations yet. Run <code>python run_pipeline.py</code> from the project root.</div>',
            unsafe_allow_html=True,
        )
    else:
        card_cols = st.columns(len(recs))
        for _, row in recs.iterrows():
            recipe_class = "recipe lead" if int(row["rank"]) == 1 else "recipe"
            uncertainty = float(row.get("predicted_retention_std", 0.0))
            acquisition = float(row.get("acquisition_score", row["predicted_retention"]))
            chem_score = float(row.get("chemistry_score", 0.0))
            vol_risk = float(row.get("volume_expansion_risk", 0.0))
            conductive = float(row["mxene_content"]) + float(row["carbon_content"])
            card = (
                f'<div class="{recipe_class}">'
                f'<div class="rank">RANK {int(row["rank"]):02d}</div>'
                f'<div class="retention">{row["predicted_retention"]:.2f}%</div>'
                '<dl>'
                f'<div><dt>Si</dt><dd>{row["si_content"]}%</dd></div>'
                f'<div><dt>MXene</dt><dd>{row["mxene_content"]}%</dd></div>'
                f'<div><dt>Alginate</dt><dd>{row["alginate_content"]}%</dd></div>'
                f'<div><dt>Carbon</dt><dd>{row["carbon_content"]}%</dd></div>'
                f'<div><dt>Conductive</dt><dd>{conductive:.0f}%</dd></div>'
                f'<div><dt>Chem score</dt><dd>{chem_score:.2f}</dd></div>'
                f'<div><dt>Expansion risk</dt><dd>{vol_risk:.2f}</dd></div>'
                f'<div><dt>Drying</dt><dd>{row["drying_temp"]} C</dd></div>'
                f'<div><dt>Mixing</dt><dd>{row["mixing_time"]} min</dd></div>'
                f'<div><dt>Uncertainty</dt><dd>{uncertainty:.2f}</dd></div>'
                '</dl>'
                f'<p class="reason">{escape(str(row["reason"]))}</p>'
                f'<span class="confidence">{escape(str(row["confidence"]).upper())} CONFIDENCE</span>'
                '</div>'
            )
            with card_cols[int(row["rank"]) - 1]:
                st.markdown(card, unsafe_allow_html=True)
                st.segmented_control(
                    f"Decision for rank {int(row['rank'])}",
                    ["Accept", "Modify", "Reject"],
                    key=f"decision_{int(row['rank'])}",
                    label_visibility="collapsed",
                    width="stretch",
                )
        st.dataframe(recs, use_container_width=True, hide_index=True)

with tabs[1]:
    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="label">Merged lab view</div>
                <h2>Master dataset</h2>
            </div>
            <p>Process, SEM, EDX, and electrochemical data joined by sample ID.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if df is None:
        st.markdown(
            '<div class="empty">Master dataset is missing. Run <code>python run_pipeline.py</code>.</div>',
            unsafe_allow_html=True,
        )
    else:
        left, right = st.columns([1.15, 1])
        with left:
            st.dataframe(df, use_container_width=True, hide_index=True)
        with right:
            st.scatter_chart(
                df,
                x="mxene_content",
                y="retention_100",
                color="alginate_content",
                use_container_width=True,
            )

with tabs[2]:
    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="label">Baseline and surrogate</div>
                <h2>Performance snapshot</h2>
            </div>
            <p>RandomForest remains the baseline predictor; Gaussian Process adds uncertainty for recommendation ranking.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if metrics is None:
        st.markdown(
            '<div class="empty">Model metrics are missing. Run <code>python run_pipeline.py</code>.</div>',
            unsafe_allow_html=True,
        )
    else:
        cols = st.columns([1, 1, 1, 1.4])
        cols[0].metric("MAE", f"{metrics['mae']:.2f}")
        cols[1].metric("RMSE", f"{metrics['rmse']:.2f}")
        cols[2].metric("R2", f"{metrics['r2']:.2f}")
        cols[3].metric("Target", metrics["target"])
        gp_metrics = metrics.get("gp_metrics")
        if gp_metrics:
            gp_cols = st.columns(3)
            gp_cols[0].metric("GP MAE", f"{gp_metrics['mae']:.2f}")
            gp_cols[1].metric("GP RMSE", f"{gp_metrics['rmse']:.2f}")
            gp_cols[2].metric("Acquisition", "UCB beta=1.0")
        st.json(metrics)

with tabs[3]:
    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="label">Model explanation</div>
                <h2>Feature importance</h2>
            </div>
            <p>Top drivers show which fabrication and material signals influence predicted retention in this synthetic MVP.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if importance is None:
        st.markdown(
            '<div class="empty">Feature importance is missing. Run <code>python run_pipeline.py</code>.</div>',
            unsafe_allow_html=True,
        )
    else:
        top_features = importance.head(12).copy()
        st.bar_chart(top_features.set_index("feature"), use_container_width=True)
        st.dataframe(importance, use_container_width=True, hide_index=True)

with tabs[4]:
    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="label">Microscopy flow</div>
                <h2>SEM segmentation and crack metrics</h2>
            </div>
            <p>Upload an SEM image or use a demo image. The pipeline enhances contrast, segments material regions, detects dark crack-like structures, and reports measured image metrics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_col, option_col = st.columns([1.2, 1])
    with upload_col:
        uploaded = st.file_uploader("SEM image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    with option_col:
        demo_paths = sorted(
            path for path in SEM_IMAGES.iterdir()
            if SEM_IMAGES.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )
        demo_name = st.selectbox("Demo image", ["None", *[path.name for path in demo_paths]])
        enhance = st.checkbox("Enhance before segmentation", value=True)
        use_sam = st.checkbox(
            "Use SAM checkpoint when available",
            value=SAM_CHECKPOINT.exists(),
            disabled=not SAM_CHECKPOINT.exists(),
        )

    image = None
    image_bytes = None
    source_name = None
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        image = image_from_bytes(image_bytes)
        source_name = uploaded.name
    elif demo_name != "None":
        demo_path = next(path for path in demo_paths if path.name == demo_name)
        image_bytes = demo_path.read_bytes()
        image = image_from_bytes(image_bytes)
        source_name = demo_path.name

    if image is None:
        st.markdown(
            '<div class="empty">Upload an SEM image or run <code>python -m src.fetch_sem_demo</code> to load demo images.</div>',
            unsafe_allow_html=True,
        )
    else:
        checkpoint = str(SAM_CHECKPOINT)
        with st.spinner("Analyzing SEM image..."):
            result = _analyze_sem_cached(image_bytes, enhance, use_sam, checkpoint)
        metrics_row = result["metrics"]

        if use_sam and metrics_row["segmentation_method"] == "Otsu threshold":
            st.warning(
                "SAM was enabled but the pipeline fell back to Otsu. "
                "Check that segment-anything is installed and the checkpoint path is valid."
            )

        st.caption(
            f"{source_name} | segmentation: {metrics_row['segmentation_method']} | enhancement: {metrics_row['enhancement']}"
        )
        left, right = st.columns(2)
        with left:
            st.image(result["working_image"], caption="Input / enhanced SEM", use_container_width=True)
        with right:
            st.image(result["overlay"], caption="Material mask + crack overlay", use_container_width=True)

        cols = st.columns(5)
        cols[0].metric("Crack area", f"{metrics_row['crack_area_fraction']:.3f}")
        cols[1].metric("Crack count", f"{metrics_row['crack_count']}")
        cols[2].metric("Length density", f"{metrics_row['crack_length_density']:.3f}")
        cols[3].metric("Mean width", f"{metrics_row['mean_crack_width']:.2f} px")
        cols[4].metric("Particle area", f"{metrics_row['particle_area_fraction']:.3f}")
        st.dataframe(pd.DataFrame([metrics_row]), use_container_width=True, hide_index=True)
