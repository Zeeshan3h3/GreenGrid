import os
import tempfile

from pipeline import robust_to_uint8, run_pipeline
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st

st.set_page_config(page_title="CanopyAI | Forest Crown & Area Platform", layout="wide")

st.markdown("""
<style>
/* App Background & Typography */
.stApp {
    background-color: #0d0f13;
    color: #e0e0e0;
}

/* Glassmorphism Metric Cards */
div[data-testid="stMetric"], div[data-testid="metric-container"] {
    background: rgba(25, 25, 30, 0.45);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease-in-out;
}

/* Hover Animations */
div[data-testid="stMetric"]:hover, div[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0, 255, 128, 0.15);
    border: 1px solid rgba(0, 255, 128, 0.3);
}

/* Label styling */
div[data-testid="stMetricLabel"] {
    color: #a0a4b0;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
}

/* Primary Number styling */
div[data-testid="stMetricValue"] {
    color: #ffffff;
    font-weight: 800;
    font-size: 2.4rem;
}

/* Download Button Polish */
.stButton > button {
    border-radius: 8px;
    font-weight: bold;
    transition: transform 0.2s;
}
.stButton > button:hover {
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

st.title("CanopyAI: Automated Crown Detection & Canopy Estimator")
st.markdown(
    "Dual-Model Engine: **DeepForest (RetinaNet)** for crown counting + "
    "**Restor OAM-TCD (Segformer)** for pixel-level canopy coverage."
)

# ---- Sidebar Controls ----
st.sidebar.header("Inference Controls")
conf_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05,
    help="Higher values eliminate false positives on ground cover/sand; "
    "lower values detect smaller saplings.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Diagnostic Legend")
st.sidebar.markdown("**Green Mask:** Segformer canopy coverage (pixel-level)")
st.sidebar.markdown("**Yellow Box:** DeepForest individual tree crown identification")

st.sidebar.markdown("---")
with st.sidebar.expander("Known limitations"):
    st.markdown(
        "- The canopy model is trained on OAM-TCD and generalizes well to typical "
        "forest/tree-cluster imagery, but performs poorly on sparse arid/scrub terrain. "
        "Verify visually on the overlay before trusting the % figure on unfamiliar scenes.\n"
        "- Cloud cover and nodata regions in the source image are not automatically "
        "excluded from the analysis area; pick an AOI that avoids them.\n"
        "- Individual tree count is most reliable in open-to-moderate canopy density; "
        "very dense closed-canopy areas with heavily touching crowns will undercount."
    )

# ---- File Inputs ----
col_in1, col_in2 = st.columns(2)
with col_in1:
    img_file = st.file_uploader(
        "Upload Orthomosaic / GeoTIFF (Required)", type=["tif", "tiff"]
    )
with col_in2:
    kml_file = st.file_uploader(
        "Upload Area of Interest / KML (Optional)",
        type=["kml"],
        help="If omitted, the pipeline analyzes the entire GeoTIFF bounding box.",
    )

if img_file:
    if st.button("Run Forest Analysis", type="primary"):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "input.tif")
            with open(img_path, "wb") as f:
                f.write(img_file.read())

            kml_path = None
            if kml_file:
                kml_path = os.path.join(tmpdir, "boundary.kml")
                with open(kml_path, "wb") as f:
                    f.write(kml_file.read())

            with st.spinner(
                "Processing raster, reprojection, and dual-model inference — "
                "large images take longer on CPU..."
            ):
                metrics, final_path, tree_mask, tree_df, zip_data = run_pipeline(
                    img_path, kml_path, conf_threshold=conf_threshold, workdir=tmpdir
                )

            st.success("Analysis complete")

            # ---- 1. Metric Scorecards ----
            st.subheader("Key Inventory & Ecological Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Stem Count", f"{metrics['tree_count']:,} trees")
            m2.metric(
                "Canopy Area",
                f"{metrics['canopy_area_ha']:.2f} ha",
                f"{metrics['canopy_area_m2']:,.0f} m\u00b2",
            )
            m3.metric("Canopy Cover", f"{metrics['canopy_pct']:.1f}%")
            m4.metric(
                "Est. CO\u2082 Storage", f"{metrics['carbon_seq_tons']:,.1f} tCO\u2082"
            )
            m5.metric(
                "Stand Density",
                f"{metrics['stand_density']:.0f} / ha",
                metrics["health_status"],
            )

            # ---- 2. Dual-Layer Visualization ----
            st.subheader("Dual-Model Inspection Overlay")
            with rasterio.open(final_path) as src:
                rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
            rgb_disp = robust_to_uint8(rgb.astype(np.float32))

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(rgb_disp)
            # Layer 1: Green Segformer mask
            ax.imshow(tree_mask, cmap="Greens", alpha=0.45)
            # Layer 2: Yellow DeepForest bounding boxes
            for _, row in tree_df.iterrows():
                rect = patches.Rectangle(
                    (row["xmin"], row["ymin"]),
                    row["xmax"] - row["xmin"],
                    row["ymax"] - row["ymin"],
                    linewidth=1.3,
                    edgecolor="#FFFF00",
                    facecolor="none",
                )
                ax.add_patch(rect)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

            # ---- 3. One-Click GIS Bundle Export ----
            st.subheader("Deliverables & GIS Interoperability")
            st.download_button(
                label="Download Complete GIS Bundle (.zip)",
                data=zip_data,
                file_name="forest_inventory_bundle.zip",
                mime="application/zip",
                help="Includes tree_inventory.csv, tree_detections.geojson, "
                "canopy_polygons.geojson, high-res visual overlay, and summary metrics.",
            )
else:
    st.info(
        "Upload a GeoTIFF (required) and optionally a KML boundary, "
        "then click **Run Forest Analysis** to begin."
    )
