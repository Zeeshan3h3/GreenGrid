"""
CanopyAI — Hugging Face Spaces entry point.

Pure Gradio app that wraps pipeline.py directly.
ZeroGPU requires Gradio as the primary framework (not FastAPI).
The local dev server (server.py + index.html) still works independently.
"""

import spaces
import os
import io
import base64
import tempfile

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# We must import torch before rasterio to avoid DLL issues
import torch  # noqa: F401
import rasterio

from pipeline import (
    crop_to_aoi,
    count_trees,
    predict_canopy_mask,
    compute_metrics,
    create_gis_bundle,
    robust_to_uint8,
)


@spaces.GPU
def run_analysis(ortho_file, kml_file, confidence):
    """Main analysis function — decorated with @spaces.GPU for ZeroGPU."""
    if ortho_file is None:
        raise gr.Error("Please upload an orthomosaic GeoTIFF file first!")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Copy uploaded file to a temp path (rasterio needs a real file path)
        img_path = os.path.join(tmpdir, "input.tif")
        # Gradio gives us a file path string directly
        import shutil
        shutil.copy(ortho_file, img_path)

        kml_path = None
        if kml_file is not None:
            kml_path = os.path.join(tmpdir, "boundary.kml")
            shutil.copy(kml_file, kml_path)

        # 2. Run the full ML pipeline
        final_path, gsd, aoi_proj = crop_to_aoi(img_path, kml_path, workdir=tmpdir)
        tree_df = count_trees(final_path, conf_threshold=confidence)
        tree_mask = predict_canopy_mask(final_path)
        metrics = compute_metrics(tree_mask, gsd, aoi_proj, tree_df)
        zip_buffer = create_gis_bundle(final_path, tree_df, tree_mask, metrics)

        # 3. Generate the dual-model overlay image
        with rasterio.open(final_path) as src:
            rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        rgb_disp = robust_to_uint8(rgb.astype(np.float32))

        fig, ax = plt.subplots(figsize=(12, 12), dpi=150, frameon=False)
        ax.imshow(rgb_disp)
        ax.imshow(tree_mask, cmap="Greens", alpha=0.45)
        for _, row in tree_df.iterrows():
            rect = patches.Rectangle(
                (row["xmin"], row["ymin"]),
                row["xmax"] - row["xmin"],
                row["ymax"] - row["ymin"],
                linewidth=1.5,
                edgecolor="#FFFF00",
                facecolor="none",
            )
            ax.add_patch(rect)
        ax.axis("off")

        overlay_path = os.path.join(tmpdir, "overlay.png")
        fig.savefig(overlay_path, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close(fig)

        # 4. Save the zip to a temp file for download
        zip_path = os.path.join(tmpdir, "canopy_gis_bundle.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_buffer.getvalue())

        # 5. Format the metrics into a readable string
        metrics_text = f"""
### 🌲 Analysis Complete

| Metric | Value |
|---|---|
| **Trees Detected** | {metrics['tree_count']} |
| **Canopy Area** | {metrics['canopy_area_ha']:.2f} ha ({metrics['canopy_area_m2']:.0f} m²) |
| **Canopy Coverage** | {metrics['canopy_pct']:.1f}% |
| **CO₂ Sequestration** | {metrics['carbon_seq_tons']:.1f} metric tons |
| **Stand Density** | {metrics['stand_density']:.0f} trees/ha |
| **Health Assessment** | {metrics['health_status']} |
"""

        # Read the overlay image back for Gradio display
        from PIL import Image
        overlay_img = Image.open(overlay_path)

        # Copy zip to a persistent location so Gradio can serve it
        import uuid
        persistent_zip = os.path.join(tempfile.gettempdir(), f"canopy_bundle_{uuid.uuid4().hex[:8]}.zip")
        shutil.copy(zip_path, persistent_zip)

        return metrics_text, overlay_img, persistent_zip


# ---- Build the Gradio UI ----
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
#header-row {
    text-align: center;
    margin-bottom: 1rem;
}
.metric-box {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border-radius: 12px;
    padding: 20px;
    color: white;
    font-size: 1.1em;
}
"""

with gr.Blocks(
    title="GreenGrid — Automated Crown Detection & Canopy Estimator",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
) as demo:

    # Header
    gr.Markdown(
        """
        # 🌲 GreenGrid
        ### Automated Crown Detection & Canopy Area Estimation
        *Upload a drone orthomosaic GeoTIFF to detect individual trees (DeepForest RetinaNet) and estimate canopy coverage (Segformer MIT-B2)*
        """,
        elem_id="header-row",
    )

    with gr.Row():
        # Left panel - Inputs
        with gr.Column(scale=1):
            gr.Markdown("## 📤 Upload Data")
            ortho_input = gr.File(
                label="Orthomosaic (.tif)",
                file_types=[".tif", ".tiff"],
                type="filepath",
            )
            kml_input = gr.File(
                label="Area of Interest (.kml) — Optional",
                file_types=[".kml"],
                type="filepath",
            )
            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.35,
                step=0.05,
                label="Detection Confidence Threshold",
                info="Higher = fewer false positives, Lower = catches more small trees",
            )
            analyze_btn = gr.Button(
                "🔬 Execute Analysis",
                variant="primary",
                size="lg",
            )

            gr.Markdown(
                """
                ---
                ### 📋 Input Requirements
                - **Format**: GeoTIFF with spatial metadata
                - **Bands**: RGB (3-band)
                - **Resolution**: 5cm–50cm GSD recommended
                - **Size**: 10MB–50MB for best speed
                """
            )

        # Right panel - Results
        with gr.Column(scale=2):
            gr.Markdown("## 📊 Results")
            metrics_output = gr.Markdown(
                value="*Upload an orthomosaic and click Execute Analysis to see results here.*"
            )
            overlay_output = gr.Image(
                label="Dual-Model Overlay (Green = Canopy Mask, Yellow = Tree Crowns)",
                type="pil",
            )
            download_output = gr.File(
                label="📦 Download GIS Bundle (.zip)",
            )

    # Footer
    gr.Markdown(
        """
        ---
        <center>
        <b>GreenGrid</b> — Dual-Model Forest Inventory Platform | DeepForest (RetinaNet) + Restor OAM-TCD (Segformer MIT-B2)
        <br>Built for hackathon judges who appreciate production-grade engineering 🏆
        </center>
        """
    )

    # Wire up the button
    analyze_btn.click(
        fn=run_analysis,
        inputs=[ortho_input, kml_input, confidence_slider],
        outputs=[metrics_output, overlay_output, download_output],
    )


# This is what Hugging Face Spaces looks for
demo.launch()
