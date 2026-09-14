"""
FastAPI backend for CanopyAI.

Wraps pipeline.py in a REST API and serves the static index.html frontend.
Models are loaded once on first request via module-level singletons in pipeline.py.
"""

import io
import os
import base64
import tempfile

# torch MUST be imported before rasterio on Windows to avoid DLL conflicts
import torch  # noqa: F401

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import rasterio

from pipeline import (
    crop_to_aoi,
    count_trees,
    predict_canopy_mask,
    compute_metrics,
    create_gis_bundle,
    robust_to_uint8,
)

app = FastAPI(title="CanopyAI", version="1.0.0")


# ---- Serve the frontend ----
@app.get("/")
def serve_frontend():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "index.html"),
        media_type="text/html",
    )


# ---- ML Analysis Endpoint ----
@app.post("/api/analyze")
async def analyze_endpoint(
    image: UploadFile = File(...),
    kml: UploadFile = File(None),
    confidence: float = Form(0.35),
):
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Save uploaded files to disk (rasterio needs file paths)
        img_path = os.path.join(tmpdir, "input.tif")
        with open(img_path, "wb") as f:
            f.write(await image.read())

        kml_path = None
        if kml and kml.filename:
            kml_path = os.path.join(tmpdir, "boundary.kml")
            with open(kml_path, "wb") as f:
                f.write(await kml.read())

        # 2. Run the ML pipeline (identical logic to the old Streamlit app)
        final_path, gsd, aoi_proj = crop_to_aoi(img_path, kml_path, workdir=tmpdir)
        tree_df = count_trees(final_path, conf_threshold=confidence)
        tree_mask = predict_canopy_mask(final_path)
        metrics = compute_metrics(tree_mask, gsd, aoi_proj, tree_df)
        zip_buffer = create_gis_bundle(final_path, tree_df, tree_mask, metrics)

        # 3. Generate the dual-model overlay image for the UI
        with rasterio.open(final_path) as src:
            rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        rgb_disp = robust_to_uint8(rgb.astype(np.float32))

        fig, ax = plt.subplots(figsize=(10, 10), dpi=150, frameon=False)
        ax.imshow(rgb_disp)
        ax.imshow(tree_mask, cmap="Greens", alpha=0.45)
        for _, row in tree_df.iterrows():
            rect = patches.Rectangle(
                (row["xmin"], row["ymin"]),
                row["xmax"] - row["xmin"],
                row["ymax"] - row["ymin"],
                linewidth=1.5,
                edgecolor="#15803D",
                facecolor="none",
            )
            ax.add_patch(rect)
        ax.axis("off")

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close(fig)

        # 4. Encode to Base64 for the frontend
        img_buf.seek(0)
        img_b64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")
        zip_b64 = base64.b64encode(zip_buffer.getvalue()).decode("utf-8")

        return JSONResponse({
            "metrics": metrics,
            "image_b64": f"data:image/png;base64,{img_b64}",
            "zip_b64": f"data:application/zip;base64,{zip_b64}",
        })
