"""
Tree crown detection + canopy area pipeline.

Backend module — kept separate from app.py on purpose. The Streamlit UI only
ever calls run_pipeline(); swapping this file's internals later (e.g. to call
the official `tcd-predict` CLI instead of the manual transformers loading
below) requires zero changes to app.py.
"""

import io
import os
import shutil
import zipfile

import torch
import geopandas as gpd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from deepforest import main as deepforest_main
from PIL import Image
from rasterio.features import shapes as rio_shapes
from rasterio.mask import mask as rio_mask
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window
from shapely.geometry import box as shapely_box
from shapely.geometry import shape
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


def robust_to_uint8(img_float, nodata_value=0):
    """Percentile-based, nodata-aware rescale to 8-bit.

    Fixes the washed-out-image bug from today: a plain min/max stretch gets
    dragged toward 0 by nodata border pixels from reprojected/rotated
    orthomosaics, crushing real canopy texture into a narrow band.
    """
    valid = img_float[img_float > nodata_value]
    if valid.size == 0:
        return img_float.astype(np.uint8)
    p2, p98 = np.percentile(valid, [2, 98])
    stretched = np.clip((img_float - p2) / (p98 - p2 + 1e-8) * 255, 0, 255)
    return stretched.astype(np.uint8)


def crop_to_aoi(raster_path, kml_path=None, target_res=0.1, workdir="."):
    """Step 1: clip the raster to the KML polygon (or use full raster bounds
    if no KML is provided), force 3-band RGB, and resample to target_res
    METERS/pixel — always in a projected CRS.

    Fixes today's geographic-CRS bug: if the source raster is in lat/lon
    (EPSG:4326), resampling to "0.1" would mean 0.1 DEGREES (~11km), not
    0.1 meters. estimate_utm_crs() picks the correct local UTM zone so the
    resolution always means what it says.
    """
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        src_bounds = src.bounds

    if kml_path and os.path.exists(kml_path):
        aoi = gpd.read_file(kml_path)
        target_crs = aoi.estimate_utm_crs() if raster_crs.is_geographic else raster_crs
        aoi_native = aoi.to_crs(raster_crs)

        with rasterio.open(raster_path) as src:
            out_image, out_transform = rio_mask(src, aoi_native.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update(
                height=out_image.shape[1], width=out_image.shape[2], transform=out_transform
            )
    else:
        # No KML: use full raster bounding box as AOI
        whole_box = shapely_box(src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top)
        aoi = gpd.GeoDataFrame({"id": [1], "geometry": [whole_box]}, crs=raster_crs)
        target_crs = aoi.estimate_utm_crs() if raster_crs.is_geographic else raster_crs

        with rasterio.open(raster_path) as src:
            out_image = src.read()
            out_meta = src.meta.copy()

    cropped_path = os.path.join(workdir, "cropped.tif")
    with rasterio.open(cropped_path, "w", **out_meta) as dst:
        dst.write(out_image)

    # Ensure strictly 3-band RGB
    rgb_path = os.path.join(workdir, "cropped_rgb.tif")
    with rasterio.open(cropped_path) as src:
        if src.count > 3:
            rgb, meta = src.read([1, 2, 3]), src.meta.copy()
            meta.update(count=3)
            with rasterio.open(rgb_path, "w", **meta) as dst:
                dst.write(rgb)
        else:
            shutil.copy(cropped_path, rgb_path)

    # Resample into projected CRS at target_res METERS/pixel
    final_path = os.path.join(workdir, "cropped_final.tif")
    with rasterio.open(rgb_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds, resolution=target_res
        )
        kwargs = src.meta.copy()
        kwargs.update(crs=target_crs, transform=transform, width=width, height=height)
        with rasterio.open(final_path, "w", **kwargs) as dst:
            for i in range(1, 4):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )

    with rasterio.open(final_path) as src:
        final_gsd = src.res[0]

    return final_path, final_gsd, aoi.to_crs(target_crs)


# ---- Module-level model singletons (loaded once on first call) ----
_deepforest_model = None
_segformer_processor = None
_segformer_model = None


def get_deepforest_model():
    global _deepforest_model
    if _deepforest_model is None:
        model = deepforest_main.deepforest()
        model.load_model("weecology/deepforest-tree")
        _deepforest_model = model
    return _deepforest_model


def get_segformer_model():
    global _segformer_processor, _segformer_model
    if _segformer_processor is None:
        processor = AutoImageProcessor.from_pretrained("restor/tcd-segformer-mit-b2")
        model = SegformerForSemanticSegmentation.from_pretrained("restor/tcd-segformer-mit-b2")
        model.eval()
        _segformer_processor, _segformer_model = processor, model
    return _segformer_processor, _segformer_model


def count_trees(final_path, conf_threshold=0.3, patch_size=400, patch_overlap=0.25):
    """Step 2a: tree count via DeepForest. Returns filtered detections DataFrame.

    The conf_threshold parameter lets the UI dynamically filter
    low-confidence detections (e.g. ground brush false positives).
    """
    model = get_deepforest_model()
    results = model.predict_tile(path=final_path, patch_size=patch_size, patch_overlap=patch_overlap)
    if results is None or len(results) == 0:
        return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])
    filtered = results[results["score"] >= conf_threshold].reset_index(drop=True)
    return filtered


def predict_canopy_mask(final_path, tile_size=512, overlap=64, tree_class_id=1):
    """Step 2b: canopy area via Restor's OAM-TCD-trained segformer model,
    applied tile-by-tile so this works on large orthomosaics without
    exhausting memory."""
    processor, model = get_segformer_model()

    with rasterio.open(final_path) as src:
        width, height = src.width, src.height
        needs_rescale = src.dtypes[0] != "uint8"
        full_mask = np.zeros((height, width), dtype=np.uint8)
        step = tile_size - overlap

        for top in range(0, height, step):
            for left in range(0, width, step):
                win_w, win_h = min(tile_size, width - left), min(tile_size, height - top)
                window = Window(left, top, win_w, win_h)
                tile = src.read([1, 2, 3], window=window)
                tile_img = np.transpose(tile, (1, 2, 0)).astype(np.float32)
                tile_img = robust_to_uint8(tile_img) if needs_rescale else tile_img.astype(np.uint8)

                pad_h, pad_w = tile_size - tile_img.shape[0], tile_size - tile_img.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile_img = np.pad(tile_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

                inputs = processor(images=Image.fromarray(tile_img), return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = torch.nn.functional.interpolate(
                    outputs.logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False
                )
                tile_mask = logits.argmax(dim=1)[0].cpu().numpy()[:win_h, :win_w]
                full_mask[top : top + win_h, left : left + win_w] = tile_mask

    return full_mask == tree_class_id


def mask_to_polygons(tree_mask, final_path):
    """Turn the binary canopy raster mask into real georeferenced polygons —
    this is the actual 'exportable GIS data' deliverable, not just a picture."""
    with rasterio.open(final_path) as src:
        transform, crs = src.transform, src.crs
    shapes_gen = rio_shapes(tree_mask.astype(np.uint8), mask=tree_mask, transform=transform)
    geoms = [shape(geom) for geom, _ in shapes_gen]
    return gpd.GeoDataFrame({"geometry": geoms}, crs=crs)


def detections_to_geodataframe(detections_df, final_path):
    """Convert DeepForest's pixel-space boxes into georeferenced boxes,
    using the raster's own affine transform — same idea as the polygons
    above, applied to individual tree detections."""
    if detections_df is None or len(detections_df) == 0:
        return gpd.GeoDataFrame({"geometry": []})
    with rasterio.open(final_path) as src:
        transform, crs = src.transform, src.crs
    geoms, scores = [], []
    for _, row in detections_df.iterrows():
        left, top = transform * (row["xmin"], row["ymin"])
        right, bottom = transform * (row["xmax"], row["ymax"])
        geoms.append(
            shapely_box(min(left, right), min(top, bottom), max(left, right), max(top, bottom))
        )
        scores.append(row.get("score", None))
    return gpd.GeoDataFrame({"score": scores}, geometry=geoms, crs=crs)


def compute_metrics(tree_mask, gsd, aoi_proj, tree_df):
    """Step 3: pixels -> real-world numbers, including ecological metrics."""
    tree_count = len(tree_df)
    canopy_area_m2 = float(tree_mask.sum()) * (gsd**2)
    canopy_area_ha = canopy_area_m2 / 10000.0
    aoi_area_m2 = float(aoi_proj.geometry.area.sum())
    canopy_pct = (canopy_area_m2 / aoi_area_m2 * 100.0) if aoi_area_m2 > 0 else 0.0

    # Climate metric: ~0.05 metric tons CO2 / m^2 canopy proxy
    carbon_seq_tons = canopy_area_m2 * 0.05

    # Stand density index (trees per hectare of canopy)
    stand_density = (tree_count / canopy_area_ha) if canopy_area_ha > 0 else 0.0

    if stand_density < 250:
        health_status = "Sparse / Open Canopy"
    elif stand_density <= 800:
        health_status = "Balanced / Optimal Density"
    else:
        health_status = "Dense / High Competition"

    return {
        "tree_count": tree_count,
        "canopy_area_m2": canopy_area_m2,
        "canopy_area_ha": canopy_area_ha,
        "canopy_pct": canopy_pct,
        "carbon_seq_tons": carbon_seq_tons,
        "stand_density": stand_density,
        "health_status": health_status,
    }


def create_gis_bundle(final_path, tree_df, tree_mask, metrics):
    """Bundles CSV, GeoJSON (trees + canopy polygons), visual overlay, and
    summary text into a downloadable in-memory ZIP archive."""
    # --- Tree detections GeoJSON ---
    trees_gdf = detections_to_geodataframe(tree_df, final_path)
    if not trees_gdf.empty:
        trees_gdf_wgs84 = trees_gdf.to_crs(epsg=4326)
    else:
        trees_gdf_wgs84 = trees_gdf

    # --- Canopy polygons GeoJSON (from Segformer mask) ---
    canopy_gdf = mask_to_polygons(tree_mask, final_path)
    if not canopy_gdf.empty:
        canopy_gdf_wgs84 = canopy_gdf.to_crs(epsg=4326)
    else:
        canopy_gdf_wgs84 = canopy_gdf

    # --- Visual overlay (matplotlib) ---
    with rasterio.open(final_path) as src:
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
    rgb_disp = robust_to_uint8(rgb.astype(np.float32))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    ax.imshow(rgb_disp)
    ax.imshow(tree_mask, cmap="Greens", alpha=0.45)
    for _, row in tree_df.iterrows():
        rect = patches.Rectangle(
            (row["xmin"], row["ymin"]),
            row["xmax"] - row["xmin"],
            row["ymax"] - row["ymin"],
            linewidth=1.2, edgecolor="#FFFF00", facecolor="none",
        )
        ax.add_patch(rect)
    ax.axis("off")

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    img_buf.seek(0)

    # --- Package into ZIP ---
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Tree inventory CSV
        if not trees_gdf_wgs84.empty:
            csv_data = trees_gdf_wgs84.drop(columns="geometry").to_csv(index=False)
            zf.writestr("tree_inventory.csv", csv_data)

        # Tree crowns GeoJSON (DeepForest boxes)
        zf.writestr("tree_detections.geojson", trees_gdf_wgs84.to_json())

        # Canopy polygons GeoJSON (Segformer mask)
        zf.writestr("canopy_polygons.geojson", canopy_gdf_wgs84.to_json())

        # High-res overlay PNG
        zf.writestr("canopy_overlay.png", img_buf.getvalue())

        # Summary text
        summary = (
            f"Tree Count: {metrics['tree_count']}\n"
            f"Canopy Area (ha): {metrics['canopy_area_ha']:.2f}\n"
            f"Canopy Cover: {metrics['canopy_pct']:.1f}%\n"
            f"CO2 Sequestration: {metrics['carbon_seq_tons']:.1f} t\n"
            f"Stand Density: {metrics['stand_density']:.1f} trees/ha "
            f"({metrics['health_status']})\n"
        )
        zf.writestr("metrics_summary.txt", summary)

    zip_buffer.seek(0)
    return zip_buffer


def run_pipeline(raster_path, kml_path=None, conf_threshold=0.3, workdir="."):
    """Top-level entry point — the ONLY function app.py calls. Swap the
    internals of predict_canopy_mask (or this whole function, to shell out
    to `tcd-predict` instead) later without touching the UI at all."""
    final_path, gsd, aoi_proj = crop_to_aoi(raster_path, kml_path, workdir=workdir)
    tree_df = count_trees(final_path, conf_threshold=conf_threshold)
    tree_mask = predict_canopy_mask(final_path)
    metrics = compute_metrics(tree_mask, gsd, aoi_proj, tree_df)
    zip_data = create_gis_bundle(final_path, tree_df, tree_mask, metrics)
    return metrics, final_path, tree_mask, tree_df, zip_data
