---
title: CanopyAI
emoji: 🌲
colorFrom: green
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# CanopyAI: Dual-Model Forest Crown & Area Platform

CanopyAI is an automated, high-performance platform for forest inventory, crown detection, and canopy area estimation. Designed for speed, memory efficiency, and a premium user experience, the platform processes drone orthomosaics to deliver actionable ecological metrics and exportable GIS deliverables.

## 🚀 Architecture

The project has been architected for production-readiness, decoupling the heavy machine learning inference from the user interface:

- **Backend (FastAPI)**: A lightweight, asynchronous Python REST API (`server.py`). It loads the heavy PyTorch models into memory exactly *once* at startup, preventing Out-Of-Memory (OOM) crashes that plague traditional data-science UI frameworks.
- **Frontend (HTML5 / Tailwind CSS)**: A static, single-page application (`index.html`) using Vanilla JavaScript. Styled with a custom "Plant Care Tracker" glassmorphism theme via Tailwind CSS and Lucide icons.
- **ML Pipeline (`pipeline.py`)**: The core AI logic. It handles geographic reprojection, tensor conversions, tile-based inference, and metric calculations.

## 🧠 Machine Learning Models

CanopyAI utilizes a "Dual-Model Engine" to extract the most accurate data from aerial imagery:

1. **DeepForest (RetinaNet)**
   - **Purpose**: Individual tree stem counting and bounding box detection.
   - **How it works**: Uses a PyTorch implementation of RetinaNet (object detection) trained on the National Ecological Observatory Network (NEON) dataset. It scans the image in overlapping patches (e.g., 400x400) to find distinct tree crowns.
2. **Restor OAM-TCD (Segformer-MIT-B2)**
   - **Purpose**: Pixel-level canopy coverage estimation.
   - **How it works**: A Hugging Face Vision Transformer (ViT) trained on Open Aerial Map (OAM) data for Tree Cover Density (TCD). It performs semantic segmentation, classifying every single pixel as either "canopy" or "non-canopy" to calculate the exact surface area of the forest.

## 📁 Input Data Requirements

To get the best results from the pipeline, your input data should meet these criteria:

- **File Format**: GeoTIFF (`.tif` or `.tiff`). The image *must* have spatial metadata (CRS and Transform) embedded.
- **Bands**: Standard RGB (3-band).
- **Resolution**: High-resolution drone imagery (Ground Sampling Distance of 5cm to 50cm per pixel) yields the best crown detection. Satellite imagery (e.g., 10m/px) will not work well for individual tree counting.
- **File Size**: 
  - The pipeline uses *tiled inference* (processing the image in small chunks), meaning it can theoretically handle infinitely large files without crashing the RAM.
  - However, for a snappy UI experience (especially on CPU), it is recommended to use cropped demonstration images **between 10MB to 50MB**.
- **Area of Interest (Optional)**: You can upload a `.kml` polygon to clip the analysis to a specific boundary. If omitted, the entire GeoTIFF bounding box is analyzed.

## 💻 How to Use the Platform

### 1. Start the Server
Open your terminal in the project directory and run the FastAPI server using Uvicorn:
```bash
uvicorn server:app --port 8000
```

### 2. Access the UI
Open your web browser and navigate to:
```text
http://localhost:8000
```

### 3. Run an Analysis
1. **Upload Datasets**: In the left panel, click the **Orthomosaic** input and select your `.tif` file (e.g., `test_file/data/RGB/DSNY_025_2018.tif`).
2. **Adjust Confidence**: The default detection confidence is `0.35`. 
   - *Increase it* if the model is falsely detecting bushes/ground as trees.
   - *Decrease it* if the model is missing small saplings.
3. **Execute**: Click **Execute Analysis**. The frontend will display a loading spinner while the FastAPI backend chunks the image and runs both neural networks.

### 4. Interpret the Results
Once complete, the dashboard will display:
- **KPI Metrics**: Total stem count, exact canopy area (m² and hectares), percentage coverage, stand density (trees/ha), and estimated CO₂ sequestration capacity.
- **Dual-Model Overlay**: An interactive visual inspection map. 
  - 🟩 **Green Mask**: Pixels classified as canopy by the Segformer model.
  - 🟨 **Yellow Boxes**: Individual tree crowns detected by the DeepForest RetinaNet model.
- **GIS Export**: Click **Download .ZIP** to get a bundled package containing `tree_inventory.csv`, GeoJSON polygons for mapping software (QGIS/ArcGIS), the summary metrics, and the high-res PNG overlay.
