import torch
import rasterio
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

print("Loading models with torch imported first...")
processor = AutoImageProcessor.from_pretrained("restor/tcd-segformer-mit-b2")
model = SegformerForSemanticSegmentation.from_pretrained("restor/tcd-segformer-mit-b2")
print("SUCCESS!")
