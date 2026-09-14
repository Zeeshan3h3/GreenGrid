import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

def test():
    print("Loading models...")
    processor = AutoImageProcessor.from_pretrained("restor/tcd-segformer-mit-b2")
    model = SegformerForSemanticSegmentation.from_pretrained("restor/tcd-segformer-mit-b2")
    model.eval()
    
    print("Models loaded.")
    tile_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    
    print("Creating inputs...")
    inputs = processor(images=Image.fromarray(tile_img), return_tensors="pt")
    
    print("Running model...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    print("Model ran!")

if __name__ == "__main__":
    test()
