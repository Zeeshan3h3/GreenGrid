import sys
import torch
import numpy as np
from deepforest import main as deepforest_main
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set standard output to handle utf-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')

def test_deepforest():
    print("Testing DeepForest Model...")
    model = deepforest_main.deepforest()
    try:
        # Using the exact same call as in pipeline.py
        model.load_model("weecology/deepforest-tree")
        print("[OK] DeepForest model loaded successfully.")
    except Exception as e:
        print(f"[FAIL] DeepForest loading failed: {e}")
        return False
    return True

def test_segformer():
    print("\nTesting Segformer Model...")
    try:
        processor = AutoImageProcessor.from_pretrained("restor/tcd-segformer-mit-b2")
        model = SegformerForSemanticSegmentation.from_pretrained("restor/tcd-segformer-mit-b2")
        model.eval()
        print("[OK] Segformer weights loaded successfully from Hugging Face.")
        
        # Test a forward pass with dummy data
        dummy_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        inputs = processor(images=dummy_img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            
        if outputs.logits is not None:
            print("[OK] Segformer forward pass successful. Output shape:", outputs.logits.shape)
            return True
        else:
            print("[FAIL] Segformer forward pass returned None.")
            return False
            
    except Exception as e:
        print(f"[FAIL] Segformer test failed: {e}")
        return False

if __name__ == "__main__":
    df_ok = test_deepforest()
    sf_ok = test_segformer()
    
    if df_ok and sf_ok:
        print("\nALL BACKEND ML MODELS ARE WORKING PROPERLY!")
    else:
        print("\nSOME MODELS FAILED TO LOAD OR RUN.")
