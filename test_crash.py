import traceback
import sys
try:
    from pipeline import run_pipeline
    import os
    img_path = r"c:\Users\mdzee\Downloads\tree_hackathon\project\test_file\data\RGB\DSNY_025_2018.tif"
    print("Running pipeline...")
    metrics, final_path, tree_mask, tree_df, zip_data = run_pipeline(img_path, kml_path=None, conf_threshold=0.35, workdir=".")
    print("Success!")
except Exception as e:
    print("Error occurred:")
    traceback.print_exc()
    sys.exit(1)
