import spaces
import uvicorn
import gradio as gr
from fastapi import FastAPI
from server import app as fastapi_app

# ZeroGPU requires at least one function decorated with @spaces.GPU to pass startup checks
@spaces.GPU
def dummy_gpu_function():
    pass

# 1. Create a dummy Gradio interface
demo = gr.Blocks()
with demo:
    gr.Markdown("# CanopyAI Backend is running.")
    gr.Markdown("The main frontend is served at the root URL (/) via FastAPI.")

# 2. Mount Gradio inside our existing FastAPI app
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

# Hugging Face Spaces automatically runs the `app` object on port 7860.
# We do not manually call uvicorn.run() here to avoid 'Address already in use' errors.
