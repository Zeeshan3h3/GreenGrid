import os
import uvicorn
import gradio as gr
from fastapi import FastAPI
from server import app as fastapi_app

# 1. Create a dummy Gradio interface
# Hugging Face Gradio spaces look for a 'demo' object or run 'python app.py'
demo = gr.Blocks()
with demo:
    gr.Markdown("# CanopyAI Backend is running.")
    gr.Markdown("The main frontend is served at the root URL (/) via FastAPI.")

# 2. Mount Gradio inside our existing FastAPI app
# This satisfies the HF Space requirements, while keeping our custom HTML UI at the root `/`
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

# 3. Start the server on port 7860 (Default for Hugging Face)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
