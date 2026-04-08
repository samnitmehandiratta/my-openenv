#!/usr/bin/env python3
"""
Physics Surrogate Environment - Entry point for Hugging Face Space
Launches FastAPI server with Gradio UI mounted
"""
import uvicorn
from app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
else:
    # When imported, just expose the app
    pass
