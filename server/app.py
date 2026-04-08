#!/usr/bin/env python3
"""
Server entry point for Physics Surrogate Environment
Located at server/app.py for multi-mode deployment validation
"""
import sys
import os

# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

# Import the FastAPI app
from app import app

def main():
    """Entry point for the server script"""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
