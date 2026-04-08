#!/usr/bin/env python3
"""
Server entry point for Physics Surrogate Environment
Located at server/app.py for multi-mode deployment validation
"""
import sys
import os

# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
