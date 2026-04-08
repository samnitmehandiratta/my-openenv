#!/usr/bin/env python3
"""
Server entry point for Physics Surrogate Environment
Located at server/app.py for multi-mode deployment validation
"""
import uvicorn


def main():
    """Entry point for the server script"""
    from app import app
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
