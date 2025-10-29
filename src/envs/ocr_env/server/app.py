# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the OCR Environment.

This module creates an HTTP server that exposes OCR environment
over HTTP endpoints, making them compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.ocr_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.ocr_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.ocr_env.server.app

"""

import os

from core.env_server import create_app

from ..models import OCRAction, OCRObservation
from .ocr_environment import OCREnvironment


# Create the environment instance
env = OCREnvironment(
)

# Create the FastAPI app with web interface and README integration
app = create_app(env, OCRAction, OCRObservation, env_name="ocr_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
