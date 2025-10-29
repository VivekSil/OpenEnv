# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for OCR Environment.

This module defines the Action, Observation, and State types for OCR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from core.env_server import Action, Observation, State


@dataclass
class OCRAction(Action):
    all_text: str


@dataclass
class OCRObservation(Observation):
    total_reward: int
    step_count: int

@dataclass
class OCRState(State):
    episode_id: str
    step_count: int
    done: bool = False
