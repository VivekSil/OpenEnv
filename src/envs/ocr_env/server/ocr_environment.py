# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OCR Environment Server Implementation.

This module wraps OCR's rl_environment.Environment and exposes it
via the OpenEnv Environment interface.
"""

from typing import Any, Dict, Tuple

from core.env_server import Action, Environment, Observation

from ..models import OCRAction, OCRObservation, OCRState

try:
    from datasets import load_dataset
    from Levenshtein import ratio as levenshtein_ratio
except:
    print("Install datasets")

def compute_anls(pred: str, gt: str) -> float:
    """
    Computes Average Normalized Levenshtein Similarity (ANLS).
    ANLS = max(0, 1 - edit_distance / max(len(pred), len(gt))) if edit_distance < 0.5 * len(gt)
    """
    if not gt:
        return 0.0
    d = levenshtein_ratio(pred.lower(), gt.lower())
    return d  # ratio() already normalized between 0 and 1


class OCREnvironment:
    def __init__(self):
        self.state = OCRState(episode_id=0, step_count=0, done=False)
        self.total_reward = 0

    def reset(self) -> OCRObservation:
        self.state.episode_id += 1
        self.state.step = 0
        self.state.done = False
        self.total_reward = 0
        return OCRObservation(total_reward=self.total_reward, step_count=self.state.step)

    def step(self, action: OCRAction) -> Tuple[OCRObservation, float, bool, OCRState]:
        self.state.step += 1
        pred = action.all_text.split("<<split>>")[0].strip()
        gt = action.all_text.split("<<split>>")[1].strip()

        # Compute reward metrics
        anls = compute_anls(pred, gt)
        lev = levenshtein_ratio(pred, gt)
        self.total_reward = 0.7 * anls + 0.3 * lev

        # Note: This is for single step environment
        self.state.done = True
        multistep = False
        # For multiple steps, this can looped wth threshold for reward
        if multistep:
            if self.state.step > 3 or self.total_reward > 0.8:
                self.state.done = True

        obs = OCRObservation(
            total_reward=self.total_reward,
            step_count=self.state.step
        )
        return obs