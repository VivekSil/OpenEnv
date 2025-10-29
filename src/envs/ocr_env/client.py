# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OCREnv HTTP Client.

This module provides the client for connecting to an OpenSpiel Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from core.client_types import StepResult

from core.http_env_client import HTTPEnvClient

from .models import OCRAction, OCRObservation, OCRState

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider


class OCREnv(HTTPEnvClient[OCRAction, OCRObservation]):
    """
    HTTP client for OpenSpiel Environment.

    This client connects to an OCREnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    """

    def _step_payload(self, action: OCRAction) -> Dict[str, Any]:
        """
        Convert OCRAction to JSON payload for step request.

        Args:
            action: OCRAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "all_text": action.all_text,
        }

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[OCRObservation]:
        """
        Parse server response into StepResult[OCRObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with OCRObservation.
        """
        obs_data = payload.get("observation", {})

        observation = OCRObservation(
            step_count=obs_data.get("step_count"),
            done=obs_data.get("done", False),
            total_reward=obs_data.get("total_reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("total_reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> OCRState:
        """
        Parse server response into OCRState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            OCRState object with environment state information.
        """
        return OCRState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            done=payload.get("done", False),
        )
