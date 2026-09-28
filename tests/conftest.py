from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

from livekit_openai_test.testing import AgentHarness


@pytest_asyncio.fixture
async def agent() -> AsyncIterator[AgentHarness]:
    """Run the configured agent in console mode for one test."""
    entrypoint_value = os.environ.get("LK_TEST_AGENT_ENTRYPOINT")
    entrypoint = Path(entrypoint_value) if entrypoint_value else None
    async with AgentHarness(entrypoint=entrypoint) as harness:
        yield harness
