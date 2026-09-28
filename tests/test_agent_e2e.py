from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("LK_RUN_E2E") != "1",
        reason="set LK_RUN_E2E=1 to run the live agent test",
    ),
]


async def test_agent_keeps_context_across_spoken_turns(agent) -> None:
    first_turn = await agent.say("My favorite color is blue. Reply only OK.")
    assert first_turn.text.strip()
    assert first_turn.audio_pcm_s16le

    second_turn = await agent.say("What is my favorite color? Reply with one word.")
    assert "blue" in second_turn.text.casefold()
    assert second_turn.audio_pcm_s16le
