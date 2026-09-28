# livekit-openai-test

A small LiveKit Agents app with a pytest harness for audio-driven end-to-end tests.

## Audio E2E tests

The `agent` fixture starts the configured Python agent in `console` mode and connects to its TCP audio protocol on loopback. It uses no LiveKit server, audio device, or changes to the LiveKit Agents SDK. Each call to `agent.say(text)` synthesizes the text with Hexgrad Kokoro, streams the resulting audio into the running agent, captures its spoken audio, and transcribes the reply with OpenAI Whisper.

The example agent needs `OPENAI_API_KEY` in `.env`. Install the test dependencies and run the example with:

```bash
uv sync --group dev
LK_RUN_E2E=1 uv run pytest -m e2e
```

For your own test, use the fixture directly:

```python
async def test_greeting(agent):
    turn = await agent.say("Hello. What can you help me with?")

    assert "help" in turn.text.lower()
    assert turn.audio_pcm_s16le  # captured mono PCM audio
    assert turn.events           # LiveKit AgentSession events for this turn
```

`turn.text` (also available as `turn.transcript`) is the assistant's audio transcribed by Whisper. `turn.user_text`, `turn.sample_rate`, `turn.audio_pcm_s16le`, and `turn.events` are available for assertions and diagnostics.

By default, the fixture runs `src/livekit_openai_test/agent.py`. Set `LK_TEST_AGENT_ENTRYPOINT` to a different agent file to exercise another agent. `LK_TEST_TURN_TIMEOUT`, `LK_TEST_CONNECT_TIMEOUT`, and `LK_TEST_MODEL_STARTUP_TIMEOUT` tune the harness timeouts.

Kokoro currently requires Python 3.10 through 3.13, while this app uses Python 3.14. The fixture therefore starts a separate Python 3.13 worker via `uv` for Kokoro and Whisper; it installs into uv's cache and leaves the app environment alone. On first use, uv downloads Python/dependencies and the TTS/STT model weights. Kokoro's English phoneme frontend also needs eSpeak NG available on the system (on Debian/Ubuntu: `sudo apt-get install espeak-ng`). Set `LK_TEST_KOKORO_VOICE` or `LK_TEST_WHISPER_MODEL` to select a different Kokoro voice or Whisper model.
