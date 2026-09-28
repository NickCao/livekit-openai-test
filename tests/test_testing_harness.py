from __future__ import annotations

import asyncio
import struct
from pathlib import Path

from livekit.protocol.agent_pb import agent_session as agent_pb

from livekit_openai_test.testing import AgentHarness

_REPLY_AUDIO = b"\x34\x12" * 960


class _FakeAudioWorker:
    def __init__(self) -> None:
        self.tts_text: str | None = None
        self.transcribed_audio: bytes | None = None
        self.transcribed_sample_rate: int | None = None

    async def tts(self, text: str) -> tuple[bytes, int]:
        self.tts_text = text
        return b"\x01\x00" * 960, 48_000

    async def transcribe(self, pcm_s16le: bytes, sample_rate: int) -> str:
        self.transcribed_audio = pcm_s16le
        self.transcribed_sample_rate = sample_rate
        return "mock agent reply"

    async def aclose(self) -> None:
        pass


class _MockConsoleHarness(AgentHarness):
    def __init__(self) -> None:
        super().__init__(Path(__file__), connect_timeout=2, turn_timeout=5)
        self.audio_worker = _FakeAudioWorker()
        self._audio_worker = self.audio_worker  # type: ignore[assignment]
        self.peer_reader: asyncio.StreamReader | None = None
        self.peer_writer: asyncio.StreamWriter | None = None
        self.peer_task: asyncio.Task[None] | None = None
        self.input_frames: list[agent_pb.AgentSessionMessage.ConsoleIO.AudioFrame] = []
        self.flush_acknowledged = False
        self.peer_error: BaseException | None = None
        self._peer_state = agent_pb.AS_INITIALIZING

    async def _start_agent(self, connect_addr: str) -> None:
        host, port = connect_addr.rsplit(":", maxsplit=1)
        self.peer_reader, self.peer_writer = await asyncio.open_connection(host, int(port))
        self.peer_task = asyncio.create_task(self._serve_protocol_peer())

    async def _serve_protocol_peer(self) -> None:
        try:
            await self._exchange_protocol_peer()
        except BaseException as exc:
            self.peer_error = exc
            if self.peer_writer is not None:
                self.peer_writer.close()
            raise

    async def _exchange_protocol_peer(self) -> None:
        await self._send_state(agent_pb.AS_LISTENING)
        for _ in range(31):  # one 20 ms speech frame plus 600 ms of trailing silence
            message = await asyncio.wait_for(self._read_message(), timeout=2)
            assert message.WhichOneof("message") == "audio_input"
            self.input_frames.append(message.audio_input)

        await self._send_state(agent_pb.AS_THINKING)
        await self._send_state(agent_pb.AS_SPEAKING)
        output = agent_pb.AgentSessionMessage.ConsoleIO.AudioFrame(
            data=_REPLY_AUDIO,
            sample_rate=48_000,
            num_channels=1,
            samples_per_channel=960,
        )
        await self._peer_send_message(agent_pb.AgentSessionMessage(audio_output=output))
        flush = agent_pb.AgentSessionMessage(
            audio_playback_flush=agent_pb.AgentSessionMessage.ConsoleIO.AudioPlaybackFlush()
        )
        await self._peer_send_message(flush)
        acknowledgement = await asyncio.wait_for(self._read_message(), timeout=2)
        self.flush_acknowledged = (
            acknowledgement.WhichOneof("message") == "audio_playback_finished"
        )
        await self._send_state(agent_pb.AS_LISTENING)

    async def _send_state(self, new_state: int) -> None:
        message = agent_pb.AgentSessionMessage()
        change = message.event.agent_state_changed
        change.old_state = self._peer_state
        change.new_state = new_state
        self._peer_state = new_state
        await self._peer_send_message(message)

    async def _peer_send_message(self, message: agent_pb.AgentSessionMessage) -> None:
        assert self.peer_writer is not None
        payload = message.SerializeToString()
        self.peer_writer.write(struct.pack(">I", len(payload)) + payload)
        await self.peer_writer.drain()

    async def aclose(self) -> None:
        if self.peer_task is not None and not self.peer_task.done():
            self.peer_task.cancel()
            await asyncio.gather(self.peer_task, return_exceptions=True)
        if self.peer_writer is not None:
            self.peer_writer.close()
            await self.peer_writer.wait_closed()
            self.peer_writer = None
        await super().aclose()

    async def _read_message(self) -> agent_pb.AgentSessionMessage:
        assert self.peer_reader is not None
        header = await self.peer_reader.readexactly(4)
        size = struct.unpack(">I", header)[0]
        message = agent_pb.AgentSessionMessage()
        message.ParseFromString(await self.peer_reader.readexactly(size))
        return message


async def test_harness_round_trip_with_mock_console_peer() -> None:
    harness = _MockConsoleHarness()
    async with harness as agent:
        try:
            turn = await agent.say("hello agent")
        except Exception:
            if harness.peer_error is not None:
                raise harness.peer_error
            raise
        assert harness.peer_task is not None
        await asyncio.wait_for(harness.peer_task, timeout=2)

    assert harness.audio_worker.tts_text == "hello agent"
    assert len(harness.input_frames) == 31
    assert all(len(frame.data) == 1_920 for frame in harness.input_frames)
    assert all(frame.sample_rate == 48_000 for frame in harness.input_frames)
    assert all(frame.num_channels == 1 for frame in harness.input_frames)
    assert harness.input_frames[0].data == b"\x01\x00" * 960
    assert harness.flush_acknowledged
    assert turn.user_text == "hello agent"
    assert turn.text == "mock agent reply"
    assert turn.audio_pcm_s16le == _REPLY_AUDIO
    assert turn.sample_rate == 48_000
    assert harness.audio_worker.transcribed_audio == _REPLY_AUDIO
    assert harness.audio_worker.transcribed_sample_rate == 48_000
    assert [event.agent_state_changed.new_state for event in turn.events] == [
        agent_pb.AS_THINKING,
        agent_pb.AS_SPEAKING,
        agent_pb.AS_LISTENING,
    ]
