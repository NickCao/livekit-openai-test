from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import socket
import struct
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from livekit import rtc
from livekit.protocol.agent_pb import agent_session as agent_pb

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ENTRYPOINT = _PROJECT_ROOT / "src" / "livekit_openai_test" / "agent.py"
_AUDIO_WORKER = _PROJECT_ROOT / "tests" / "audio_worker.py"
_TCP_MAX_MESSAGE_SIZE = 1 << 20
_WIRE_SAMPLE_RATE = 48_000
_WIRE_CHANNELS = 1
_FRAME_SAMPLES = 960  # 20 ms at 48 kHz
_FRAME_BYTES = _FRAME_SAMPLES * 2  # mono signed 16-bit PCM
_TRAILING_SILENCE_SECONDS = 0.6
_AUDIO_WORKER_REQUIREMENTS = (
    "kokoro>=0.9.4,<0.10",
    "openai-whisper>=20240930",
    "scipy>=1.11,<2",
)
_AUDIO_WORKER_MAX_LINE_SIZE = 64 << 20


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """One audio-driven turn and the assistant audio transcribed by Whisper."""

    user_text: str
    text: str
    audio_pcm_s16le: bytes
    sample_rate: int
    events: tuple[agent_pb.AgentSessionEvent, ...]

    @property
    def transcript(self) -> str:
        """Alias for ``text`` that makes the Whisper source explicit."""
        return self.text


@dataclass(slots=True)
class _TurnCapture:
    audio: bytearray = field(default_factory=bytearray)
    events: list[agent_pb.AgentSessionEvent] = field(default_factory=list)
    sample_rate: int | None = None


class _AudioModelWorker:
    """Run Kokoro and Whisper in a Python version supported by both packages."""

    def __init__(self, *, startup_timeout: float) -> None:
        self._startup_timeout = startup_timeout
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._audio_python = os.environ.get("LK_TEST_AUDIO_PYTHON", "3.13")
        self._kokoro_voice = os.environ.get("LK_TEST_KOKORO_VOICE", "af_heart")
        self._whisper_model = os.environ.get("LK_TEST_WHISPER_MODEL", "base.en")

    async def tts(self, text: str) -> tuple[bytes, int]:
        response = await self._request({"action": "tts", "text": text})
        return base64.b64decode(response["audio"], validate=True), int(response["sample_rate"])

    async def transcribe(self, pcm_s16le: bytes, sample_rate: int) -> str:
        response = await self._request(
            {
                "action": "transcribe",
                "audio": base64.b64encode(pcm_s16le).decode("ascii"),
                "sample_rate": sample_rate,
                "channels": _WIRE_CHANNELS,
            }
        )
        return str(response["text"]).strip()

    async def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            await self._start()
            assert self._process is not None
            if self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("audio model worker was started without stdio pipes")

            self._process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            await self._process.stdin.drain()
            try:
                line = await asyncio.wait_for(
                    self._process.stdout.readline(), timeout=self._startup_timeout
                )
            except TimeoutError as exc:
                raise TimeoutError("Kokoro/Whisper audio worker did not respond in time") from exc

            if not line:
                return_code = await self._process.wait()
                raise RuntimeError(f"Kokoro/Whisper audio worker exited with code {return_code}")

            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(f"Kokoro/Whisper audio worker failed: {response['error']}")
            return response

    async def _start(self) -> None:
        if self._process is not None:
            if self._process.returncode is None:
                return
            raise RuntimeError(
                f"Kokoro/Whisper audio worker exited with code {self._process.returncode}"
            )

        uv = os.environ.get("UV") or shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required to create the Python 3.13 Kokoro/Whisper worker")

        command = [
            uv,
            "run",
            "--quiet",
            "--no-project",
            "--python",
            self._audio_python,
        ]
        for requirement in _AUDIO_WORKER_REQUIREMENTS:
            command.extend(("--with", requirement))
        command.append(str(_AUDIO_WORKER))

        worker_env = os.environ.copy()
        worker_env["LK_TEST_KOKORO_VOICE"] = self._kokoro_voice
        worker_env["LK_TEST_WHISPER_MODEL"] = self._whisper_model
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                cwd=_PROJECT_ROOT,
                env=worker_env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                # Base64 audio replies exceed asyncio's default 64 KiB readline limit.
                stderr=None,
                limit=_AUDIO_WORKER_MAX_LINE_SIZE,
            )
            assert self._process.stdout is not None
            line = await asyncio.wait_for(
                self._process.stdout.readline(), timeout=self._startup_timeout
            )
        except TimeoutError as exc:
            await self.aclose()
            raise TimeoutError(
                "Kokoro/Whisper setup timed out while uv installed dependencies or loaded models"
            ) from exc
        except BaseException:
            await self.aclose()
            raise

        if not line:
            return_code = await self._process.wait()
            await self.aclose()
            raise RuntimeError(
                "Kokoro/Whisper worker exited before loading models "
                f"(code {return_code}; see uv/model output above)"
            )

        ready = json.loads(line)
        if not ready.get("ready"):
            error = ready.get("error", "worker failed to initialize")
            await self.aclose()
            raise RuntimeError(f"Kokoro/Whisper audio worker failed to start: {error}")

    async def aclose(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return

        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()


class AgentHarness:
    """Start an agent in console mode and drive it through its TCP audio protocol."""

    def __init__(
        self,
        entrypoint: str | Path | None = None,
        *,
        connect_timeout: float | None = None,
        turn_timeout: float | None = None,
        model_startup_timeout: float | None = None,
    ) -> None:
        configured_entrypoint = entrypoint or os.environ.get("LK_TEST_AGENT_ENTRYPOINT")
        path = Path(configured_entrypoint) if configured_entrypoint else _DEFAULT_ENTRYPOINT
        self.entrypoint = path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()
        self.connect_timeout = connect_timeout or float(
            os.environ.get("LK_TEST_CONNECT_TIMEOUT", "120")
        )
        self.turn_timeout = turn_timeout or float(os.environ.get("LK_TEST_TURN_TIMEOUT", "90"))
        self.model_startup_timeout = model_startup_timeout or float(
            os.environ.get("LK_TEST_MODEL_STARTUP_TIMEOUT", "900")
        )

        self.events: list[agent_pb.AgentSessionEvent] = []
        self._agent_state: int | None = None
        self._state_history: list[int] = []
        self._state_condition = asyncio.Condition()
        self._say_lock = asyncio.Lock()
        self._connected = asyncio.Event()
        self._reader_error: BaseException | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task[None] | None = None
        self._server: asyncio.AbstractServer | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._log_task: asyncio.Task[None] | None = None
        self._agent_logs: deque[str] = deque(maxlen=80)
        self._active_capture: _TurnCapture | None = None
        self._audio_worker = _AudioModelWorker(startup_timeout=self.model_startup_timeout)
        self._closed = False

    async def __aenter__(self) -> AgentHarness:
        if not self.entrypoint.is_file():
            raise FileNotFoundError(f"agent entrypoint does not exist: {self.entrypoint}")

        self._server = await asyncio.start_server(self._accept_connection, "127.0.0.1", 0)
        socket_info = self._server.sockets[0].getsockname()
        connect_addr = f"127.0.0.1:{socket_info[1]}"
        try:
            await self._start_agent(connect_addr)
            await self._wait_for_connection()
            await self._wait_for_state(
                agent_pb.AS_LISTENING,
                timeout=self.connect_timeout,
                description="the agent to enter listening state",
            )
            # Let TcpAudioInput start its consumer loop before the first frame arrives.
            await asyncio.sleep(0.25)
        except BaseException:
            await self.aclose()
            raise
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def say(self, text: str) -> AgentTurn:
        """Speak ``text`` to the agent and return its spoken reply transcribed by Whisper."""
        if not text.strip():
            raise ValueError("say() requires non-empty text")

        async with self._say_lock:
            await self._wait_for_state(
                agent_pb.AS_LISTENING,
                timeout=self.turn_timeout,
                description="the agent to be ready for the next turn",
            )
            pcm_24k, tts_sample_rate = await self._audio_worker.tts(text)
            wire_pcm = self._resample_to_wire(pcm_24k, tts_sample_rate)
            state_start = len(self._state_history)
            capture = _TurnCapture()
            self._active_capture = capture
            try:
                await self._send_user_audio(wire_pcm)
                await self._wait_for_reply(state_start)
                audio = bytes(capture.audio)
                if not audio:
                    raise RuntimeError(
                        "the agent finished speaking but sent no audio over the console TCP link"
                    )
                sample_rate = capture.sample_rate or _WIRE_SAMPLE_RATE
                transcript = await self._audio_worker.transcribe(audio, sample_rate)
                return AgentTurn(
                    user_text=text,
                    text=transcript,
                    audio_pcm_s16le=audio,
                    sample_rate=sample_rate,
                    events=tuple(capture.events),
                )
            finally:
                self._active_capture = None

    def _accept_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._writer is not None:
            writer.close()
            return
        self._reader = reader
        self._writer = writer
        sock = writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._read_task = asyncio.create_task(self._read_messages())
        self._connected.set()

    async def _start_agent(self, connect_addr: str) -> None:
        """Start the console process; subclasses can substitute an in-process protocol peer."""
        process_env = os.environ.copy()
        process_env["PYTHONUNBUFFERED"] = "1"
        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "livekit.agents",
            "console",
            str(self.entrypoint),
            "--connect-addr",
            connect_addr,
            cwd=_PROJECT_ROOT,
            env=process_env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._log_task = asyncio.create_task(self._drain_agent_logs())

    async def _wait_for_connection(self) -> None:
        if self._process is None:
            try:
                await asyncio.wait_for(self._connected.wait(), timeout=self.connect_timeout)
            except TimeoutError as exc:
                raise TimeoutError(
                    f"protocol peer did not connect within {self.connect_timeout:g}s"
                ) from exc
            return

        connected_task = asyncio.create_task(self._connected.wait())
        exited_task = asyncio.create_task(self._process.wait())
        done, pending = await asyncio.wait(
            (connected_task, exited_task),
            timeout=self.connect_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        if self._connected.is_set():
            return
        if exited_task in done:
            raise RuntimeError(
                "agent exited before connecting to the console TCP listener\n" + self._log_tail()
            )
        raise TimeoutError(
            f"agent did not connect to console TCP listener within {self.connect_timeout:g}s\n"
            + self._log_tail()
        )

    async def _wait_for_state(self, state: int, *, timeout: float, description: str) -> None:
        async with self._state_condition:
            try:
                await asyncio.wait_for(
                    self._state_condition.wait_for(
                        lambda: self._agent_state == state or self._reader_error is not None
                    ),
                    timeout=timeout,
                )
            except TimeoutError as exc:
                raise TimeoutError(
                    f"timed out waiting for {description}; last agent state={self._agent_state}\n"
                    + self._log_tail()
                ) from exc
        self._raise_if_disconnected()

    async def _wait_for_reply(self, state_start: int) -> None:
        def completed() -> bool:
            states = self._state_history[state_start:]
            try:
                speaking_index = states.index(agent_pb.AS_SPEAKING)
            except ValueError:
                return False
            return agent_pb.AS_LISTENING in states[speaking_index + 1 :]

        async with self._state_condition:
            try:
                await asyncio.wait_for(
                    self._state_condition.wait_for(
                        lambda: completed() or self._reader_error is not None
                    ),
                    timeout=self.turn_timeout,
                )
            except TimeoutError as exc:
                states = self._state_history[state_start:]
                raise TimeoutError(
                    f"agent did not finish a speaking turn within {self.turn_timeout:g}s; "
                    f"states={states}\n{self._log_tail()}"
                ) from exc
        self._raise_if_disconnected()

    async def _read_messages(self) -> None:
        assert self._reader is not None
        try:
            while True:
                header = await self._reader.readexactly(4)
                size = struct.unpack(">I", header)[0]
                if size > _TCP_MAX_MESSAGE_SIZE:
                    raise RuntimeError(f"console TCP message is too large: {size} bytes")
                payload = await self._reader.readexactly(size)
                message = agent_pb.AgentSessionMessage()
                message.ParseFromString(payload)
                await self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except (asyncio.IncompleteReadError, ConnectionError, OSError) as exc:
            self._reader_error = ConnectionError("agent closed the console TCP connection")
            self._reader_error.__cause__ = exc
        except Exception as exc:
            self._reader_error = exc
        finally:
            async with self._state_condition:
                self._state_condition.notify_all()

    async def _handle_message(self, message: agent_pb.AgentSessionMessage) -> None:
        kind = message.WhichOneof("message")
        if kind == "audio_output":
            if self._active_capture is not None:
                self._active_capture.audio.extend(message.audio_output.data)
                self._active_capture.sample_rate = message.audio_output.sample_rate
            return

        if kind == "audio_playback_flush":
            # This harness captures instead of playing audio, so acknowledge immediately.
            finished = agent_pb.AgentSessionMessage.ConsoleIO.AudioPlaybackFinished()
            await self._send_message(
                agent_pb.AgentSessionMessage(audio_playback_finished=finished)
            )
            return

        if kind != "event":
            return

        event = agent_pb.AgentSessionEvent()
        event.CopyFrom(message.event)
        self.events.append(event)
        if self._active_capture is not None:
            self._active_capture.events.append(event)
        if event.HasField("agent_state_changed"):
            self._agent_state = event.agent_state_changed.new_state
            self._state_history.append(self._agent_state)
            async with self._state_condition:
                self._state_condition.notify_all()

    async def _send_user_audio(self, pcm_s16le_48k: bytes) -> None:
        if self._writer is None or self._writer.is_closing():
            raise RuntimeError("agent is not connected to the console TCP listener")

        silence = bytes(
            int(_WIRE_SAMPLE_RATE * _TRAILING_SILENCE_SECONDS) * 2 * _WIRE_CHANNELS
        )
        audio = pcm_s16le_48k + silence
        next_frame_at = asyncio.get_running_loop().time()
        frame_duration = _FRAME_SAMPLES / _WIRE_SAMPLE_RATE
        for offset in range(0, len(audio), _FRAME_BYTES):
            frame = audio[offset : offset + _FRAME_BYTES]
            if len(frame) < _FRAME_BYTES:
                frame += bytes(_FRAME_BYTES - len(frame))
            audio_frame = agent_pb.AgentSessionMessage.ConsoleIO.AudioFrame(
                data=frame,
                sample_rate=_WIRE_SAMPLE_RATE,
                num_channels=_WIRE_CHANNELS,
                samples_per_channel=_FRAME_SAMPLES,
            )
            await self._send_message(agent_pb.AgentSessionMessage(audio_input=audio_frame))
            next_frame_at += frame_duration
            await asyncio.sleep(max(0.0, next_frame_at - asyncio.get_running_loop().time()))

    async def _send_message(self, message: agent_pb.AgentSessionMessage) -> None:
        if self._writer is None or self._writer.is_closing():
            raise RuntimeError("agent is not connected to the console TCP listener")
        payload = message.SerializeToString()
        self._writer.write(struct.pack(">I", len(payload)) + payload)
        await self._writer.drain()

    @staticmethod
    def _resample_to_wire(pcm: bytes, sample_rate: int) -> bytes:
        if sample_rate == _WIRE_SAMPLE_RATE:
            return pcm
        if len(pcm) % 2:
            raise ValueError("Kokoro returned malformed signed 16-bit PCM")
        input_frame = rtc.AudioFrame(
            data=pcm,
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(pcm) // 2,
        )
        resampler = rtc.AudioResampler(
            input_rate=sample_rate,
            output_rate=_WIRE_SAMPLE_RATE,
            num_channels=1,
        )
        frames = resampler.push(input_frame)
        frames.extend(resampler.flush())
        return b"".join(bytes(frame.data) for frame in frames)

    async def _drain_agent_logs(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        while line := await self._process.stdout.readline():
            self._agent_logs.append(line.decode("utf-8", errors="replace").rstrip())

    def _raise_if_disconnected(self) -> None:
        if self._reader_error is not None:
            raise RuntimeError(
                f"agent console TCP connection ended: {self._reader_error}\n{self._log_tail()}"
            ) from self._reader_error

    def _log_tail(self) -> str:
        return "\n".join(self._agent_logs) or "(no agent log output captured)"

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True

        server = self._server
        if server is not None:
            server.close()
            self._server = None

        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()

        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except (ConnectionError, OSError):
                pass
            self._writer = None

        if self._read_task is not None and not self._read_task.done():
            self._read_task.cancel()
            await asyncio.gather(self._read_task, return_exceptions=True)
        if self._log_task is not None:
            await asyncio.gather(self._log_task, return_exceptions=True)
        if server is not None:
            await server.wait_closed()

        await self._audio_worker.aclose()
