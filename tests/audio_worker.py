from __future__ import annotations

import base64
import contextlib
import json
import math
import os
import sys
import traceback
from typing import Any, TextIO


def _emit(stream: TextIO, value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False), file=stream, flush=True)


def _load_models() -> tuple[Any, Any, Any, Any, Any]:
    import numpy as np
    import torch
    import whisper
    from kokoro import KPipeline
    from scipy import signal

    torch.set_num_threads(max(1, int(os.environ.get("LK_TEST_AUDIO_THREADS", "2"))))
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    whisper_model = whisper.load_model(
        os.environ.get("LK_TEST_WHISPER_MODEL", "base.en"), device="cpu"
    )
    return np, signal, pipeline, torch, whisper_model


def _tts(np: Any, torch: Any, pipeline: Any, text: str) -> bytes:
    chunks = []
    for result in pipeline(
        text,
        voice=os.environ.get("LK_TEST_KOKORO_VOICE", "af_heart"),
        speed=1.0,
    ):
        audio = result.audio
        if audio is None:
            continue
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().numpy()
        chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))

    if not chunks:
        raise RuntimeError("Kokoro returned no speech audio")
    waveform = np.concatenate(chunks)
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    return pcm.tobytes()


def _transcribe(
    np: Any,
    signal: Any,
    whisper_model: Any,
    pcm_s16le: bytes,
    sample_rate: int,
    channels: int,
) -> str:
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("sample_rate and channels must be positive")

    samples = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        usable_samples = len(samples) - (len(samples) % channels)
        samples = samples[:usable_samples].reshape(-1, channels).mean(axis=1)
    if sample_rate != 16_000:
        divisor = math.gcd(sample_rate, 16_000)
        samples = signal.resample_poly(
            samples,
            up=16_000 // divisor,
            down=sample_rate // divisor,
        )
    samples = np.asarray(samples, dtype=np.float32)

    result = whisper_model.transcribe(
        samples,
        fp16=False,
        language="en",
        temperature=0,
        condition_on_previous_text=False,
        verbose=False,
    )
    return str(result.get("text", "")).strip()


def main() -> int:
    protocol_stdout = sys.stdout
    # Some dependencies spawn installers that inherit the OS stdout fd, bypassing
    # contextlib.redirect_stdout and corrupting the JSON-lines protocol.
    saved_stdout_fd = os.dup(sys.stdout.fileno())
    load_error: Exception | None = None
    try:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        with contextlib.redirect_stdout(sys.stderr):
            np, signal, pipeline, torch, whisper_model = _load_models()
    except Exception as exc:
        load_error = exc
    finally:
        os.dup2(saved_stdout_fd, sys.stdout.fileno())
        os.close(saved_stdout_fd)

    if load_error is not None:
        traceback.print_exception(
            type(load_error), load_error, load_error.__traceback__, file=sys.stderr
        )
        _emit(
            protocol_stdout,
            {"ready": False, "error": f"{type(load_error).__name__}: {load_error}"},
        )
        return 1

    _emit(protocol_stdout, {"ready": True})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            action = request.get("action")
            with contextlib.redirect_stdout(sys.stderr):
                if action == "tts":
                    result = {
                        "audio": base64.b64encode(
                            _tts(np, torch, pipeline, str(request["text"]))
                        ).decode("ascii"),
                        "sample_rate": 24_000,
                    }
                elif action == "transcribe":
                    pcm = base64.b64decode(request["audio"], validate=True)
                    result = {
                        "text": _transcribe(
                            np,
                            signal,
                            whisper_model,
                            pcm,
                            int(request["sample_rate"]),
                            int(request.get("channels", 1)),
                        )
                    }
                else:
                    raise ValueError(f"unknown audio worker action: {action!r}")
            _emit(protocol_stdout, result)
        except Exception as exc:
            _emit(protocol_stdout, {"error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
