"""Progress reporting utilities for long-running optional model operations."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

_BAR_WIDTH = 28
_SPINNER = ("|", "/", "-", "\\")


def _truncate(text: str, max_len: int = 72) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[: max(0, max_len - 3)]}..."


def _build_bar(step: int) -> str:
    position = step % (2 * _BAR_WIDTH - 2)
    if position >= _BAR_WIDTH:
        position = 2 * _BAR_WIDTH - 2 - position
    cells = ["-"] * _BAR_WIDTH
    cells[position] = "="
    return "".join(cells)


@dataclass
class _TaskResult:
    value: Any = None
    error: BaseException | None = None
    complete: threading.Event = field(default_factory=threading.Event)


def run_with_progress(task: Callable[[], Any], progress_state: dict[str, str] | None = None) -> Any:
    """Run ``task`` on a worker thread and render a compact terminal heartbeat."""
    outcome = _TaskResult()

    def invoke() -> None:
        try:
            outcome.value = task()
        except BaseException as error:  # re-raised on the caller thread
            outcome.error = error
        finally:
            outcome.complete.set()

    worker = threading.Thread(target=invoke, name="raiw-progress-task", daemon=True)
    worker.start()
    started_at = time.monotonic()
    frame = 0
    terminal = sys.__stderr__
    while not outcome.complete.wait(0.1):
        message = _truncate((progress_state or {}).get("message", "Processing..."))
        elapsed = int(time.monotonic() - started_at)
        if terminal is not None:
            terminal.write(
                f"\r\033[2K  {_SPINNER[frame % len(_SPINNER)]} [{_build_bar(frame)}] {elapsed:>3}s  {message}"
            )
            terminal.flush()
        frame += 1

    worker.join()
    elapsed = int(time.monotonic() - started_at)
    message = _truncate((progress_state or {}).get("message", "Processing..."))
    if terminal is not None:
        terminal.write(f"\r\033[2K  Completed in {elapsed}s  {message}\n")
        terminal.flush()
    if outcome.error is not None:
        raise outcome.error
    return outcome.value


def _silence_diffusers() -> None:
    from diffusers.utils import logging as diffusers_logging

    diffusers_logging.set_verbosity_error()
    disable = getattr(diffusers_logging, "disable_progress_bar", None)
    if callable(disable):
        disable()


def _configure_quiet_libraries() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    operations = (
        lambda: __import__("transformers").logging.set_verbosity_error(),
        _silence_diffusers,
        lambda: __import__("huggingface_hub").logging.set_verbosity_error(),
    )
    for operation in operations:
        with contextlib.suppress(Exception):
            operation()


def silence_library_output(
    run_func: Callable[[], Any],
    set_progress: Callable[[str], None] | None = None,
) -> Callable[[], Any]:
    """Wrap a model call so third-party progress bars do not corrupt our CLI UI."""

    def quiet_call() -> Any:
        if set_progress is not None:
            set_progress("Preparing model runtime...")
        _configure_quiet_libraries()
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            warnings.simplefilter("ignore")
            if set_progress is not None:
                set_progress("Running watermark regeneration...")
            return run_func()

    return quiet_call


@dataclass
class _PipelineMonitor:
    total_steps: int
    device: str
    update: Callable[[str], None]
    bar_len: int
    label: str
    pre_phases: list[tuple[int, str]]
    post_phases: list[tuple[int, str]]
    first_step: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    started_at: float = field(default_factory=time.monotonic)
    last_step_at: float = field(default_factory=time.monotonic)

    def callback(self, step: int, _timestep: int, _latents: Any) -> None:
        self.first_step.set()
        now = time.monotonic()
        self.last_step_at = now
        current = min(self.total_steps, step + 1)
        filled = round(self.bar_len * current / self.total_steps)
        elapsed = now - self.started_at
        eta = elapsed * max(0, self.total_steps - current) / max(1, current)
        bar = "#" * filled + "." * (self.bar_len - filled)
        self.update(
            f"{self.label} [{bar}] {current}/{self.total_steps}, "
            f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left, {self.device}"
        )

    def _phase_message(self, phases: list[tuple[int, str]], elapsed: float) -> str:
        message = phases[0][1]
        for threshold, candidate in phases:
            if elapsed < threshold:
                break
            message = candidate
        return message

    def monitor(self) -> None:
        while not self.first_step.wait(0.4):
            elapsed = time.monotonic() - self.started_at
            self.update(self._phase_message(self.pre_phases, elapsed))
        decode_started: float | None = None
        while not self.done.wait(0.4):
            if time.monotonic() - self.last_step_at < 1.5:
                decode_started = None
                continue
            decode_started = decode_started or time.monotonic()
            self.update(self._phase_message(self.post_phases, time.monotonic() - decode_started))

    def start(self) -> threading.Thread:
        self.started_at = self.last_step_at = time.monotonic()
        self.first_step.clear()
        self.done.clear()
        thread = threading.Thread(target=self.monitor, name="raiw-pipeline-progress", daemon=True)
        thread.start()
        return thread


def make_pipeline_progress(
    effective_steps: int,
    device: str,
    set_progress: Callable[[str], None],
    *,
    bar_len: int = 20,
    label: str = "Denoising",
    pre_phases: list[tuple[int, str]] | None = None,
    post_phases: list[tuple[int, str]] | None = None,
) -> tuple[Callable[..., None], threading.Event, threading.Event, Callable[[], threading.Thread]]:
    """Build a callback and monitor for the legacy Diffusers callback interface."""

    def qualify(entries: list[tuple[int, str]]) -> list[tuple[int, str]]:
        return [(second, f"{text} on {device}") for second, text in entries]

    monitor = _PipelineMonitor(
        total_steps=max(1, effective_steps),
        device=device,
        update=set_progress,
        bar_len=bar_len,
        label=label,
        pre_phases=pre_phases or qualify([(0, "Encoding image"), (8, "Preparing denoiser"), (20, "Starting sampler")]),
        post_phases=post_phases or qualify([(0, "Decoding image"), (10, "Finalizing pixels"), (45, "Still decoding")]),
    )
    return monitor.callback, monitor.first_step, monitor.done, monitor.start


def is_mps_error(error: Exception) -> bool:
    """Return whether an error message identifies Apple's MPS backend."""
    return "mps" in str(error).casefold()
