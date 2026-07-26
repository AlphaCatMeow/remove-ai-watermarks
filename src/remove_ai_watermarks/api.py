"""High-level convenience API: clean an image (or array) in one call.

The low-level building blocks live in ``watermark_registry`` (localize -> fill) and
``image_io`` (Unicode-safe, alpha-preserving IO). This module ties them into the two
calls a caller usually wants, so a library user does not have to decode images, wire
up metadata provenance, or preserve the alpha channel by hand:

    import remove_ai_watermarks as raiw
    raiw.remove_visible("in.png", "out.png")                 # path -> file, provenance auto
    result, removed = raiw.remove_visible(bgr_array)         # array -> array
    raiw.remove_visible("shot.png", "out.png", sensitivity="strict")
    raiw.visible_provenance("in.png")                        # -> frozenset({"gemini"})

Imports stay lazy (inside the functions), so ``import remove_ai_watermarks`` is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from remove_ai_watermarks.watermark_registry import Backend, Sensitivity


@dataclass(frozen=True)
class _VisibleInput:
    """Normalized visible-removal input with its file-only context."""

    bgr: NDArray[Any]
    alpha: NDArray[Any] | None = None
    path: Path | None = None
    provenance: frozenset[str] = frozenset()


def visible_provenance(source: str | Path) -> frozenset[str]:
    """Vendor keys that the file's local metadata confirms, the evidence that drives
    the ``auto`` sensitivity (relaxing a corroborated mark's detection trust gate).

    Mapping: a Google/Gemini C2PA issuer -> ``"gemini"``; a China-AIGC (TC260) label
    -> ``"doubao"``/``"jimeng"``; a ``samsung_genai`` marker -> ``"samsung"``.
    Best-effort: any read error yields an empty set (no relaxation). Metadata-only, so
    it never loads cv2/torch.
    """
    import contextlib

    path = Path(source)
    with contextlib.suppress(Exception):
        from remove_ai_watermarks import identify

        rep = identify.identify(path, check_visible=False, check_invisible=False)
        signal_names = {signal.name for signal in rep.signals}
        keys: set[str] = set()
        platform = (rep.platform or "").lower()
        if "google" in platform or "gemini" in platform:
            keys.add("gemini")
        if "aigc" in signal_names:
            keys |= {"doubao", "jimeng"}
        if "samsung_genai" in signal_names:
            keys.add("samsung")
        return frozenset(keys)
    return frozenset()


def _load_visible_input(source: str | Path | NDArray[Any]) -> _VisibleInput:
    """Normalize a path/array source without making the public operation stateful."""
    if not isinstance(source, (str, Path)):
        return _VisibleInput(source)

    from remove_ai_watermarks import image_io

    path = Path(source)
    bgr, alpha = image_io.read_bgr_and_alpha(path)
    if bgr is None:
        raise ValueError(f"Could not read image: {source}")
    return _VisibleInput(bgr=bgr, alpha=alpha, path=path, provenance=visible_provenance(path))


def _write_visible_result(
    loaded: _VisibleInput,
    result: NDArray[Any],
    removed: list[str],
    output: str | Path,
    *,
    strip_metadata: bool,
    write_noop: bool,
) -> None:
    """Write one visible-removal result while preserving a true no-op losslessly."""
    if not removed and not write_noop:
        return

    from remove_ai_watermarks import image_io

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = loaded.path
    if not removed and source_path is not None and source_path.suffix.lower() == out_path.suffix.lower():
        # Copy the ORIGINAL bytes instead of lossily re-encoding a no-op. An in-place
        # call needs no copy and would otherwise raise shutil.SameFileError.
        if source_path.resolve() != out_path.resolve():
            import shutil

            shutil.copyfile(source_path, out_path)
    else:
        # imwrite is contractually NON-RAISING, so this bool is the only signal the file
        # was not created. Unchecked, the metadata strip below ran on a nonexistent path
        # and surfaced as a confusing "cannot read image <INPUT>" naming the OUTPUT path
        # (Tier E, 2026-07-20). Raise here so a library caller and the CLI both get an
        # accurate message about the write.
        if not image_io.write_bgr_with_alpha(out_path, result, loaded.alpha):
            raise OSError(f"failed to write output (is the destination writable?): {out_path}")

    if strip_metadata:
        from remove_ai_watermarks import metadata

        metadata.remove_ai_metadata(out_path, out_path)


def remove_visible(
    source: str | Path | NDArray[Any],
    output: str | Path | None = None,
    *,
    sensitivity: Sensitivity = "auto",
    backend: Backend = "auto",
    strip_metadata: bool = True,
    write_noop: bool = True,
) -> tuple[NDArray[Any], list[str]]:
    """Remove every detected known visible AI mark through localize then fill.

    The registry currently covers the Gemini sparkle; Doubao, Jimeng, Qwen, Kling,
    Yuanbao, Samsung, RunningHub, Baidu, and LibLibAI text marks; and the Jimeng
    pill. Returns ``(result_bgr, [labels removed])``.

    ``source`` is a file path OR a BGR ndarray. For a PATH, metadata provenance is read
    automatically (so ``sensitivity="auto"`` recovers a moved/faint mark whenever the
    file still carries its provenance) and the alpha channel is preserved on write; for
    an ARRAY there is no metadata to read and no separate alpha plane. When ``output``
    is given the cleaned image is written there (alpha rejoined for a path source); the
    array is always returned as well, so an empty ``removed`` list tells a caller nothing
    known was found (e.g. route to the diffusion ``all`` path or ``erase``).

    ``sensitivity`` (``auto``/``strict``) and ``backend``
    (``auto``/``cv2``/``migan``/``lama``) are the same controls as the CLI.

    ``strip_metadata`` (default True, matching the CLI ``visible --strip-metadata``)
    also strips AI provenance metadata (C2PA/EXIF/XMP/IPTC) from the written output via
    the lossless :func:`metadata.remove_ai_metadata`, so a library call does exactly
    what the CLI does. Only applies when ``output`` is given.

    ``write_noop`` (default True) controls whether ``output`` is written when NOTHING was
    removed: True writes a clean passthrough copy (an idempotent clean); False leaves the
    output path untouched, so a caller that treats "no mark" as "produce nothing" (the CLI
    ``visible`` no-mark contract) does not clobber a pre-existing file at that path.
    """
    from remove_ai_watermarks import watermark_registry

    # Reject a removed sensitivity loudly; `Sensitivity` is a Literal and not enforced
    # at runtime, so a 0.15 caller would otherwise get `auto` behaviour in silence.
    watermark_registry.validate_sensitivity(sensitivity)
    loaded = _load_visible_input(source)
    result, removed = watermark_registry.remove_auto_marks(
        loaded.bgr,
        sensitivity=sensitivity,
        provenance=loaded.provenance,
        backend=backend,
    )
    if output is not None:
        _write_visible_result(
            loaded,
            result,
            removed,
            output,
            strip_metadata=strip_metadata,
            write_noop=write_noop,
        )
    return result, removed
