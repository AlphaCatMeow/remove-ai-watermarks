"""DWT-DCT decoder compatible with invisible-watermark's ``dwtDct`` path.

Derived from ShieldMnt/invisible-watermark ``imwatermark/maxDct.py`` (MIT),
trimmed to the matrix path used by Stable Diffusion, SDXL, and FLUX. The block
scan is vectorized rather than transcribed, so the file no longer reads line by
line against upstream; what it preserves is the output, bit for bit. See
[`docs/module-internals.md`](../../docs/module-internals.md) for the
measurements and for why a faster hand-rolled transform is not available.

Copyright (c) 2021 ShieldMnt

The complete upstream license is distributed in
``licenses/invisible-watermark-MIT.txt``.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pywt

if TYPE_CHECKING:
    from numpy.typing import NDArray

_DEFAULT_SCALES = (0, 36, 36)
_DEFAULT_BLOCK = 4


class _DecodeMaxDct:
    """Extract frequency-domain bits using the upstream matrix algorithm."""

    def __init__(
        self,
        wm_lengths: tuple[int, ...],
        scales: tuple[int, int, int] = _DEFAULT_SCALES,
        block: int = _DEFAULT_BLOCK,
    ) -> None:
        self._wm_lengths = wm_lengths
        self._scales = scales
        self._block = block

    def decode(self, bgr: NDArray[Any]) -> dict[int, NDArray[Any]]:
        row, col, _channels = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        trimmed = yuv[: row // 4 * 4, : col // 4 * 4]

        per_channel = [
            self._frame_bits(self._approximation(trimmed, channel), self._scales[channel])
            for channel in range(2)
            if self._scales[channel] > 0
        ]
        # Each channel restarts the bit index at 0, so the buckets come from a
        # per-channel arange rather than one running counter.
        index = np.concatenate([np.arange(bits.size) for bits in per_channel] or [np.zeros(0, dtype=np.int64)])
        weights = np.concatenate(per_channel or [np.zeros(0)])

        decoded: dict[int, NDArray[Any]] = {}
        for wm_len in self._wm_lengths:
            bucket = index % wm_len
            sums = np.bincount(bucket, weights=weights, minlength=wm_len)
            counts = np.bincount(bucket, minlength=wm_len)
            decoded[wm_len] = sums * 255 > counts * 127
        return decoded

    @staticmethod
    def _approximation(trimmed: NDArray[Any], channel: int) -> NDArray[Any]:
        """The Haar approximation band, and only it.

        ``dwt2`` is ``dwtn``: it transforms along axis 0, then along axis 1 over
        both halves, and three of the four bands it returns are discarded here.
        Two ``dwt`` calls keeping ``[0]`` skip that, and transposing between them
        lets pywt walk a contiguous axis instead of a column.

        The result must stay bit-identical to ``dwt2``'s, which is why the
        transform is left to pywt however slow that is: the caller's threshold is
        ``peak % 36 > 18.0``, and for uint8 input the exact value is a multiple
        of 0.5, so it lands exactly on the threshold often enough that a 1-ulp
        difference flips real bits.
        """
        if trimmed.shape[0] == 0 or trimmed.shape[1] == 0:
            # Reachable: a 1x65536 image clears the caller's area check. Left to
            # dwt2 so the exception stays the one this module has always raised.
            return pywt.dwt2(trimmed[:, :, channel], "haar")[0]
        columns = cv2.transpose(cv2.extractChannel(trimmed, channel))
        along_rows = pywt.dwt(columns, "haar", axis=1)[0]
        return pywt.dwt(cv2.transpose(along_rows), "haar", axis=1)[0]

    def _frame_bits(self, frame: NDArray[Any], scale: int) -> NDArray[Any]:
        """One bit per 4x4 block, in row-major block order.

        Upstream's per-block loop, said to numpy once instead of to the
        interpreter ~135k times per image.
        """
        block = self._block
        rows = frame.shape[0] // block
        cols = frame.shape[1] // block
        if rows == 0 or cols == 0:
            return np.zeros(0, dtype=np.float64)
        aligned = frame[: rows * block, : cols * block]
        blocks = aligned.reshape(rows, block, cols, block).swapaxes(1, 2)
        peak = np.abs(blocks.reshape(rows * cols, block * block)[:, 1:]).max(axis=1)
        return ((peak % scale) > 0.5 * scale).astype(np.float64)


def decode_dwt_dct(bgr: NDArray[Any], wm_len: int) -> NDArray[Any]:
    """Extract ``wm_len`` watermark bits from a BGR image."""
    return decode_dwt_dct_lengths(bgr, (wm_len,))[wm_len]


def decode_dwt_dct_lengths(bgr: NDArray[Any], wm_lengths: tuple[int, ...]) -> dict[int, NDArray[Any]]:
    """Extract several watermark lengths with one DWT and block scan."""
    if bgr.size == 0 or min(bgr.shape[:2]) * max(bgr.shape[:2]) < 256 * 256:
        raise RuntimeError("image too small, should be larger than 256x256")
    if not wm_lengths or any(wm_len <= 0 for wm_len in wm_lengths):
        raise ValueError("watermark lengths must be positive")
    return _DecodeMaxDct(wm_lengths=tuple(dict.fromkeys(wm_lengths))).decode(bgr)
