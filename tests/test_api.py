"""High-level convenience API (remove_visible / visible_provenance) and the lazy
top-level re-exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import remove_ai_watermarks as raiw
from remove_ai_watermarks import api

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "provenance"
DOUBAO = SAMPLES / "doubao-1.png"
CHATGPT = SAMPLES / "chatgpt-1.png"


class TestTopLevelExports:
    def test_lazy_reexports_resolve(self):
        assert raiw.remove_visible is api.remove_visible
        assert raiw.visible_provenance is api.visible_provenance

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            _ = raiw.does_not_exist

    def test_bare_import_is_light(self):
        # importing the package must not pull the heavy cv2/torch stack (PEP 562 lazy).
        # Checked in a FRESH interpreter -- another test in this process may already
        # have imported cv2, so an in-process sys.modules check would be flaky.
        import subprocess
        import sys

        code = "import remove_ai_watermarks, sys; print(int(any(m in sys.modules for m in ('cv2','torch'))))"
        out = subprocess.run(  # noqa: S603  -- fixed sys.executable + literal code, no untrusted input
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        )
        assert out.stdout.strip() == "0", f"bare import pulled a heavy module: {out.stdout!r}"


class TestRemoveVisibleArray:
    def test_array_no_mark_is_noop_copy(self):
        arr = np.zeros((256, 256, 3), np.uint8)
        result, removed = raiw.remove_visible(arr, backend="cv2")
        assert removed == []
        assert result.shape == arr.shape
        assert np.array_equal(result, arr)

    def test_array_accepts_knobs(self):
        arr = np.zeros((256, 256, 3), np.uint8)
        result, removed = raiw.remove_visible(arr, sensitivity="strict", backend="cv2")
        assert removed == []
        assert result.shape == arr.shape

    def test_bad_source_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Could not read image"):
            raiw.remove_visible(tmp_path / "nope.png")


@pytest.mark.skipif(not DOUBAO.exists(), reason="doubao sample not present")
class TestRemoveVisiblePath:
    def test_path_removes_and_writes(self, tmp_path):
        out = tmp_path / "clean.png"
        result, removed = raiw.remove_visible(DOUBAO, out, backend="cv2")
        assert out.exists()
        assert any("Doubao" in lbl for lbl in removed)
        assert result.shape[2] == 3

    def test_path_no_output_returns_without_writing(self, tmp_path):
        # output=None returns the array but writes nothing
        result, _ = raiw.remove_visible(DOUBAO, backend="cv2")
        assert result.ndim == 3


class TestNoOpPreservesOriginal:
    def test_no_mark_copies_original_bytes(self, tmp_path):
        # A clean image (no mark) same-format-out must be copied VERBATIM, not
        # re-encoded -- so a no-op never degrades the original ("work with originals").
        import filecmp

        from PIL import Image

        src = tmp_path / "clean.jpg"
        Image.fromarray(np.full((40, 40, 3), 120, np.uint8), "RGB").save(src, quality=90)
        out = tmp_path / "clean_out.jpg"
        _, removed = raiw.remove_visible(str(src), str(out), sensitivity="strict", backend="cv2")
        assert removed == []
        assert filecmp.cmp(str(src), str(out), shallow=False)  # byte-identical


class TestVisibleProvenance:
    @pytest.mark.skipif(not DOUBAO.exists(), reason="doubao sample not present")
    def test_doubao_tc260_maps_to_bytedance(self):
        prov = raiw.visible_provenance(DOUBAO)
        # TC260 label -> ByteDance family (both doubao and jimeng)
        assert {"doubao", "jimeng"} <= prov

    @pytest.mark.skipif(not CHATGPT.exists(), reason="chatgpt sample not present")
    def test_openai_image_has_no_visible_vendor(self):
        # OpenAI C2PA is not one of the visible-mark vendors -> empty provenance
        assert raiw.visible_provenance(CHATGPT) == frozenset()

    def test_unreadable_path_is_empty(self, tmp_path):
        assert raiw.visible_provenance(tmp_path / "missing.png") == frozenset()

    def test_uses_report_signals_for_falsy_metadata_values(self, monkeypatch, tmp_path):
        """An empty TC260 object and Samsung genAIType=0 are still present signals.

        The report has already normalized those values, so the public API must not
        re-read the file and accidentally discard them by truthiness.
        """
        from types import SimpleNamespace

        from remove_ai_watermarks import identify

        report = SimpleNamespace(
            platform=None,
            signals=[SimpleNamespace(name="aigc"), SimpleNamespace(name="samsung_genai")],
        )
        monkeypatch.setattr(identify, "identify", lambda *args, **kwargs: report)

        assert raiw.visible_provenance(tmp_path / "synthetic.png") == frozenset({"doubao", "jimeng", "samsung"})


class TestRemoveVisibleOutputPath:
    """Output-path robustness: in-place clean (#3) and a missing output dir (#4)."""

    def _write_clean(self, p: Path) -> None:
        from remove_ai_watermarks import image_io

        image_io.imwrite(str(p), np.full((128, 128, 3), 200, np.uint8))

    def test_inplace_clean_no_crash(self, tmp_path: Path):
        p = tmp_path / "clean.png"
        self._write_clean(p)
        _, removed = raiw.remove_visible(str(p), str(p), backend="cv2")
        assert removed == []
        assert p.exists()

    def test_creates_missing_output_dir(self, tmp_path: Path):
        src = tmp_path / "in.png"
        self._write_clean(src)
        out = tmp_path / "sub" / "out.png"
        raiw.remove_visible(str(src), str(out), backend="cv2")
        assert out.exists()
