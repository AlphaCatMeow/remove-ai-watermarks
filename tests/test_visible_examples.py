"""The visible-mark example gallery is complete and self-consistent.

Two failures this suite exists to catch:
  * a mark registered without a committed example (the gallery lags the registry);
  * an engine that no longer detects its own canonical example (the gallery is
    generated from the engines' measured geometry, so this is a regression tripwire).

The examples are SYNTHETIC (``scripts/render_visible_examples.py`` composites the
committed silhouettes onto a generated base). User uploads never enter the repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from remove_ai_watermarks import watermark_registry as wr
from remove_ai_watermarks.image_io import imread
from remove_ai_watermarks.video import VIDEO_VISIBLE_MARKS, identify_video

_ROOT = Path(__file__).resolve().parents[1]
_GALLERY = _ROOT / "data" / "fixtures" / "visible"

_IMAGE_KEYS = [m.key for m in wr.known_marks()]


class TestGallery:
    def test_every_registered_mark_has_an_example(self) -> None:
        missing = [key for key in _IMAGE_KEYS if not (_GALLERY / key / "example.png").is_file()]
        assert missing == [], f"registered without an example: {missing}; run scripts/render_visible_examples.py"

    @pytest.mark.parametrize("key", _IMAGE_KEYS)
    def test_engine_detects_its_own_example(self, key: str) -> None:
        img = imread(str(_GALLERY / key / "example.png"))
        assert img is not None, key
        det = wr.get_mark(key).detect(img, provenance=False)
        assert det.detected, f"{key}: confidence {det.confidence:.3f} on its own example"

    def test_gallery_has_no_stray_directories(self) -> None:
        known = set(_IMAGE_KEYS) | set(VIDEO_VISIBLE_MARKS) | {"README.md"}
        extra = sorted(p.name for p in _GALLERY.iterdir() if p.name not in known)
        assert extra == [], f"gallery holds unregistered examples: {extra}; remove or register them"


class TestVideoGallery:
    def test_every_registered_video_mark_has_an_example(self) -> None:
        missing = [key for key in VIDEO_VISIBLE_MARKS if not (_GALLERY / key / "example.mp4").is_file()]
        assert missing == [], f"video mark without an example: {missing}; run scripts/render_visible_examples.py"

    def test_selection_accepts_each_example(self) -> None:
        # The shipped temporal selection (not just the per-frame detector) must
        # accept the clip: table order resolves cross-template ties, so the example
        # must carry the discriminative variant of its mark.
        for key in VIDEO_VISIBLE_MARKS:
            rep = identify_video(_GALLERY / key / "example.mp4", check_visible=True)
            assert rep.visible_mark == key, f"{key}: selection returned {rep.visible_mark!r}"
