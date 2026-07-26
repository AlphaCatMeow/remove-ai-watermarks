"""Render SYNTHETIC detection silhouettes for the CJK vendor text marks (data-safe).

Adding a mark needs only a DETECTION silhouette, and it must be font-rendered rather
than derived from user uploads: the corpus is real user content and may never reach a
tracked asset (see the repo CLAUDE.md data-safety rule). Seeing real samples to learn
the glyphs, weight and layout is fine; the committed template stays synthetic.

Covered here:
  qwen     "千问AI生成"  -- Alibaba Tongyi Qianwen, bottom-right, 3-lobed logo + text
  xinghui  "星绘AI生成"  -- ByteDance 星绘, bottom-right, 4-point sparkle + text
  yuanbao  "元宝\nAI生成" -- Tencent Yuanbao, bottom-right, two-line italic block
           (REGISTERED 2026-07-25 after fixing the negative-shear clipping in this
           renderer and matching both light and dark mark polarities)
  kling    "可灵AI 3.0"  -- Kuaishou Kling, bottom-right, spiral logo + text
           (REGISTERED 2026-07-21, kling_engine.py)

The leading LOGO is deliberately NOT rendered. It is the part that varies most between
releases and is hardest to reproduce synthetically, while the CJK run is stable and is
what actually discriminates one vendor from another (the shared `AI生成` tail is exactly
what does NOT discriminate -- see the rival-margin mechanism in _text_mark_engine).

Regenerate with:  uv run python scripts/render_vendor_silhouettes.py

STATUS 2026-07-21: `qwen_alpha.png` IS registered (`qwen_engine.py`) -- the 2026-07-18
blocker quoted below turned out to be mis-sized GEOMETRY (two size modes + a locate box
that clipped the first glyph), not segmentation, and was solved by the TC260-producer
cohort harvest + `vendor_mark_calibrate.py` (117 labelled frames; full record in
`docs/verification-plan.md`). `xinghui_alpha.png` is still NOT registered: one confirmed
corpus example is nothing to calibrate a gate against.

--- the 2026-07-18 record, kept as the history of the failed first attempt ---
Measured on 14 hand-verified 千问 positives from the corpus,
the then-current detect architecture (top-hat glyph blob -> binary TM_CCOEFF_NORMED)
could not see this mark AT ALL:

    same pipeline, each mark scored with its OWN template, on real positives
      doubao   n=40   mean NCC 0.723   median 0.835   >= 0.40 gate: 82%
      qwen     n=14   mean NCC 0.170   median 0.179   >= 0.40 gate:  0%

Three checks ruled out the obvious explanations, in order:
  1. NOT the synthetic render. A template cut from an ACTUAL Qwen mark scores the same
     as the font-rendered one (real-vs-real 0.307 vs synthetic 0.308) -- and real masks
     do not even match EACH OTHER.
  2. NOT the morphology kernel size. Scaling MORPH_OPEN/CLOSE with the box height (they
     are fixed 5px, ~9% of a 57px-tall box) gained only +0.014 mean and moved nothing
     across the gate.
  3. NOT the appearance thresholds. Sweeping tophat_delta / logo_min_luma / kernel
     reached at best mean 0.35 with 4/14 over the gate.

The blocker was named SEGMENTATION on a faint mark: Doubao is stamped bold and opaque,
so the white top-hat returns a clean glyph blob; the Qwen mark is a thin translucent
overlay that shatters into specks, and no template can match a blob that is not there.
The `tophat` front-end (built later, for doubao) removed that blocker -- and 千问 STILL
did not register, because the real residual was geometry. See the 2026-07-21 status
above.

星绘 additionally has only ONE confirmed example in the corpus, so even a working
front-end could not have its threshold calibrated yet.

UPDATE 2026-07-20: the named blocker is GONE, and the retry is still inconclusive.
`detect_frontend="tophat"` (built later, for doubao) is exactly the "grayscale correlation
on the raw top-hat" this note asked for, so the 2026-07-18 ruling rests on a premise that
no longer holds and must not simply be inherited. Two things were measured against it, and
neither settles the question:

  * A GENERIC template of the shared `AI生成` tail -- attractive because GB 45438-2025
    guarantees that run across vendors, so one template would cover 千问 / 百度 / 星绘 and
    anything compliant that ships next. Measured on the tophat front-end at the shipped
    3-rung ladder: a bold 千问 positive scores 0.407 against clean corners at p99 0.298 /
    max 0.321. It separates on that one frame, but only by a hair, and a 4-glyph template
    is inherently less specific than a 6-glyph one -- the shorter the run, the more
    arbitrary corner structure correlates with it.
  * The FULL 千问 template on the same front-end scores 0.248 against a clean max of 0.537,
    i.e. no separation at all -- WORSE than the generic tail, which is the opposite of
    what the specificity argument predicts and is itself a reason to distrust n=1.

The blocker is now EVIDENCE, not architecture: this session found exactly one 千问 and one
百度 positive (both by eyeballing doubao-provenance misses), and the 14 positives quoted
above were not preserved anywhere the current scripts can reach. Nothing should be
registered off a single frame.

UPDATE 2026-07-21 (the resolution): the evidence arrived via the TC260 producer-USCC
cohort trick (`scripts/vendor_cohort_harvest.py` -- 117 labelled 千问 frames from metadata
alone), and the registration shipped the same day (`qwen_engine.py`). The "no separation
at all" reading above was the MIS-SIZED geometry, not the mark: at the fitted geometry the
full template separates the cohort from clean corners 0.662 vs 0.134 (p50). The traps
below still bind any NEXT vendor: score with `alpha_height_frac`, not the silhouette's own
aspect ratio (the latter inflated the clean p99 from 0.30 to 0.58 and made every
comparison meaningless); keep the ladder at the shipped rungs for gate-setting, since a
wide sweep hands clean corners many extra chances to match; and re-filter the clean arm
per candidate -- the 2026-07-18 `present: []` labels mean "no REGISTERED mark", so qwen
-cohort frames visibly carrying 千问AI生成 sat in it (see `vendor_mark_calibrate.load_sets`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ASSETS = Path(__file__).resolve().parents[1] / "src" / "remove_ai_watermarks" / "assets"
# STHeiti Medium approximates the semibold CJK sans these marks are set in; the exact
# family is unpublished for every vendor (GB 45438-2025 only requires a legible face).
_FONT = "/System/Library/Fonts/STHeiti Medium.ttc"

MARKS = {
    "qwen_alpha.png": "千问AI生成",
    "xinghui_alpha.png": "星绘AI生成",
    # Yuanbao's stamp is a TWO-LINE block (元宝 over AI生成), left-aligned, tightly
    # stacked and ITALIC-SLANTED (measured on the 2026-07-21 cohort sheet + real tophat
    # responses); a rare one-line variant exists but the stacked block is dominant.
    "yuanbao_alpha.png": "元宝\nAI生成",
    # Kling (可灵) stamps a thin light-gray one-line "可灵AI 3.0" bottom-right (an
    # "Omni" suffix variant and a latin "KlingAI 3.0" variant also exist; the CJK
    # run without the suffix is the common core). The leading spiral logo is NOT
    # rendered (logos vary; the text run discriminates).
    "kling_alpha.png": "可灵AI 3.0",
    # The "cat-logo" cohort (USCC 91110108562144110X) stamps an outline cat-head +
    # bold "AI生成", bottom-right. PARKED 2026-07-21: the cohort is 19 copies of
    # only 2 unique carriers -- nothing to calibrate recall against (the xinghui
    # rule). The probe is ready: this silhouette scores 0.50 on the mark vs 0.333
    # max on a diverse clean arm, so registration is a gate pick (0.42) the moment
    # more unique carriers arrive.
    "catlogo_alpha.png": "CATLOGO",  # sentinel: drawn by draw_catlogo(), not font-rendered
    # RunningHub (ComfyUI platform, USCC 91340100MAEB4N8H76, 73-frame cohort
    # 2026-07-22): white one-line "RunningHub AI生成" text mark.
    "runninghub_alpha.png": "RunningHub AI生成",
    # LibLibAI / 哩布哩布AI (USCC 91110105MACJ6K1C8A, 15-frame cohort): white
    # "LibLibAI" wordmark with a triangle logo (logo not rendered, logos vary).
    "liblib_alpha.png": "LibLibAI",
    # Zhipu Qingyan (USCC 91110108MA01KP2T5U, 7-frame cohort): white bold
    # "清言·AI生成" with a circular logo (logo not rendered). PARKED 2026-07-22
    # as a measured negative: on both front-ends the cohort scores 0.34-0.39
    # against a clean-arm max of 0.34-0.37 -- no separation at any render/box
    # setting (text-only and logo-composite templates both plateau ~0.34 raw;
    # the white semi-transparent text on variable backgrounds is the wall).
    # Silhouette stays as the starting point for a structural/learned lever.
    "qingyan_alpha.png": "清言·AI生成",
    # MiniMax / Hailuo (6-frame cohort): "MINIMAX" + "Hailuo AI" latin wordmarks.
    # PARKED 2026-07-22: only 1 of the 6 cohort frames carries a visible mark --
    # nothing to calibrate recall against (the xinghui rule). Registration is a
    # gate pick once more unique carriers arrive.
    "hailuo_alpha.png": "Hailuo AI",
    # Baidu (USCC 91110000802100433B, 16-frame cohort): white bold "百度" text
    # + a separate white rounded tag with dark "AI生成", bottom-right. Detection
    # keys on the 百度 text run ONLY: a two-component template (text+pill) scored
    # at clean-arm levels (pill = bright-blob magnet, clean p95 0.45-0.55 vs cohort
    # ~0.5, no separation on either front-end, 2026-07-22); the text-only silhouette
    # separates (cohort 0.39-0.65 vs clean max 0.352). The white tag is removed
    # with the mark because the fill blob covers both bright components.
    "baidu_alpha.png": "百度",
}

# Per-mark post-processing for the multi-line / slanted stamps (see render()).
MARK_OPTS: dict[str, dict[str, Any]] = {
    # Re-fitted 2026-07-25 after the old affine transform was found to clip the
    # lower line and retain a large blank right half. Hiragino Sans GB W6, tight
    # leading, a 2px dilation, and -0.60 shear match the standard Yuanbao stamp.
    "yuanbao_alpha.png": {
        "gap_frac": 0.05,
        "dilate": 2,
        "shear": -0.60,
        "font": "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "font_index": 2,
    },
    # Qingyan's real stamp is a heavier weight than STHeiti Medium -- Hiragino
    # Sans GB W6 matches the measured stroke (2026-07-22; with Medium the
    # silhouette aspect came out 0.19 vs the real 0.28 and NCC plateaued ~0.3).
    "qingyan_alpha.png": {"font": "/System/Library/Fonts/Hiragino Sans GB.ttc", "font_index": 2},
    # LibLibAI's wordmark is set in an Arial-class grotesque, not STHeiti:
    # measured 2026-07-22 across 7 candidate fonts, Arial lifts the cohort
    # positives from 0.31-0.47 to 0.42-0.73 while the full-corpus false-fire arm
    # DROPS to max 0.398 (generic latin UI text matches the wrong font less).
    "liblib_alpha.png": {"font": "/System/Library/Fonts/Supplemental/Arial.ttf"},
}


def render(text: str, width: int = 335, opts: dict[str, Any] | None = None) -> np.ndarray:
    """Binary glyph silhouette (255 = glyph), sized to the doubao asset's convention.

    Matching doubao's 335px asset width keeps the `alpha_*_frac` numbers transferable,
    since these marks are the same house style and scale. A "\n" in ``text`` renders a
    multi-line block: lines drawn left-aligned at one shared font size with a tight
    gap, then optional stroke dilation and an italic shear (see MARK_OPTS).
    """
    opts = opts or {}
    gap_frac = float(opts.get("gap_frac", 0.15))
    dilate = int(opts.get("dilate", 0))
    shear_k = float(opts.get("shear", 0.0))
    font_path = str(opts.get("font", _FONT))
    font_index = int(opts.get("font_index", 0))
    probe = Image.new("L", (10, 10))
    d0 = ImageDraw.Draw(probe)
    lines = text.split("\n")
    size = 8
    while size < 200:  # grow until the LONGEST line fills the target width
        f = ImageFont.truetype(font_path, size, index=font_index)
        if max(d0.textbbox((0, 0), ln, font=f)[2] for ln in lines) >= width * 0.98:
            break
        size += 1
    font = ImageFont.truetype(font_path, size, index=font_index)
    boxes = [d0.textbbox((0, 0), ln, font=font) for ln in lines]
    line_h = max(bb[3] - bb[1] for bb in boxes)
    gap = max(1, int(line_h * gap_frac))
    w = max(bb[2] - bb[0] for bb in boxes)
    h = line_h * len(lines) + gap * (len(lines) - 1)
    pad = max(2, int(line_h * 0.12))
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    draw = ImageDraw.Draw(im)
    y = pad
    for ln, bb in zip(lines, boxes, strict=True):
        draw.text((pad - bb[0], y - bb[1]), ln, font=font, fill=255)
        y += line_h + gap
    sil = np.array(im)
    if dilate or shear_k:
        import cv2

        if dilate:
            sil = cv2.dilate(sil, np.ones((dilate, dilate), np.uint8))
        if shear_k:
            hh, ww = sil.shape
            extra = int(abs(shear_k) * hh)
            offset = extra if shear_k < 0 else 0
            sil = cv2.warpAffine(
                sil,
                np.float32([[1, shear_k, offset], [0, 1, 0]]),
                (ww + extra, hh),
            )
            ys, xs = np.where(sil > 0)
            if xs.size:
                sil = sil[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return sil


def draw_catlogo(width: int = 335) -> np.ndarray:
    """The cat-logo mark: an outline cat-head (integrated pointy ears, two dot eyes)
    + a bold "AI生成" run, drawn synthetically from the measured layout (cat ~1.08x
    the glyph height, stroke ~9%, gap ~35%). Proportions were iterated against a real
    tophat response (2026-07-21): a solid filled head scored 0.35, this outline form
    0.50 -- the parked probe, see MARKS."""
    probe = Image.new("L", (10, 10))
    d0 = ImageDraw.Draw(probe)
    text = "AI生成"
    size = 8
    while size < 200:
        f = ImageFont.truetype(_FONT, size)
        if d0.textbbox((0, 0), text, font=f)[2] >= width * 0.60:
            break
        size += 1
    font = ImageFont.truetype(_FONT, size)
    bb = d0.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    cs = int(th * 1.08)
    stroke = max(2, int(th * 0.09))
    gap = int(th * 0.35)

    def head(s: int) -> Image.Image:
        im = Image.new("L", (s, s), 0)
        d = ImageDraw.Draw(im)
        f = float(s)
        pts = [
            (0.12 * f, 0.95 * f),
            (0.10 * f, 0.45 * f),
            (0.12 * f, 0.30 * f),
            (0.20 * f, 0.05 * f),  # left ear tip
            (0.40 * f, 0.24 * f),  # left ear valley
            (0.60 * f, 0.24 * f),  # right ear valley
            (0.80 * f, 0.05 * f),  # right ear tip
            (0.88 * f, 0.30 * f),
            (0.90 * f, 0.45 * f),
            (0.88 * f, 0.95 * f),
        ]
        d.line([*pts, pts[0]], fill=255, width=stroke, joint="curve")
        r = max(1.5, stroke * 0.7)
        d.ellipse([0.35 * f - r, 0.60 * f - r, 0.35 * f + r, 0.60 * f + r], fill=255)
        d.ellipse([0.65 * f - r, 0.60 * f - r, 0.65 * f + r, 0.60 * f + r], fill=255)
        return im

    w = cs + gap + tw
    h = max(th, cs)
    pad = max(2, int(h * 0.12))
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    im.paste(head(cs), (pad, pad + (h - cs) // 2))
    ImageDraw.Draw(im).text((pad + cs + gap - bb[0], pad + (h - th) // 2 - bb[1]), text, font=font, fill=255)
    return np.array(im)


def main() -> None:
    try:
        for name, text in MARKS.items():
            sil = draw_catlogo() if text == "CATLOGO" else render(text, opts=MARK_OPTS.get(name))
            Image.fromarray(sil).save(_ASSETS / name)
            print(f"wrote {_ASSETS / name}  ({sil.shape[1]}x{sil.shape[0]})  text={text!r}")
    except OSError as e:
        print(f"Font not found ({e}); install a CJK font or edit _FONT.", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
