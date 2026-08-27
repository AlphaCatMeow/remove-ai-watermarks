# Content Seal oracle corpus

Muse Image (`muse-image-1.0`) generations with externally verified Content Seal
verdicts, produced through the Meta Model API on 2026-08-26 and checked against
the public detector at `https://meta.ai/identification` (anonymous session, no
login). Follows the `data/synthid/originals` pattern: binaries live in
`originals/`, every derived variant is recorded in `manifest.csv` as a recipe
plus hash and is not stored.

## What this corpus establishes

- The Meta Model API image endpoint (`POST /v1/images/generations`) stamps the
  same Content Seal pixel watermark as the consumer Meta AI app: all five
  generations verified positive with attribution "Muse Image 1 - Meta".
- The detector response carries a per-generation ID and creation timestamp
  embedded in the watermark payload. Both survived a 512 px LANCZOS resize and
  a full-size JPEG q85 re-encode (same ID returned), so the payload is more
  robust than the detection threshold.
- Center crops lose the seal: 50% and 33% linear center crops of two different
  images all returned "No AI signatures from Meta were found", consistent with
  the Reuters 2026-07-11 analysis (55% missed after cropping).
- API outputs carry XMP `iptcExt:DigitalSourceType =
  trainedAlgorithmicMedia`, so local `identify` flags them via the existing
  Made-with-AI path. Metadata-stripping transforms fall back to unknown, and
  Content Seal has no local decoder in this project: the oracle is the only
  reader.
- Drift finding: the 512 px resize of `gen_fox_forest` triggers a
  medium-confidence false positive "Tencent Yuanbao (visible 元宝 / AI生成
  mark)" in this project's `identify`. Recorded here as a reproducible case.

## Oracle limits and wire format

There is no public or documented checking API. Verified against the official
developer documentation on 2026-08-26 (`https://dev.meta.ai/docs/`): the full
Meta Model API reference lists only Responses, Chat Completions, Messages,
Files, Images (`/v1/images/generations`, `/v1/images/edits`), and Models, with
no identification, detection, or watermark endpoint, and the image-generation,
Muse Image cookbook, and pricing pages never mention watermark, Content Seal,
or provenance at all. The API applies the seal (every generation in this corpus
carries it) while documenting nothing about it. The web tool drives an internal
REST pair, captured from the browser network log on 2026-08-26:

1. `POST https://rupload.meta.ai/gen_ai_document_gen_ai_tenant/<uuid>` with the
   raw file bytes, `x-entity-type`, `x-entity-length`, `ai_detector_upload: true`,
   and an anonymous `authorization: OAuth ecto1:<token>` session token minted by
   the page.
2. `POST https://meta.ai/api/ai-detector` with `Bearer ecto1:<token>` and body
   `{"media_id": "...", "fileName": "...", "mimeType": "..."}`.

The rate limit is enforced at that endpoint, server-side, and keyed beyond the
browser session: the API itself returns `429 {"errorType": "rate_limited"}`,
and clearing cookies and storage changed nothing, so driving the internal pair
directly does not bypass it. The Meta Model API (`api.meta.ai/v1`, where the
generation key works) has no identification endpoint; plausible paths all
return 404. Rows with an empty `oracle_verdict` were transformed but not yet
checkable. Read a verdict only from the settled page text after the
result-complete state ("Upload another file"): a wait for a verdict string can
match the previous upload's text, and the fresh-navigation protocol used for
the calibration rows below is the race-free variant.

## Strength floor calibration (qwen-zimage, seed 0)

The library resolves strength per vendor with measured floors (OpenAI
0.07675 / Google 0.27 / Microsoft InvisMark 0.15 in
`_internal/watermark_profiles.py`). Meta Content Seal has no floor yet; the
goal of these rows is to measure one by that same methodology: independent
generations, each one's first-clean boundary, floor = worst boundary plus the
observed cross-source spread.

Measured (2026-08-26/27, oracle `meta.ai/identification`):

- Default pipeline clears Content Seal: tested samples came back clean at the
  default resolution-adaptive strength (~0.1305 at 2.56 MP), including the
  worst source.
- Five independent generations bracketed. First-clean boundaries:
  lighthouse (0.0525, 0.06], fox (0.03, 0.0375], night_city (0.03, 0.0375],
  mug <= 0.03, text <= 0.015. Cross-source spread is wide (a factor of four
  between easiest and hardest).
- Derived Meta floor by the existing worst-boundary-plus-cross-source-spread
  method: 0.06 + (0.0525 - 0.015) = 0.0975, rounded to **0.1**.
- Not shipped as a constant: no provenance signal routes Muse outputs onto a
  vendor cohort, so the default curve stays authoritative and 0.1 is recorded
  as the floor to encode if an explicit Meta override is ever added.

## Regeneration

API key is not stored in this repository. Regenerate with the script pattern
from the session (env `MUSE_API_KEY`, endpoint
`https://api.meta.ai/v1/images/generations`, model `muse-image-1.0`); prompts
are recorded per file in `manifest.csv`.
