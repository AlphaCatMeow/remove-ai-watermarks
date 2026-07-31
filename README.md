# Remove AI Watermarks

Remove AI provenance marks from images and video you generated yourself:

- known visible labels such as the Gemini sparkle and vendor text marks;
- invisible pixel watermarks through diffusion regeneration;
- C2PA, EXIF, XMP, IPTC, and related AI metadata.

Video support covers metadata inspection and removal, visible Sora, Veo,
Seedance, Dola, Hailuo, and Kling mark removal, and experimental VAE
regeneration that produces a video SynthID candidate for external verification.

> Try it online at [raiw.cc](https://raiw.cc) if you do not want to install Python
> or run diffusion models locally.

[![PyPI](https://img.shields.io/pypi/v/remove-ai-watermarks?logo=pypi&logoColor=white)](https://pypi.org/project/remove-ai-watermarks/)
[![Python](https://img.shields.io/pypi/pyversions/remove-ai-watermarks?logo=python&logoColor=white)](https://pypi.org/project/remove-ai-watermarks/)
[![Downloads](https://static.pepy.tech/badge/remove-ai-watermarks/month)](https://pepy.tech/project/remove-ai-watermarks)
[![License](https://img.shields.io/pypi/l/remove-ai-watermarks?color=blue)](LICENSE)
[![Tests](https://github.com/wiltodelta/remove-ai-watermarks/actions/workflows/test.yml/badge.svg)](https://github.com/wiltodelta/remove-ai-watermarks/actions/workflows/test.yml)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-db61a2?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/wiltodelta)

> This project is for lawful use on content you own. It does not target stock
> agency previews or other watermarks that protect third party paid content.
> See [scope, safety, and legal notes](docs/legal-and-safety.md).

## Choose what you want to do

| Goal | Command | GPU |
| --- | --- | --- |
| Find provenance signals and watermarks | `identify` | No |
| Remove known visible AI marks | `visible` | No |
| Erase a region you select | `erase` | No |
| Strip AI metadata | `metadata` | No |
| Strip AI metadata from video | `video metadata` | No |
| Remove a registered visible AI mark from video | `video visible` | No |
| Generate an externally verifiable video SynthID candidate | `video invisible` | Recommended |
| Regenerate an image to disrupt invisible watermarks | `invisible` | Recommended |
| Run visible, invisible, and metadata removal | `all` | Recommended |
| Process a directory | `batch` | Depends on mode |

## Quick start

Install the core CLI:

```bash
uv tool install remove-ai-watermarks
```

Inspect an image:

```bash
remove-ai-watermarks identify image.png
```

Remove a known visible mark and AI metadata:

```bash
remove-ai-watermarks visible image.png -o clean.png
```

Strip metadata without running visible inpainting or diffusion:

```bash
remove-ai-watermarks metadata image.png --remove -o clean.png
```

Inspect or remove AI metadata from an MP4, MOV, M4V, WebM, MKV, AVI, or FLV
file:

```bash
remove-ai-watermarks video metadata input.mp4 --check
remove-ai-watermarks video metadata input.mp4 --remove -o clean.mp4
```

The metadata command does not transcode video or audio streams. When `-o` is
omitted it writes `<source>_clean` and preserves the original. MP4 and MOV
inspection includes the native TC260 `AIGC` tag in
`moov.udta.meta.keys/ilst`, including a `moov` placed after the media payload.
MKV and WebM inspection reads the normative
`Segment.Tags.Tag.SimpleTag` placement. AVI uses `LIST/INFO/AIGC`, while FLV
uses `script.onMetaData.AIGC`. The non-ISOBMFF formats are remuxed with stream
copy for removal.

Remove a supported visible video mark:

```bash
remove-ai-watermarks video visible input.mp4 -o clean.mp4
remove-ai-watermarks video visible veo.mp4 --mark veo -o veo_clean.mp4
remove-ai-watermarks video visible seedance.mp4 --mark seedance -o seedance_clean.mp4
remove-ai-watermarks video visible dola.mp4 --mark dola -o dola_clean.mp4
remove-ai-watermarks video visible hailuo.mp4 --mark hailuo -o hailuo_clean.mp4
remove-ai-watermarks video visible kling.mp4 --mark kling -o kling_clean.mp4
```

This path scans the complete sequence before changing pixels. It accepts only a
mark that repeats at a stable position across adjacent frames, then reuses the
same OpenCV, MI-GAN, or LaMa fill backends as image removal. Audio is copied
without re-encoding and is allowed to reach its natural end; the video stream
is transcoded because its pixels change. The default `--mark auto` scans all
providers in one decode pass and selects the first stable match in the
specificity order shown below. Pass an explicit mark to restrict detection to
one provider.
Sora covers the moving Sora 2 mascot and wordmark. Veo covers both the current
four-point diamond and the legacy `Veo` text. Seedance covers the fixed boxed
`AI` label, Dola covers the fixed `Dola AI` text, Hailuo covers the composite
`MINIMAX | hailuo AI` label, and Kling covers the bottom-right `KLING AI`
label with its version suffix. A completed encode is published atomically. No
output is written when no stable mark is found.

Generate a video SynthID candidate:

```bash
uv tool install --force "remove-ai-watermarks[gpu]"
remove-ai-watermarks video invisible input.mp4 -o candidate.mp4
```

This path regenerates the complete sequence with one latent-noise field shared
across time, copies complete audio, strips source metadata, and publishes the
completed encode atomically. It cannot verify
Google's proprietary pixel watermark locally. The command therefore labels
every output `UNVERIFIED` and prints the exact Gemini Flash verification
prompt.

For invisible watermark removal, install the diffusion dependencies:

```bash
uv tool install --force "remove-ai-watermarks[gpu]"
remove-ai-watermarks invisible image.png -o clean.png
```

If the local detectors cannot confirm an invisible watermark but you know the
image came from an AI generator, add `--force`:

```bash
remove-ai-watermarks invisible image.png -o clean.png --force
```

See the [installation guide](docs/installation.md) for Homebrew, uv, optional
features, and development setup.

## Examples

### Visible Gemini mark

| Before | After |
| --- | --- |
| ![Image with a visible Gemini watermark](demo_banana_before.png) | ![Image after visible watermark removal](demo_banana_after.png) |

### High quality invisible removal

The `qwen-zimage` profile is the highest fidelity option for face heavy images.
It is CUDA only and uses a much larger model stack than the default ControlNet
profile.

```bash
uv tool install --force "remove-ai-watermarks[qwen-zimage]"
remove-ai-watermarks invisible image.png -o clean.png \
  --pipeline qwen-zimage --force
```

| OpenAI example before | OpenAI example after |
| --- | --- |
| [![OpenAI portrait grid before qwen-zimage](data/synthid/originals/ChatGPT%20Image%20May%2030,%202026,%2010_31_08%20AM.png)](data/synthid/originals/ChatGPT%20Image%20May%2030,%202026,%2010_31_08%20AM.png) | [![OpenAI portrait grid after qwen-zimage](docs/images/qwen-zimage/ChatGPT/ChatGPT%20Image%20May%2030,%202026,%2010_31_08%20AM_full_clean.png)](docs/images/qwen-zimage/ChatGPT/ChatGPT%20Image%20May%2030,%202026,%2010_31_08%20AM_full_clean.png) |

| Gemini example before | Gemini example after |
| --- | --- |
| [![Gemini sign before qwen-zimage](data/synthid/originals/Gemini_Generated_Image_633uuy633uuy633u.png)](data/synthid/originals/Gemini_Generated_Image_633uuy633uuy633u.png) | [![Gemini sign after qwen-zimage](docs/images/qwen-zimage/Gemini/Gemini_Generated_Image_633uuy633uuy633u_full_clean.png)](docs/images/qwen-zimage/Gemini/Gemini_Generated_Image_633uuy633uuy633u_full_clean.png) |

These exact output files were checked with the matching provider verifiers. That
result applies to these files, not to every seed, image, or future watermark
version.

## Common recipes

### Remove every detected visible mark

```bash
remove-ai-watermarks visible image.png -o clean.png
```

The default `--mark auto` checks all registered visible marks and removes every
match. If the mark is visible to you but the detector misses it, select its
region explicitly:

```bash
remove-ai-watermarks erase image.png \
  --region 1640,1930,400,100 \
  -o clean.png
```

`--region` uses `x,y,width,height` and may be repeated.

### Use a learned fill backend

The core install uses OpenCV inpainting when no learned backend is installed.
For more difficult backgrounds:

```bash
uv tool install --force "remove-ai-watermarks[migan]"
remove-ai-watermarks visible image.png -o clean.png --backend migan
```

```bash
uv tool install --force "remove-ai-watermarks[lama]"
remove-ai-watermarks visible image.png -o clean.png --backend lama
```

### Reduce CUDA memory use

```bash
remove-ai-watermarks invisible image.png -o clean.png \
  --cpu-offload --force
```

CPU offload lowers CUDA memory pressure by moving model components between CPU
and GPU. It is slower and has no effect on CPU or MPS.

### Process a directory

```bash
remove-ai-watermarks batch ./images --mode visible
remove-ai-watermarks batch ./images --mode all
```

## What the tool can recognize

Visible mark support includes:

- Google Gemini and Nano Banana sparkle;
- Doubao, Jimeng, Qwen, Kling, Baidu, LibLibAI, and RunningHub labels;
- one calibrated Samsung Galaxy AI label variant.

Metadata and provenance inspection covers C2PA, EXIF, XMP, IPTC, common
generator parameters, China TC260 AIGC labels, and several vendor specific
signals. Optional decoders add support for open DWT-DCT watermarks and Adobe
TrustMark.

The exact support matrix, including important locale and detector limits, lives
in [supported signals](docs/supported-signals.md).

## How it works

Visible removal follows three steps:

1. Detect a registered mark in its expected area.
2. Build a mask around the mark.
3. Fill only the masked region with OpenCV, MI-GAN, or LaMa.

Metadata removal uses format aware stripping. JPEG metadata removal preserves
the encoded image scan instead of recompressing it. Native MP4/MOV TC260 values
are blanked without changing box sizes or media offsets. Other supported
containers use their corresponding metadata path.

Invisible removal is different. It regenerates the image through a diffusion
pipeline to disrupt pixel and frequency domain watermarks. This changes the
image and cannot guarantee that a proprietary verifier will reject every
output.

See [supported signals](docs/supported-signals.md) and
[known limitations](docs/known-limitations.md) for the full technical boundary.

## Python API

```python
import remove_ai_watermarks as raiw

result, removed = raiw.remove_visible("watermarked.png", "clean.png")
print(removed)

report = raiw.inspect_video_metadata("input.mp4")
cleaned = raiw.remove_video_metadata("input.mp4")
candidate = raiw.remove_video_invisible("input.mp4", "candidate.mp4")
visible = raiw.remove_video_visible("input.mp4", "clean.mp4")
print(visible.mark)
veo = raiw.remove_video_visible("veo.mp4", "veo_clean.mp4", mark="veo")
seedance = raiw.remove_video_visible(
    "seedance.mp4",
    "seedance_clean.mp4",
    mark="seedance",
)
dola = raiw.remove_video_visible("dola.mp4", "dola_clean.mp4", mark="dola")
```

The high level API accepts a file path or a BGR NumPy array. For path inputs it
also reads provenance metadata, preserves alpha, and can strip AI metadata from
the written result.

See the [Python API guide](docs/python-api.md) for visible removal, provenance
inspection, metadata stripping, and diffusion usage.

## ComfyUI

The separate
[ComfyUI Remove AI Watermarks](https://github.com/wiltodelta/ComfyUI-remove-ai-watermarks)
package provides nodes for visible removal, detection, region erasing, and
invisible removal.

## Important limitations

- A missing local signal means unknown, not clean. Proprietary pixel
  watermarks may remain after metadata has been stripped.
- Visible removal reconstructs a small region. Results depend on the background
  and selected fill backend.
- Invisible removal changes the whole image and may alter faces, text, or fine
  detail.
- Visible video removal recognizes the moving Sora 2 wordmark, the current Veo
  diamond plus legacy `Veo` text, the Seedance boxed `AI` label, and the fixed
  Dola, Hailuo, and Kling labels. It does not recognize the older Sora Turbo
  corner swirl or unregistered layouts from those providers.
  The classical OpenCV backend can smear structured backgrounds; use MI-GAN or
  LaMa when recovery quality matters.
- Video SynthID regeneration changes resolution, frame rate, and image detail.
  It produces a candidate, not a locally verified clean video. The matching
  Google verifier is still required for every important output.
- MP4/MOV/M4V metadata stripping currently reads the full container into
  memory, so very large videos are not an intended metadata input yet.
- `qwen-zimage` requires CUDA. The other diffusion profiles also support the
  devices listed by `remove-ai-watermarks invisible --help`.
- Provider watermark systems can change. Validate important outputs with the
  provider's own verifier when one is available.

The shipped `video invisible` command uses the candidate-producing side of the
oracle-gated workflow. The companion `scripts/video_synthid_sweep.py` harness
builds a matched re-encode control plus VAE-regenerated candidates and leaves
the verifier verdict blank:

```bash
uv run --extra gpu python scripts/video_synthid_sweep.py input.mp4 -o sweep/
```

The control must still be SynthID-positive before a negative candidate can
count as removal evidence. In the 2026-07-29 calibration, matched controls from
two public Veo videos remained positive and the default stronger candidate was
negative on both. A weaker candidate was negative on only one, demonstrating
that the removal threshold is content dependent. This is calibration evidence,
not a universal guarantee for new videos or future verifier versions.

## Documentation

Start with the [documentation index](docs/index.md).

- [Installation](docs/installation.md)
- [CLI guide](docs/cli.md)
- [Python API](docs/python-api.md)
- [Supported signals](docs/supported-signals.md)
- [Known limitations](docs/known-limitations.md)
- [Scope, safety, and legal notes](docs/legal-and-safety.md)
- [Module internals](docs/module-internals.md)
- [Release and distribution](docs/release-and-distribution.md)

Research notes and historical experiments are listed separately in the
[documentation index](docs/index.md). They explain past decisions but do not
define the current public API.

## Contributing

Install the development environment and run the project gate:

```bash
uv sync --frozen --extra dev
bash maintain.sh
```

See [module internals](docs/module-internals.md) before changing a subsystem
with documented invariants.

## License

[Apache 2.0](LICENSE). Copyright 2025-2026 wiltodelta.
