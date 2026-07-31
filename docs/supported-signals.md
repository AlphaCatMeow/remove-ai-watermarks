# Supported signals

This page describes the current support boundary. A check mark means that the
repository contains a corresponding code path. It does not guarantee detection
or removal on every future vendor version.

## Visible marks

The `visible` command registers these mark keys:

| Key | Mark | Expected area | Important limit |
| --- | --- | --- | --- |
| `gemini` | Google Gemini sparkle | Usually bottom right | Detection includes a false positive gate. |
| `doubao` | `豆包AI生成` | Bottom right | Vendor specific text detector. |
| `jimeng` | `★ 即梦AI` | Bottom right | Vendor specific text detector. |
| `qwen` | `千问AI生成` | Bottom right | Strict visual gate. |
| `kling` | `可灵AI 3.0` | Bottom right | Only calibrated variants are covered. |
| `yuanbao` | `元宝` over `AI生成` | Bottom right | Standard two-line variant only. |
| `samsung` | `✦ Contenuti generati dall'AI` | Bottom left | Calibrated for the Italian text variant. |
| `runninghub` | `RunningHub AI生成` | Top left | Strict visual and position gates. |
| `baidu` | `百度 AI生成` | Bottom right | Detector and extended removal footprint. |
| `liblib` | `LibLibAI` | Bottom center | Includes a minimum image size gate. |
| `jimeng_pill` | `AI生成` pill | Top left | Weak detector with additional product and background gates. |

`--mark auto` evaluates all registered marks and removes every selected match.
Known marks are localized to a mask, then the selected fill backend reconstructs
the masked area.

Marks from other vendors are not detected automatically. Use `erase --region`
when you can select the affected area yourself.

## Fill backends

| Backend | Install | Behavior |
| --- | --- | --- |
| `cv2` | `remove-ai-watermarks[visible]` | Classical OpenCV inpainting |
| `migan` | `remove-ai-watermarks[migan]` | MI-GAN through ONNX Runtime |
| `lama` | `remove-ai-watermarks[lama]` | big-LaMa through ONNX Runtime |
| `auto` | Depends on installed extras | Selects LaMa, then MI-GAN, then OpenCV |

The learned backends download model files on first use.

## Metadata and provenance

The inspection and stripping code handles signals in these groups:

- C2PA Content Credentials and supported cloud manifest references;
- EXIF and XMP generator fields;
- IPTC AI disclosure fields;
- PNG text chunks and embedded generation parameters;
- China TC260 AIGC labels in supported metadata placements;
- xAI and Grok EXIF signature fields;
- Samsung AI editing markers;
- Hugging Face job metadata;
- open Stable Diffusion style DWT-DCT watermarks with the `detect` extra;
- Adobe TrustMark with the `trustmark` extra.

`identify` combines detected signals into a `ProvenanceReport`. It reports
unknown when evidence is absent. It never treats missing metadata as proof that
an image is human made.

## File and container formats

Pixel based image commands discover these extensions:

- PNG;
- JPEG;
- WebP;
- HEIC and HEIF;
- AVIF.

HEIC, HEIF, and AVIF pixel decoding requires the independent `heif` extra in
addition to the selected pixel feature. Metadata scanning does not.

Metadata inspection and removal additionally have container paths for:

- JPEG XL metadata;
- MP4, MOV, M4V, and M4A;
- WebM, MKV, MKA, MP3, WAV, FLAC, OGG, OGA, Opus, and AAC when ffmpeg is
  available.

JPEG image metadata stripping removes targeted metadata segments without
re-encoding the entropy coded image scan. PNG and WebP removal preserves pixel
values through lossless output paths. HEIC, HEIF, AVIF, and other containers
use their format specific paths.

## Invisible watermarks

The `invisible` command uses diffusion regeneration. It targets watermark
patterns by changing the image rather than decoding and deleting a known
payload.

Current pipeline values:

- `controlnet`;
- `sdxl`;
- `qwen`;
- `qwen-zimage`;
- legacy alias `default`, which resolves to `sdxl`.

SynthID does not have a public local pixel decoder in this project. The tool can
infer likely presence from supported provenance metadata, but after that
metadata is removed a local negative result is inconclusive.

The optional `detect` extra is different: it provides a local decoder for the
open DWT-DCT watermark used by some Stable Diffusion, SDXL, and FLUX workflows.
That signal is carrier and transformation sensitive, so a negative is still
not a universal clean verdict.

## Provider overview

| Provider or family | Visible | Invisible path | Metadata or provenance |
| --- | --- | --- | --- |
| Google Gemini | Sparkle | Diffusion regeneration for SynthID | C2PA and related source signals |
| OpenAI image generators | None registered | Diffusion regeneration for supported invisible signals | C2PA and generator provenance |
| Stable Diffusion and SDXL | None registered | Diffusion regeneration; optional open decoder | Embedded parameters and text metadata |
| FLUX | None registered | Diffusion regeneration; optional open decoder | C2PA for supported sources |
| Adobe Firefly | None registered | No proprietary local decoder | C2PA; optional TrustMark decoder |
| Midjourney | None registered | No registered pixel decoder | EXIF, XMP, and IPTC signals |
| ByteDance generators | Doubao and Jimeng marks | No registered pixel decoder | TC260 AIGC and supported C2PA signals |
| Qwen | Qwen mark | No registered pixel decoder | TC260 AIGC |
| Kling | Kling mark | No registered pixel decoder | TC260 AIGC |
| Baidu | Baidu mark | No registered pixel decoder | TC260 AIGC |
| LibLibAI | LibLibAI mark | No registered pixel decoder | TC260 AIGC |
| RunningHub | RunningHub mark | No registered pixel decoder | TC260 AIGC |
| Samsung Galaxy AI | One locale specific mark | No registered pixel decoder | C2PA and Samsung markers |

For detector thresholds, measured limits, and incident history, see
[module internals](module-internals.md) and
[known limitations](known-limitations.md).
