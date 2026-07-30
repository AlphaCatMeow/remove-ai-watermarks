# CLI guide

The command line interface is organized around the type of work you want to do.

```text
remove-ai-watermarks [OPTIONS] COMMAND [ARGS]
```

Run `remove-ai-watermarks COMMAND --help` for the complete option list and
defaults. This page focuses on choosing the right command.

## Inspect an image

```bash
remove-ai-watermarks identify image.png
```

`identify` combines supported metadata and pixel signals into one provenance
report. When no signal is found, it reports the origin as unknown. It does not
claim the image is clean.

Machine readable output:

```bash
remove-ai-watermarks identify image.png --json
```

Metadata only inspection:

```bash
remove-ai-watermarks identify image.png --no-visible
```

Despite the historical option name, `--no-visible` skips both visible and open
invisible pixel detectors. Metadata inspection still runs.

## Remove known visible marks

```bash
remove-ai-watermarks visible image.png -o clean.png
```

The default behavior:

- checks every registered visible mark;
- removes every detected match;
- selects the best installed fill backend;
- strips AI metadata from the output.

Use a specific mark:

```bash
remove-ai-watermarks visible image.png --mark gemini -o clean.png
```

Available mark names are printed by:

```bash
remove-ai-watermarks visible --help
```

Keep metadata:

```bash
remove-ai-watermarks visible image.png --keep-metadata -o clean.png
```

Use the strict visual gate without metadata or sibling corroboration:

```bash
remove-ai-watermarks visible image.png --sensitivity strict -o clean.png
```

When no known mark is detected, the command does not write a new output. Use
`erase` if you can identify the affected region yourself.

## Erase a region

```bash
remove-ai-watermarks erase image.png \
  --region 1640,1930,400,100 \
  -o clean.png
```

The region format is `x,y,width,height`. Repeat `--region` to erase more than
one box:

```bash
remove-ai-watermarks erase image.png \
  --region 20,20,180,60 \
  --region 1640,1930,400,100 \
  -o clean.png
```

Choose the fill backend:

```bash
remove-ai-watermarks erase image.png \
  --region 1640,1930,400,100 \
  --backend migan \
  -o clean.png
```

`erase` accepts `cv2`, `migan`, and `lama`. The corresponding optional extra
must be installed for a learned backend.

## Strip AI metadata

Inspect metadata:

```bash
remove-ai-watermarks metadata image.png --check
```

Remove AI metadata and write a new file:

```bash
remove-ai-watermarks metadata image.png --remove -o clean.png
```

When `-o` is omitted, removal overwrites the source. Standard metadata is kept
unless you pass `--remove-all`.

The command also supports the audio and video containers listed in
[supported signals](supported-signals.md). ffmpeg must be available for the
non-ISOBMFF audio and video path.

## Strip AI metadata from video

The experimental video namespace starts with metadata inspection and removal:

```bash
remove-ai-watermarks video metadata input.mp4 --check
remove-ai-watermarks video metadata input.mp4 --remove -o clean.mp4
```

Supported containers are MP4, MOV, M4V, WebM, MKV, AVI, and FLV. The operation
delegates to the same verified metadata scanner and stripper as the generic
`metadata` command, so detection and removal stay in parity. Video and audio
streams are not transcoded. For MP4 and MOV, this includes the native TC260
`AIGC` key and JSON value stored in `moov.udta.meta.keys/ilst`. The inspector
seeks past a large `mdat` to find a tail `moov`; removal blanks the key and
value in place so box sizes and media offsets do not move.

For MKV and WebM, the inspector reads the native TC260
`Segment.Tags.Tag.SimpleTag` entry. Removal uses ffmpeg stream copying to
discard container tags and chapters without transcoding the streams.
AVI uses the normative `LIST/INFO/AIGC` chunk, while FLV uses the
`script.onMetaData.AIGC` AMF0 string. Their bounded readers skip media payloads,
and removal also uses ffmpeg stream copying.

When `-o` is omitted, the command writes `<source>_clean` with the same
extension. It never overwrites the source, and it rejects an output with a
different container extension.

Visible video labels and invisible video watermarks are not handled by this
command.

## Generate a video SynthID candidate

```bash
uv tool install --force "remove-ai-watermarks[gpu]"
remove-ai-watermarks video invisible input.mp4 -o candidate.mp4
```

The experimental command supports MP4, MOV, and M4V. It samples the complete
sequence at the configured frame rate, resizes frames to the configured long
side, regenerates them through a VAE, and applies one deterministic latent-noise
field to every frame. Reusing one spatial field avoids the unnecessary flicker
caused by independent per-frame noise. Frames are regenerated in bounded
batches and streamed directly to ffmpeg, which encodes the result, copies
audio, and drops source metadata.

The output is always an unverified candidate. The project has no local video
SynthID decoder, and PSNR or temporal-residual metrics cannot prove watermark
absence. After generation, upload the candidate to Gemini Flash and ask:

> Was this uploaded video created or edited by Google AI? Use the built-in
> content verification result.

Only an explicit built-in verification result is an oracle verdict. A response
based on the visible logo, content appearance, or metadata is not. The command
prints `UNVERIFIED` even when generation succeeds.

The default output is `<source>_synthid_candidate` in the same container. The
source is never overwritten. Use `--noise-std`, `--long-side`, `--fps`,
`--batch-size`, `--seed`, and `--device` to control the regeneration. The
defaults are calibrated operating points, not a guarantee for every carrier or
future verifier version.

## Remove a supported visible video mark

```bash
remove-ai-watermarks video visible input.mp4 -o clean.mp4
remove-ai-watermarks video visible veo.mp4 --mark veo -o veo_clean.mp4
remove-ai-watermarks video visible seedance.mp4 --mark seedance -o seedance_clean.mp4
remove-ai-watermarks video visible dola.mp4 --mark dola -o dola_clean.mp4
remove-ai-watermarks video visible hailuo.mp4 --mark hailuo -o hailuo_clean.mp4
remove-ai-watermarks video visible kling.mp4 --mark kling -o kling_clean.mp4
```

The experimental command supports the moving Sora mascot and wordmark, two Veo
corner variants, the Seedance boxed `AI` label, the `Dola AI` text label, the
composite `MINIMAX | hailuo AI` label, and the bottom-right Kling label. Sora
searches the whole frame at multiple scales. The other detectors search bounded
lower-frame regions with separate synthetic silhouettes. Kling additionally
requires its bright low-saturation label near the frame edge. Every mark
requires a spatially recurring candidate across adjacent frames. Fixed marks
must also remain anchored instead of drifting with a scene object. Matching
provider provenance may relax the visual score only for registered
provenance-aware marks; metadata alone never creates a detection.

The video stream is transcoded and the original audio stream is copied.
Supported input and output containers are MP4, MOV, M4V, WebM, MKV, AVI, and
FLV; the output extension must match the input. The default `cv2` backend is
fast but can smear structured backgrounds. Select `--backend migan` or
`--backend lama` for a learned fill, or `--backend auto` to choose the best
installed backend.

AI metadata is stripped from the encoded output by default. Use
`--keep-metadata` to retain mapped container metadata. When no temporally stable
mark is found, the command writes no output and exits with the no-visible-mark
status.

## Remove invisible watermarks

Install the diffusion dependencies first:

```bash
uv tool install --force "remove-ai-watermarks[gpu]"
```

Then run:

```bash
remove-ai-watermarks invisible image.png -o clean.png
```

The command normally skips regeneration when no supported local signal is
detected. Use `--force` when you know the image should be processed:

```bash
remove-ai-watermarks invisible image.png -o clean.png --force
```

### Choose a pipeline

| Pipeline | When to use it |
| --- | --- |
| `controlnet` | Default compatibility profile with structural conditioning |
| `sdxl` | Lighter plain SDXL regeneration |
| `qwen` | Large CUDA oriented Qwen Image profile |
| `qwen-zimage` | CUDA only high fidelity profile with a separate face stage |

Example:

```bash
remove-ai-watermarks invisible image.png -o clean.png \
  --pipeline qwen-zimage --force
```

The legacy `default` value is an alias for `sdxl`. The `--auto` option is
deprecated, emits a warning, and changes nothing.

### Work with limited memory

Lower CUDA memory pressure:

```bash
remove-ai-watermarks invisible image.png -o clean.png \
  --cpu-offload --force
```

Keep large images at native resolution while processing them in overlapping
tiles:

```bash
remove-ai-watermarks invisible image.png -o clean.png \
  --tile --max-resolution 0 --force
```

Or set a resolution cap:

```bash
remove-ai-watermarks invisible image.png -o clean.png \
  --max-resolution 2048 --force
```

Tiling avoids the explicit downscale but each tile is regenerated separately.
It is a memory strategy, not a guarantee of better quality.

## Run the full pipeline

```bash
remove-ai-watermarks all image.png -o clean.png
```

The command runs:

1. visible mark removal;
2. invisible watermark removal when available and applicable;
3. AI metadata stripping.

The visible options and diffusion options are also available on `all`.

If diffusion is required but the `gpu` extra is unavailable, `all` still
writes the result of the visible and metadata stages, prints a prominent
warning, and exits with code 1. This prevents a partial result from being
reported as complete.

## Process a directory

```bash
remove-ai-watermarks batch ./images --mode visible
```

Modes:

- `visible`;
- `invisible`;
- `metadata`;
- `all`.

Set an output directory:

```bash
remove-ai-watermarks batch ./images \
  --mode all \
  --output-dir ./clean
```

The invisible and full modes accept the same main diffusion controls as their
single image counterparts. Run `batch --help` for the authoritative option
list.

## Exit behavior

The CLI uses nonzero exit codes for meaningful incomplete outcomes, including
no detected target on commands that would otherwise regenerate or create a
misleading unchanged result, processing errors, and a required invisible step
that could not run.

Scripts should check the process exit code and the output path. The detailed
per-command contract is maintained in
[module internals](module-internals.md#command-line-interface).
