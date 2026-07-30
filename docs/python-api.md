# Python API

Use the high level API for normal application integration. Low level detector
and pipeline modules are intended for maintainers and specialized workflows.

## Remove visible marks

```python
import remove_ai_watermarks as raiw

result, removed = raiw.remove_visible(
    "watermarked.png",
    "clean.png",
)
```

The function returns:

- the result as a BGR NumPy array;
- a list of labels that were removed.

An empty `removed` list means that no registered visible mark was selected. It
does not prove the image has no metadata or invisible watermark.

### Path input

For a path input, `remove_visible`:

- reads metadata provenance for the default `auto` sensitivity;
- preserves a separate alpha channel;
- writes the output when an output path is supplied;
- strips AI metadata from the written output by default;
- preserves the original bytes for a same-format no-op copy.

```python
result, removed = raiw.remove_visible(
    "watermarked.png",
    "clean.png",
    sensitivity="auto",
    backend="auto",
    strip_metadata=True,
)
```

Set `write_noop=False` if the output path must remain untouched when nothing is
removed:

```python
result, removed = raiw.remove_visible(
    "input.png",
    "clean.png",
    write_noop=False,
)
```

### Array input

Array inputs are BGR NumPy arrays. They do not carry file metadata or a separate
alpha plane:

```python
import cv2
import remove_ai_watermarks as raiw

image = cv2.imread("input.png")
result, removed = raiw.remove_visible(image, backend="cv2")
```

## Inspect provenance

Get the vendor keys used by visible removal:

```python
import remove_ai_watermarks as raiw

vendors = raiw.visible_provenance("input.png")
```

Get the full provenance report:

```python
from pathlib import Path

from remove_ai_watermarks.identify import identify

report = identify(Path("input.png"))
print(report.platform)
print(report.signals)
```

Use `check_visible=False` and `check_invisible=False` for metadata only
inspection:

```python
report = identify(
    Path("input.png"),
    check_visible=False,
    check_invisible=False,
)
```

## Strip metadata

```python
from pathlib import Path

from remove_ai_watermarks.metadata import has_ai_metadata, strip_and_verify

source = Path("input.png")
output = Path("clean.png")

if has_ai_metadata(source):
    output_path, surviving_markers = strip_and_verify(source, output)
    if surviving_markers:
        raise RuntimeError(
            f"AI metadata remains in {output_path}: {surviving_markers}"
        )
```

Use `strip_and_verify` when your application reports that stripping succeeded.
It checks the written output and returns `(output_path, surviving_markers)`.
When the first strip leaves markers in a malformed but raster-decodable image,
it normalizes the container through `image_io` and checks again. That recovery
path preserves the pixels but drops standard metadata. Treat a nonempty
`surviving_markers` mapping as a failure.

`remove_ai_metadata` is the lower level fail-safe transformer. It may copy an
undecodable input through unchanged, so its return alone must not be presented
as proof that metadata was removed.

## Inspect and strip video metadata

The experimental high level video API supports MP4, MOV, M4V, WebM, and MKV:

```python
import remove_ai_watermarks as raiw

report = raiw.inspect_video_metadata("input.mp4")
if report.has_ai_metadata:
    result = raiw.remove_video_metadata("input.mp4")
    if result.remaining:
        raise RuntimeError(f"AI metadata remains: {result.remaining}")
```

`remove_video_metadata` does not transcode video or audio streams. Its default
output is `input_clean.mp4`, leaving the source untouched. An explicit output
must use the same container extension as the source.

The returned `VideoMetadataResult` records the source, output, metadata detected
before removal, and any markers remaining after the verified strip. MP4/MOV
inspection recognizes the native TC260 `AIGC` entry in
`moov.udta.meta.keys/ilst`; its removal preserves container size and encoded
stream bytes. MKV/WebM inspection recognizes the corresponding
`Segment.Tags.Tag.SimpleTag` representation; its removal requires ffmpeg for a
stream-copy remux.

## Generate a video SynthID candidate

```python
import remove_ai_watermarks as raiw

result = raiw.remove_video_invisible(
    "input.mp4",
    "candidate.mp4",
    device="auto",
)
assert result.requires_external_verification
if result.remaining_metadata:
    raise RuntimeError(f"AI metadata remains: {result.remaining_metadata}")
```

`remove_video_invisible` supports MP4, MOV, and M4V. It regenerates the complete
video through a VAE in bounded batches, shares one seeded latent-noise field
across all frames, streams pixels to ffmpeg, copies audio, and strips source
metadata. The default output is
`input_synthid_candidate.mp4`; a distinct same-container output is required.

The returned `VideoInvisibleResult` includes output geometry, frame rate, frame
count, paired PSNR, and the motion-compensated temporal-residual ratio. Those
fields measure fidelity and flicker only. They are not a SynthID detector.
`requires_external_verification` is always true because Google does not publish
a local decoder for this video payload. Verify the candidate with Gemini
Flash's built-in content verification before treating it as watermark-negative.

## Remove a supported visible video mark

```python
import remove_ai_watermarks as raiw

result = raiw.remove_video_visible(
    "sora.mp4",
    "sora_clean.mp4",
    backend="cv2",
    strip_metadata=True,
)
if result.output is None:
    print("No temporally stable Sora mark was found")

veo_result = raiw.remove_video_visible(
    "veo.mp4",
    "veo_clean.mp4",
    mark="veo",
)
seedance_result = raiw.remove_video_visible(
    "seedance.mp4",
    "seedance_clean.mp4",
    mark="seedance",
)
dola_result = raiw.remove_video_visible(
    "dola.mp4",
    "dola_clean.mp4",
    mark="dola",
)
```

`remove_video_visible` scans the complete video before writing output. It
combines synthetic multi-scale visual matching with temporal consistency, so an
isolated lookalike in one frame is not enough to authorize inpainting. The
supported `mark` values are `sora`, `veo`, `seedance`, and `dola`. The Veo
detector recognizes the current four-point diamond and the legacy `Veo` text.
Seedance recognizes the boxed `AI` label, while Dola recognizes its compact
text label. Each variant has an independent synthetic silhouette and calibrated
temporal policy.

The returned `VideoVisibleResult` records the total, detected, and removed frame
counts plus any AI metadata that survived the output encode. The function
returns `output=None` and writes no file when no stable mark is selected. Video
pixels are transcoded through ffmpeg while the source audio stream is copied.

## Remove invisible watermarks

```python
from pathlib import Path

from remove_ai_watermarks.invisible_engine import InvisibleEngine

engine = InvisibleEngine(
    pipeline="controlnet",
    device=None,
    cpu_offload=False,
)

engine.remove_watermark(
    Path("watermarked.png"),
    Path("clean.png"),
)
```

`device=None` selects the device automatically. Supported explicit values are
defined by the CLI and runtime device resolver.

For limited CUDA memory:

```python
engine = InvisibleEngine(
    pipeline="controlnet",
    cpu_offload=True,
)
```

For the CUDA only high fidelity profile:

```python
engine = InvisibleEngine(pipeline="qwen-zimage")
```

The `qwen-zimage` extra must be installed for that profile.

The full `remove_watermark` signature includes strength, steps, guidance,
seeding, tiling, resolution, upscaling, and postprocessing controls. Read the
method signature in
[`invisible_engine.py`](../src/remove_ai_watermarks/invisible_engine.py) or use
the CLI guide for the concepts.
Defaults can differ between the Python method and CLI profile resolution, so
pass values explicitly when reproducibility matters.
