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
