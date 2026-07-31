# CLI guide

The command line interface is organized around the type of work you want to do.

```text
remove-ai-watermarks [OPTIONS] COMMAND [ARGS]
```

Run `remove-ai-watermarks COMMAND --help` for the complete option list and
defaults. This page focuses on choosing the right command.

## Command dependency map

| Command or signal | Required installation |
| --- | --- |
| `metadata` and metadata-only `identify` | Default package |
| Visible signals in `identify` | `remove-ai-watermarks[visible]` (`pixels` is the minimal runtime) |
| Open DWT-DCT signals in `identify` | `remove-ai-watermarks[detect]` |
| Adobe TrustMark signals in `identify` | `remove-ai-watermarks[trustmark]` |
| `visible` and `erase` with OpenCV | `remove-ai-watermarks[visible]` (`pixels` is the minimal runtime) |
| `visible` or `erase` with MI-GAN | `remove-ai-watermarks[migan]` |
| `visible` or `erase` with big-LaMa | `remove-ai-watermarks[lama]` |
| `invisible` | `remove-ai-watermarks[diffusion]` |
| `invisible --pipeline qwen-zimage` | `remove-ai-watermarks[qwen-zimage]` |
| HEIC/HEIF/AVIF pixel input | Add `remove-ai-watermarks[heif]` |
| Every production command and backend | `remove-ai-watermarks[all]` |

`batch` requires the same extra as its selected mode. Extras can be combined in
one installation, for example `remove-ai-watermarks[visible,detect,heif]`.

## Inspect an image

```bash
remove-ai-watermarks identify image.png
```

`identify` always inspects supported metadata. When pixel extras are installed,
it also evaluates supported visible and invisible pixel signals. When no signal
is found, it reports the origin as unknown. It does not claim the image is
clean.

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

Install `remove-ai-watermarks[visible]` before using `visible` or `erase`.

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

## Remove invisible watermarks

Install the diffusion dependencies first:

```bash
uv tool install --force "remove-ai-watermarks[diffusion]"
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

The `all` command and the `all` installation extra are separate concepts. The
command runs every applicable stage. Installing `remove-ai-watermarks[all]`
makes every production backend available; a smaller installation such as
`remove-ai-watermarks[visible,diffusion]` can also run the command with fewer
optional backends.

```bash
remove-ai-watermarks all image.png -o clean.png
```

The command runs:

1. visible mark removal;
2. invisible watermark removal when available and applicable;
3. AI metadata stripping.

The visible options and diffusion options are also available on `all`.

If diffusion is required but the `diffusion` extra is unavailable, `all` still
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
