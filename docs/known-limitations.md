# Known limitations

This page describes current product limits. Historical measurements and
superseded experiments live in the research archive listed in
[the documentation index](index.md).

## Visible removal

### Fill quality depends on the background

Visible removal changes only the selected mask, but the hidden pixels still
have to be reconstructed.

- OpenCV is fast and dependency free. It works well on flat backgrounds but
  can smear texture or repeated structure.
- MI-GAN is a lighter learned backend. It can improve natural texture but may
  ghost or invent structure.
- LaMa is the heaviest learned backend and is generally the strongest option
  for difficult backgrounds.

`--backend auto` selects LaMa when available, then MI-GAN, then OpenCV.

No backend can recover detail that is completely hidden by an opaque mark. A
successful detection therefore does not guarantee a visually perfect fill.

### Automatic detection covers registered variants only

The registry contains vendor and locale specific templates. A redesigned mark,
an unsupported locale, a different position, or a crop may be missed.

Known examples:

- Samsung detection is calibrated for the Italian
  `Contenuti generati dall'AI` text variant.
- The Jimeng top-left pill has a weak visual detector and is intentionally
  subject to additional product and background checks.
- Kling support covers the calibrated variants rather than every Kling label.

Use `erase --region` when you can see and select an unsupported or missed mark.

### Strict and automatic sensitivity trade recall for precision

`--sensitivity strict` uses the visual gate alone. The default `auto` mode may
relax a mark only when metadata or a confidently detected sibling mark
corroborates the same product.

There is no blanket "this image is AI" relaxation. That information does not
identify the vendor, mark, or location and caused unacceptable false
detections in the removed experimental mode.

## Invisible removal

### Regeneration is lossy

Invisible removal does not decode and delete a payload. It regenerates the
image through a diffusion pipeline. Faces, text, colors, and fine detail can
change even when the watermark is successfully disrupted.

ControlNet is the default compatibility profile. It conditions on edges to
preserve structure, but edges do not preserve identity or exact texture.

The CUDA only `qwen-zimage` profile adds a separate face stage and is the
highest fidelity option in the current implementation. It is larger, slower,
and still may alter small text or difficult faces.

### Removal cannot be verified locally for proprietary SynthID

The project has no public local SynthID pixel decoder. It can infer likely
presence from supported provenance metadata, but a missing metadata proxy is
not a negative pixel verdict.

For important outputs:

1. preserve the original;
2. process a copy;
3. verify with the matching provider tool when available;
4. do not assume one provider's verifier covers another provider's payload.

Provider systems can change, so a result verified on one file, seed, or version
is not a permanent certification.

### Strength is content and seed dependent

For SDXL and ControlNet, the CLI resolves an unset strength from the detected
vendor:

- OpenAI: `0.10`;
- Google: `0.15`;
- unknown: `0.15`.

An explicit `--strength` overrides these defaults. The defaults are operating
points, not universal guarantees. Near a removal threshold, different content
or a different random seed may change the verifier result.

The base Qwen and `qwen-zimage` profiles have profile specific strength
behavior. Consult `remove-ai-watermarks invisible --help` and the source of
[`watermark_profiles.py`](../src/remove_ai_watermarks/noai/watermark_profiles.py)
for the current resolver.

### Pipelines have different quality tradeoffs

| Pipeline | Main limit |
| --- | --- |
| `controlnet` | Edge conditioning can preserve a watermark carrying region too closely, and faces may drift. |
| `sdxl` | Flat graphics and precise structure may receive too little or unhelpful change. |
| `qwen` | Large CUDA oriented model; face smoothing can still be significant. |
| `qwen-zimage` | CUDA only, large model stack, and limited broad certification across seeds and content. |

The legacy `default` profile name maps to `sdxl`. The `--auto` flag is
deprecated, emits a warning, and changes nothing.

## Resolution and memory

### Small images are enlarged before SDXL based diffusion

The SDXL, ControlNet, and base Qwen paths use a default minimum long side of
`1024`. Smaller inputs are enlarged before diffusion and restored to their
original dimensions afterward. Set `--min-resolution 0` to disable the floor.

`qwen-zimage` does not apply this SDXL minimum resolution floor.

### Large images stay at native resolution unless capped

`--max-resolution 0` means no explicit downscale cap. A positive value caps the
long side before diffusion and restores the result afterward. This reduces
memory use but introduces a downscale and upscale round trip.

`--tile` preserves the input dimensions while running the diffusion stage in
overlapping tiles. It avoids the explicit downscale, but it is not pixel
lossless: each tile is independently regenerated. With `qwen-zimage`, only the
global Qwen stage is tiled; the face stage runs after tile blending.

### CPU offload is CUDA only

`--cpu-offload` moves Diffusers model components between CPU and CUDA instead of
keeping the complete standard pipeline in GPU memory. For `qwen-zimage`, it
forces the face stack to use its offload path.

The option reduces CUDA memory pressure at the cost of speed. It has no effect
on CPU or MPS and fails loudly when a CUDA Diffusers pipeline does not expose
the required offload method.

### MPS may fall back to CPU

The SDXL paths include an MPS out-of-memory fallback that reloads on CPU. A run
that appears much slower after an MPS failure may be continuing on CPU.

Memory needs depend on the pipeline, input size, dtype, and machine. Use tiling,
a resolution cap, or a lighter pipeline when necessary.

## Metadata and formats

### Missing metadata does not mean clean

Screenshots, social platforms, and re-encoding can remove metadata while a
pixel watermark remains. `identify` therefore reports unknown rather than
clean when no supported signal is found.

### JPEG XL is metadata only

The metadata path recognizes JPEG XL containers, but the visible and diffusion
image paths do not list `.jxl` as a supported pixel format because the package
does not include a JPEG XL pixel decoder.

### HEIC, HEIF, and AVIF use a Pillow fallback

OpenCV does not decode these formats in the project. `image_io.imread` falls
back to Pillow with `pillow-heif`. A corrupt or truncated file may still fail to
decode.

### Some metadata removal requires ffmpeg

WebM, Matroska, MP3, WAV, FLAC, OGG, Opus, and AAC container metadata is stripped
through ffmpeg with stream copying. The operation fails if ffmpeg is absent or
cannot parse the input.

### Visible video removal is provider-specific and still experimental

The experimental `video metadata` command and high level video API inspect and
strip supported AI provenance metadata without transcoding streams.

`video visible` and `remove_video_visible` additionally support the moving
Sora 2 mascot and wordmark, the current Veo four-point diamond, the legacy
`Veo` text, the Seedance boxed `AI` label, and the fixed `Dola AI` text.
Detection requires a recurring visual candidate across adjacent frames.
Seedance, Dola, and Veo candidates must remain anchored rather than drifting
with a scene object. Provider provenance can recover low-contrast runs only
after visual evidence exists, so metadata alone does not erase a clean API
export.
Historical Sora Turbo exports use a small OpenAI swirl in the corner rather
than the moving mascot-and-wordmark design; that earlier variant is not
detected by the `sora` video mark. Other provider video labels and proprietary
invisible video watermarks are not supported yet.

Visible removal transcodes the video stream and copies audio. Its frame-local
fill is not a motion-aware video inpainting model. OpenCV can leave a visible
smear where the mark overlaps a hard edge or structured texture, and the smear
can vary over time. MI-GAN and LaMa improve individual frames but do not
guarantee temporal coherence. The Veo diamond uses a shape mask to limit damage
outside the symbol. Seedance fills the full localized box because a synthetic
outline mask left part of the real translucent border visible in an end-to-end
check. OpenCV may therefore soften texture inside that small box; use MI-GAN or
LaMa when reconstruction quality matters. The current encoder also emits a
constant-frame-rate output at the decoded stream rate, so variable-frame-rate
preservation is not yet guaranteed.

Native TC260 metadata in MP4/MOV is supported at its normative
`moov.udta.meta.keys/ilst` placement, including non-faststart files whose
`moov` follows a large media payload. MKV/WebM is supported at the normative
`Segment.Tags.Tag.SimpleTag` placement and uses ffmpeg for stream-copy removal.
The corresponding FLV and AVI native tags are not parsed yet.

The current ISOBMFF stripper reads an MP4, MOV, or M4V container into memory
before rewriting its metadata boxes. A streaming box copier is required before
the experimental command is appropriate for very large video files.

### Metadata transformation is fail safe

`remove_ai_metadata` may copy an undecodable file through unchanged instead of
raising. User facing callers must use `strip_and_verify` and inspect its
surviving marker mapping before reporting success. `strip_and_verify` recovers
when `image_io` can still decode the raster by normalizing the container and
checking again. A truly undecodable file still reports the surviving markers.
The CLI uses this verified path.

### Sixteen bit PNG output is not preserved

The Pillow based PNG metadata rewrite uses the normal image save path and may
reduce a sixteen bit PNG to eight bits. A byte-level PNG metadata stripper
would be required to preserve that bit depth.

## Detection extras

The `detect` extra decodes an open DWT-DCT watermark used in some Stable
Diffusion, SDXL, and FLUX workflows. That decoder is sensitive to the carrier
and transformations. A negative result is not a universal negative.

The `trustmark` extra adds Adobe TrustMark decoding. The implementation retains
an additional JPEG re-encode gate because isolated decoder hits can otherwise
be content noise.

External AI versus real image classifiers are out of scope. The project
identifies concrete local provenance signals instead of shipping a generic
statistical classifier.

## Output and traceability

Removing file-local signals does not remove:

- provider account history;
- server side copies or provenance stores;
- perceptual fingerprints;
- evidence that an image passed through a removal pipeline;
- legal disclosure duties.

See [scope, safety, and legal notes](legal-and-safety.md).
