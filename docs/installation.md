# Installation

Python 3.10.1 or newer is required.

## Core install

The core package provides:

- provenance inspection;
- visible watermark removal with OpenCV;
- manual region erasing with OpenCV;
- AI metadata inspection and removal.

Install it as an isolated command with uv:

```bash
uv tool install remove-ai-watermarks
```

Or with pipx:

```bash
pipx install remove-ai-watermarks
```

You can also install the Homebrew package on macOS or Linux:

```bash
brew install wiltodelta/tap/remove-ai-watermarks
```

## Invisible watermark removal

Diffusion based removal needs the `gpu` extra:

```bash
uv tool install --force "remove-ai-watermarks[gpu]"
```

The code supports CUDA, XPU, MPS, and CPU devices. A GPU is recommended because
CPU inference is slow.

For the CUDA only Qwen Image plus Z-Image profile:

```bash
uv tool install --force "remove-ai-watermarks[qwen-zimage]"
```

The `qwen-zimage` extra includes the normal `gpu` dependencies.

## Optional features

Install only what you need:

| Extra | Adds |
| --- | --- |
| `migan` | MI-GAN ONNX fill backend |
| `lama` | big-LaMa ONNX fill backend |
| `detect` | Open DWT-DCT watermark decoder used by `identify` |
| `trustmark` | Adobe TrustMark decoder |
| `esrgan` | Real-ESRGAN upscaling before diffusion |
| `qwen-zimage` | CUDA only Qwen Image plus Z-Image pipeline |

Example:

```bash
uv tool install --force "remove-ai-watermarks[migan,detect]"
```

Some optional models download their weights on first use.

## Install from the repository

```bash
git clone https://github.com/wiltodelta/remove-ai-watermarks.git
cd remove-ai-watermarks
uv sync --frozen
```

Add the feature groups required for your work:

```bash
uv sync --frozen --extra dev
uv sync --frozen --extra dev --extra gpu
```

Run commands from the repository root:

```bash
uv run remove-ai-watermarks --help
```

## Development setup

Install development dependencies:

```bash
uv sync --frozen --extra dev
```

Run the complete project gate:

```bash
bash maintain.sh
```

The script runs dependency checks, linting, formatting checks, type checking,
and the test suite.

## Hugging Face authentication

Pass a Hugging Face token directly when the selected model or account requires
one:

```bash
remove-ai-watermarks invisible image.png --hf-token "$HF_TOKEN"
```

The CLI also loads `HF_TOKEN` from the environment and from a local `.env`
file. The same name is documented in `.env.example`.

## Troubleshooting

### The first model run is slow

Diffusion and learned fill backends may download model weights on first use.
Later runs reuse their caches.

### The command skips invisible removal

The normal behavior is to skip diffusion when no supported local signal is
found. A missing signal does not prove that the image is clean. If you know the
image came from a relevant generator, use `--force`.

If the CLI reports that diffusion dependencies are unavailable, install the
`gpu` extra.
