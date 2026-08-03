---
globs: ["src/**/*.py", "tests/**/*.py", "scripts/**/*.py", "pyproject.toml", "uv.lock", "maintain.sh", ".github/workflows/*.yml"]
description: Command contracts, project gate, typing boundaries, and model-adjacent test invariants.
---

# Development invariants

## Command contracts

Every single-image command declares `source` with `dir_okay=False`; `batch` declares its directory with `file_okay=False`. Keep `tests/test_cli_robustness.py::TestDirectoryInputIsRejected` as the regression guard.

Exit-code and no-signal behavior is a public contract. Read the command-line section of [`../../docs/module-internals.md`](../../docs/module-internals.md) before changing it.

Do not add an option whose only outcome is an error. Model id, step count and CFG are fixed by the profile, so none of them is a parameter of the CLI, `InvisibleEngine`, or `WatermarkRemover` -- they were accepted-then-rejected for a while, which moved the failure several frames below the caller and advertised choices the pinned stack cannot honor. If a value cannot vary, delete the knob rather than validating it.

`device` is the deliberate exception and stays a library parameter: `None`/`"auto"` detect, `"cuda"` pins without detecting, and everything else raises at construction. It is not a CLI option, because the only value a user could usefully type is the one auto-detection already finds.

The same rule applies to install hints: name the extra that actually makes the command work (`qwen-zimage`, not `diffusion`), and keep the printed command shell-quoted -- bare `pkg[extra]` is a glob in zsh.

## Local gate

Run `bash maintain.sh` from the repository root. The authoritative type gate is scoped to `src/`; full-project Pyright can exhaust Node memory on the ML dependency graph.

Boundary modules for cv2, Torch, and Diffusers may carry narrow per-file relaxations for unknown third-party types. Keep pure-logic files strict, preserve the local piexif stub, and fix real errors before widening a pragma.

## Model-adjacent tests

Do not classify an entire module as untestable because its main path downloads a model. Keep pure behavior covered without downloads, including:

- target-size selection in `test_invisible_engine.py`;
- unsharp and adaptive-polish helpers in `test_humanizer.py`;
- tiling geometry and blending in `test_tiling.py`;
- prompt-embedding cache keying, storage round-trip, and the cross-pipeline reuse
  that lets a stack load without its text encoder, in `test_qwen_zimage_pipeline.py`;
- the face stack's dtype, in `test_qwen_zimage_pipeline.py`. A subclass that changes
  the pipeline dtype for its own global model must not change the inherited face
  stage's; `sdxl-zimage` shipped doing exactly that and crashed on every image with a
  face. When one profile inherits another's stage, guard the invariants that stage
  relies on, not just the code path.

Use availability checks only for paths that actually load large models.

Environment setup, dependency recovery, CI behavior, and fixture policy: [`../../docs/development.md`](../../docs/development.md).
