---
globs: ["src/**/*.py", "tests/**/*.py", "scripts/**/*.py", "pyproject.toml", "uv.lock", "maintain.sh", ".github/workflows/*.yml"]
description: Command contracts, project gate, typing boundaries, and model-adjacent test invariants.
---

# Development invariants

## Command contracts

Every single-image command declares `source` with `dir_okay=False`; `batch` declares its directory with `file_okay=False`. Keep `tests/test_cli_robustness.py::TestDirectoryInputIsRejected` as the regression guard.

Exit-code and no-signal behavior is a public contract. Read the command-line section of [`../../docs/module-internals.md`](../../docs/module-internals.md) before changing it.

## Local gate

Run `bash maintain.sh` from the repository root. The authoritative type gate is scoped to `src/`; full-project Pyright can exhaust Node memory on the ML dependency graph.

Boundary modules for cv2, Torch, and Diffusers may carry narrow per-file relaxations for unknown third-party types. Keep pure-logic files strict, preserve the local piexif stub, and fix real errors before widening a pragma.

## Model-adjacent tests

Do not classify an entire module as untestable because its main path downloads a model. Keep pure behavior covered without downloads, including:

- target-size selection in `test_invisible_engine.py`;
- unsharp and adaptive-polish helpers in `test_humanizer.py`;
- mocked device fallback in `test_img2img_runner.py`;
- tiling geometry and blending in `test_tiling.py`.

Use availability checks only for paths that actually load large models.

Environment setup, dependency recovery, CI behavior, and fixture policy: [`../../docs/development.md`](../../docs/development.md).
