# Release and distribution

This page describes the release behavior defined in this repository. External
registry state can change independently, so verify it during a release.

## Release sources of truth

The package version appears in:

- `pyproject.toml`;
- `src/remove_ai_watermarks/__init__.py`;
- the root package entry generated in `uv.lock`.

Update the first two, then refresh the lock file with uv. Do not edit a
line-number-specific location in `uv.lock`; its package order changes.

## Publish flow

PyPI publishing is triggered by a published GitHub Release, not by a tag push
alone.

The expected sequence:

1. update the version sources and lock file;
2. run the complete project gate;
3. commit the release change;
4. create an annotated `vX.Y.Z` tag;
5. push the commit and tag;
6. publish the GitHub Release.

`.github/workflows/publish.yml` then:

1. checks that the release tag matches `pyproject.toml`;
2. builds the package with uv;
3. publishes with `uv publish` through PyPI trusted publishing.

The workflow uses GitHub OIDC through the `pypi` environment. It does not read a
PyPI API token from the repository.

## Post-release distribution

`.github/workflows/distribute.yml` runs on the same published-release event. It
waits for the matching source distribution to appear on PyPI, then:

- updates the Homebrew tap formula URL and SHA-256;
- triggers a factory rebuild of the Hugging Face Space.

The workflow can also be started manually with an optional version input.
Conda-forge updates are outside this workflow.

If a distribution job fails because a repository or Hugging Face credential is
invalid, rotate the corresponding GitHub secret and rerun the failed job. A
manual Homebrew formula update is the fallback when its automation is blocked.

## Source distribution boundary

The wheel includes the package under `src/`.

The source distribution explicitly excludes `/data` through
`[tool.hatch.build.targets.sdist]` in `pyproject.toml`. Keep that exclusion:
calibration captures and test corpora do not belong in the published package
archive.

## Build backend

The package uses hatchling through the unpinned `hatchling` build requirement in
`pyproject.toml`. Uploading uses uv rather than the older twine-based action.

## Other channels

The repository includes a conda recipe under `packaging/conda/recipe.yaml`.
Keep its runtime dependencies aligned with `pyproject.toml`.

The ComfyUI nodes are maintained and versioned separately from this package.
A library release does not by itself publish a new ComfyUI node version.

## Release verification

After publication, verify:

- both wheel and source distribution exist on PyPI;
- the package version matches the tag;
- the Homebrew formula points to the new source distribution;
- the distribution workflow completed successfully;
- a clean install can run `remove-ai-watermarks --version`.
