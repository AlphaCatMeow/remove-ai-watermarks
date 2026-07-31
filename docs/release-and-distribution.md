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
- updates the repository conda recipe version and source-distribution SHA-256;
- triggers a factory rebuild of the Hugging Face Space.

The workflow can also be started manually with an optional version input.
Conda-forge updates are outside this workflow.

If a distribution job fails because a repository or Hugging Face credential is
invalid, rotate the corresponding GitHub secret and rerun the failed job. A
manual Homebrew formula update is the fallback when its automation is blocked.

The conda job uses the published artifact rather than a locally built archive
as the hash source and commits the resulting recipe change to `main`. Runtime
dependency mapping remains review-controlled: keep it aligned with the core
dependencies in `pyproject.toml`, and document any conda-forge package that is
unavailable and must be omitted. PyPI's version-split PyAV dependency maps to
`av >=16` in conda: the solver selects the Python-3.10-compatible build or the
current line according to the environment.

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

The ComfyUI nodes are maintained and versioned in their own repository. After
the matching source distribution appears on PyPI, `distribute.yml` dispatches
that repository's sync workflow with the exact library version and waits for it
to finish. The sync updates the dependency floor, runs compatibility tests,
bumps the node patch version, and publishes to the ComfyUI Registry only when
those tests pass. Its daily schedule remains as a recovery path if a release
dispatch is interrupted. The `COMFYUI_RELEASE_TOKEN` repository secret is a
fine-grained token limited to the ComfyUI node repository, with Actions read and
write access.

## Release verification

After publication, verify:

- both wheel and source distribution exist on PyPI;
- the package version matches the tag;
- the Homebrew formula points to the new source distribution;
- the distribution workflow completed successfully;
- the repository's conda recipe matches the published version and source
  distribution;
- the ComfyUI Registry node requires the new library version;
- a clean install can run `remove-ai-watermarks --version`.
