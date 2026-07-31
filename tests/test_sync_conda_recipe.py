"""Tests for the release-time conda recipe synchronizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.sync_conda_recipe import update_recipe

_OLD_SHA = "a" * 64
_NEW_SHA = "b" * 64
_RECIPE = f"""\
schema_version: 1

context:
  version: "1.2.3"

source:
  sha256: {_OLD_SHA}
"""


def test_update_recipe_replaces_version_and_hash() -> None:
    updated = update_recipe(_RECIPE, version="2.0.0", sha256=_NEW_SHA)

    assert '  version: "2.0.0"' in updated
    assert f"  sha256: {_NEW_SHA}" in updated
    assert "1.2.3" not in updated
    assert _OLD_SHA not in updated


@pytest.mark.parametrize("sha256", ["", "A" * 64, "f" * 63, "not-a-hash"])
def test_update_recipe_rejects_invalid_hash(sha256: str) -> None:
    with pytest.raises(ValueError, match="sha256"):
        update_recipe(_RECIPE, version="2.0.0", sha256=sha256)


def test_update_recipe_rejects_ambiguous_recipe() -> None:
    duplicate = _RECIPE + f"\nsource_two:\n  sha256: {_OLD_SHA}\n"

    with pytest.raises(ValueError, match="exactly one"):
        update_recipe(duplicate, version="2.0.0", sha256=_NEW_SHA)


def test_repository_recipe_stays_metadata_only() -> None:
    recipe = Path("packaging/conda/recipe.yaml").read_text(encoding="utf-8")

    assert "    - av >=16\n" not in recipe
