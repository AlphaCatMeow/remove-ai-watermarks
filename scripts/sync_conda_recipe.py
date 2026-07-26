"""Update the conda recipe to a published package version and sdist hash."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_VERSION_LINE = re.compile(r'^  version: "[^"]+"$', re.MULTILINE)
_SHA256_LINE = re.compile(r"^  sha256: [0-9a-f]{64}$", re.MULTILINE)


def update_recipe(text: str, *, version: str, sha256: str) -> str:
    """Return recipe text with exactly one context version and source hash updated."""
    if not version or any(char.isspace() for char in version):
        raise ValueError("version must be a non-empty value without whitespace")
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")

    updated, version_count = _VERSION_LINE.subn(f'  version: "{version}"', text)
    updated, sha_count = _SHA256_LINE.subn(f"  sha256: {sha256}", updated)
    if version_count != 1 or sha_count != 1:
        raise ValueError(
            "expected exactly one context version and one source sha256 "
            f"(found version={version_count}, sha256={sha_count})"
        )
    return updated


def main() -> None:
    """Update the selected recipe in place."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path("packaging/conda/recipe.yaml"),
    )
    args = parser.parse_args()

    original = args.recipe.read_text()
    updated = update_recipe(original, version=args.version, sha256=args.sha256)
    args.recipe.write_text(updated)
    log.info("Updated %s to version %s", args.recipe, args.version)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
