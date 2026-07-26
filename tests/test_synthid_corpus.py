"""Tests for the SynthID corpus ingestion script."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

# scripts/ is not an installed package; add it to the path for import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_corpus

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "synthid_corpus"
QUALITY_SET = CORPUS_DIR / "quality_sets" / "full_pipeline_quality_2026-07-25.csv"

EXPECTED_QUALITY_SOURCE_FILENAMES = {
    "ChatGPT Image May 30, 2026, 10_31_08 AM.png",
    "ChatGPT Image May 31, 2026, 02_02_23 PM.png",
    "ChatGPT Image May 31, 2026, 02_03_55 PM.png",
    "Gemini_Generated_Image_3mc4t93mc4t93mc4.png",
    "Gemini_Generated_Image_633uuy633uuy633u.png",
    "Gemini_Generated_Image_akdbeiakdbeiakdb.png",
    "Gemini_Generated_Image_y48j3cy48j3cy48j.png",
}


def _manifest_rows(root: Path) -> list[dict[str, str]]:
    with open(root / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_reusable_quality_set_has_expected_inputs_and_valid_hashes() -> None:
    with open(QUALITY_SET, newline="") as f:
        rows = list(csv.DictReader(f))

    # Keep this literal independent of the CSV so deleting a fixture fails.
    assert {row["source_filename"] for row in rows} == EXPECTED_QUALITY_SOURCE_FILENAMES
    for row in rows:
        corpus_path = CORPUS_DIR / row["corpus_path"]
        assert corpus_path.is_file(), corpus_path
        assert hashlib.sha256(corpus_path.read_bytes()).hexdigest() == row["sha256"]


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples not present")
class TestIngest:
    def test_ingest_openai_flags_synthid_metadata(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(
            synthid_corpus.cli,
            ["ingest", str(SAMPLES_DIR / "chatgpt-1.png"), "--label", "pos", "--root", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output

        rows = _manifest_rows(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["label"] == "pos"
        assert row["synthid_metadata"] == "yes"
        assert int(row["width"]) > 0
        assert int(row["height"]) > 0
        # The copied file lands under images/pos/ with a sha-prefixed name.
        assert (tmp_path / "images" / "pos" / row["filename"]).exists()

    def test_ingest_firefly_not_flagged(self, tmp_path: Path):
        runner = CliRunner()
        runner.invoke(
            synthid_corpus.cli,
            ["ingest", str(SAMPLES_DIR / "firefly-1.png"), "--label", "neg", "--root", str(tmp_path)],
        )
        rows = _manifest_rows(tmp_path)
        assert len(rows) == 1
        assert rows[0]["synthid_metadata"] == ""  # Adobe signs C2PA but not SynthID

    def test_ingest_dedupes_by_sha256(self, tmp_path: Path):
        runner = CliRunner()
        args = ["ingest", str(SAMPLES_DIR / "chatgpt-1.png"), "--label", "pos", "--root", str(tmp_path)]
        runner.invoke(synthid_corpus.cli, args)
        runner.invoke(synthid_corpus.cli, args)  # second time: duplicate
        assert len(_manifest_rows(tmp_path)) == 1

    def test_status_on_empty_corpus(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(synthid_corpus.cli, ["status", "--root", str(tmp_path)])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()
