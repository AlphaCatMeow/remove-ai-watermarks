"""Tests for the standalone structural AI-generation scorer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import ai_score


def _complete_record() -> dict[str, Any]:
    return {
        "noise": {"noise_std": 1.0, "noise_kurtosis": 2.0},
        "fft": {
            "cfa_peak": 3.0,
            "cfa_peaks": [2.5, 3.0],
            "fft_band_energy": list(range(8)),
        },
        "ela": {"ela_mean": 4.0, "ela_p95": 5.0},
        "gradient": {"laplacian_var": 6.0, "gradient_hist": list(range(10))},
        "color": {
            "saturation_mean": 0.5,
            "value_mean": 0.75,
            "color_hist_4x4x4": list(range(64)),
        },
        "dct": {
            "benford_mad": 0.1,
            "dct_ac_hist": [[1] * 21 for _ in range(8)],
        },
        "jpeg_forensics": {
            "subsampling": "4:4:4",
            "progressive": True,
            "quant_tables": {
                "0": list(range(1, 65)),
                "1": list(range(65, 129)),
            },
            "huffman_tables_hex": ["00ff", "abcd12"],
            "scan_count": 10,
            "restart_interval": 4,
            "precision_bits": 8,
            "adobe_transform": 1,
            "jfif": {"version": "1.1"},
        },
        "pil": {"width": 2000, "height": 1000},
        "content_format": "jpeg",
    }


def test_v1_feature_schema_is_fixed_for_sparse_records() -> None:
    assert len(ai_score.feature_names("v1")) == 97
    assert len(ai_score.features_of({}, schema="v1")) == 97


def test_v2_feature_schema_includes_existing_forensic_data() -> None:
    record = _complete_record()

    names = ai_score.feature_names("v2")
    values = ai_score.features_of(record, schema="v2")
    by_name = dict(zip(names, values, strict=True))

    assert len(names) == len(values) == 406
    assert by_name["cfa_peak_0"] == 2.5
    assert by_name["cfa_peak_1"] == 3.0
    assert by_name["dct_ac_0_0"] == 1 / 21
    assert by_name["dct_ac_7_20"] == 1 / 21
    assert by_name["jpeg_quant_0_0"] == 1.0
    assert by_name["jpeg_quant_1_63"] == 128.0
    assert by_name["jpeg_quant_table_count"] == 2.0
    assert by_name["jpeg_huffman_table_count"] == 2.0
    assert by_name["jpeg_huffman_total_bytes"] == 5.0
    assert by_name["jpeg_scan_count"] == 10.0
    assert by_name["jpeg_jfif_present"] == 1.0
    assert by_name["format_webp"] == 0.0
    assert by_name["format_isobmff"] == 0.0
    assert by_name["format_other"] == 0.0


def test_v2_feature_schema_is_fixed_when_forensics_are_missing() -> None:
    names = ai_score.feature_names("v2")
    values = ai_score.features_of({}, schema="v2")

    assert len(names) == len(values) == 406
    assert np.isnan(values[names.index("dct_ac_0_0")])
    assert np.isnan(values[names.index("jpeg_quant_0_0")])
    assert np.isnan(values[names.index("jpeg_scan_count")])


def test_grouped_stratified_split_keeps_hashes_on_one_side() -> None:
    labels = np.asarray([1, 1, 1, 0, 0, 0, 1, 0])
    hashes = np.asarray(["a", "a", "b", "c", "c", "d", "e", "f"])

    train, test = ai_score.grouped_stratified_split(labels, hashes, test_size=0.5, random_state=7)

    assert set(hashes[train]).isdisjoint(set(hashes[test]))
    assert set(labels[train]) == {0, 1}
    assert set(labels[test]) == {0, 1}
    assert sorted(np.concatenate([train, test]).tolist()) == list(range(len(labels)))


def test_grouped_stratified_split_rejects_conflicting_labels() -> None:
    labels = np.asarray([0, 1, 0, 1])
    hashes = np.asarray(["same", "same", "negative", "positive"])

    with np.testing.assert_raises_regex(ValueError, "conflicting labels"):
        ai_score.grouped_stratified_split(labels, hashes)


def test_temporal_holdout_excludes_hashes_seen_during_training() -> None:
    dates = np.asarray(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-04"])
    hashes = np.asarray(["repeated", "old", "middle", "new-a", "repeated", "new-b"])

    train, test, cutoff = ai_score.temporal_holdout_split(dates, hashes, train_fraction=0.5)

    assert cutoff == "2026-01-03"
    assert set(hashes[train]).isdisjoint(set(hashes[test]))
    assert set(hashes[test]) == {"new-a", "new-b"}


def test_legacy_model_bundle_defaults_to_v1_schema() -> None:
    assert ai_score.model_schema({}) == "v1"
    assert ai_score.model_schema({"feature_schema": "v2"}) == "v2"
