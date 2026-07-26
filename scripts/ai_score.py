"""AI-generation scorer for scanned datasets.

Trains a gradient-boosted classifier on the metadata-derived labels of a
scan_dataset.py output and scores every file, WITHOUT needing the
metadata to be present at scoring time: the features are pixel and
container statistics, so a metadata-stripped file still gets a score.
The model is distribution-specific by design; evaluate it again before
using it on an unrelated target distribution.

Modes:
    uv run --with scikit-learn python scripts/ai_score.py train <scan_glob> <model.pkl> [v1|v2]
    uv run --with scikit-learn python scripts/ai_score.py score <scan_glob> <model.pkl> <out.jsonl>

<scan_glob> is a glob of scan_dataset shards, e.g. 'data/scan/part_*.jsonl'.
score output: one JSON line per file with file, sha256, ai_score, and the
label evidence when a metadata label existed (for monitoring drift).
"""

import glob
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

# AI-generator names in C2PA claim_generator_info (metadata-level truth).
AI_GENERATORS = (
    "openai",
    "adobe firefly",
    "adobe_firefly",
    "microsoft responsible ai",
    "microsoft_designer",
    "black forest labs",
    "fal-ai",
    "bria",
    "chatgpt",
    "stability",
    "dreamina",
    "canva",
)
# Human-origin software (negative evidence, NOT proof by itself).
HUMAN_SOFTWARE = (
    "photoshop",
    "lightroom",
    "picsart",
    "snapseed",
    "paint.net",
    "gimp",
    "capture one",
    "meitu",
    "xingtu",
    "snow",
)

_FEAT_SCALAR = [
    ("noise", "noise_std"),
    ("noise", "noise_kurtosis"),
    ("fft", "cfa_peak"),
    ("ela", "ela_mean"),
    ("ela", "ela_p95"),
    ("gradient", "laplacian_var"),
    ("color", "saturation_mean"),
    ("color", "value_mean"),
    ("dct", "benford_mad"),
]
FeatureSchema = Literal["v1", "v2"]
_FFT_BANDS = 8
_GRADIENT_BINS = 10
_COLOR_BINS = 64
_DCT_AC_POSITIONS = 8
_DCT_AC_BINS = 21
_JPEG_QUANT_TABLES = 2
_JPEG_QUANT_VALUES = 64
_SCORE_BATCH = 4096

_V1_FEATURE_NAMES = (
    *(key for _, key in _FEAT_SCALAR),
    *(f"fft_band_energy_{index}" for index in range(_FFT_BANDS)),
    *(f"gradient_hist_{index}" for index in range(_GRADIENT_BINS)),
    *(f"color_hist_4x4x4_{index}" for index in range(_COLOR_BINS)),
    "jpeg_444",
    "jpeg_progressive",
    "megapixels",
    "aspect_ratio",
    "format_jpeg",
    "format_png",
)
_V2_EXTRA_FEATURE_NAMES = (
    "cfa_peak_0",
    "cfa_peak_1",
    *(f"dct_ac_{position}_{bin_index}" for position in range(_DCT_AC_POSITIONS) for bin_index in range(_DCT_AC_BINS)),
    *(f"jpeg_quant_{table}_{index}" for table in range(_JPEG_QUANT_TABLES) for index in range(_JPEG_QUANT_VALUES)),
    "jpeg_quant_table_count",
    "jpeg_huffman_table_count",
    "jpeg_huffman_total_bytes",
    "jpeg_scan_count",
    "jpeg_restart_interval",
    "jpeg_precision_bits",
    "jpeg_adobe_transform",
    "jpeg_jfif_present",
    "format_webp",
    "format_isobmff",
    "format_other",
)
_SCHEMA_FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "v1": _V1_FEATURE_NAMES,
    "v2": _V1_FEATURE_NAMES + _V2_EXTRA_FEATURE_NAMES,
}


def parse_schema(value: Any) -> FeatureSchema:
    """Validate a feature-schema token (CLI argument or model-bundle field)."""
    if value == "v1" or value == "v2":
        return value
    raise ValueError(f"Unknown feature schema: {value!r}")


def feature_names(schema: FeatureSchema = "v1") -> tuple[str, ...]:
    """Return the stable ordered feature names for a model schema."""
    return _SCHEMA_FEATURE_NAMES[parse_schema(schema)]


def _fixed_values(values: Any, count: int, *, normalize: bool = False) -> list[float]:
    """Return a fixed-width float vector, padding absent values with NaN."""
    if not isinstance(values, (list, tuple)):
        return [float("nan")] * count
    result = [float(value) for value in values[:count]]
    if normalize and result:
        total = sum(value for value in result if math.isfinite(value))
        if total > 0:
            result = [value / total for value in result]
    return result + [float("nan")] * (count - len(result))


def _optional_float(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    return float(value) if isinstance(value, (int, float)) else float("nan")


def label_of(record: dict[str, Any]) -> int | None:
    """1 = strong metadata AI, 0 = human-origin, None = unlabeled."""
    store = record.get("c2pa_store") or {}
    for m in (store.get("manifests") or {}).values():
        for cgi in m.get("claim_generator_info") or []:
            if any(g in str(cgi.get("name", "")).lower() for g in AI_GENERATORS):
                return 1
        for a in m.get("assertions") or []:
            d = a.get("data") if isinstance(a.get("data"), dict) else {}
            if "trainedAlgorithmicMedia" in str(d.get("digitalSourceType", "")):
                return 1
    for c in record.get("png_chunks", []):
        t = c.get("text", "")
        kw = t.split("\x00")[0]
        if kw in ("prompt", "workflow", "parameters") or "AIGC" in t[:80]:
            return 1
    exif = record.get("exif", {})
    z = exif.get("0th") or {}
    e = exif.get("Exif") or {}
    if z.get("Make") and z.get("Model") and (e.get("MakerNote") or e.get("LensModel") or e.get("LensMake")):
        return 0
    for c in record.get("png_chunks", []):
        if c.get("apple_screenshot_marker"):
            return 0
    for v in (record.get("iptc") or {}).values():
        if str(v).strip() == "Screenshot":
            return 0
    if any(t in str(z.get("Software", "")).lower() for t in HUMAN_SOFTWARE):
        return 0
    return None


def features_of(record: dict[str, Any], *, schema: FeatureSchema = "v1") -> list[float]:
    """The structural feature vector (pixel + container stats, no metadata)."""
    v = [float((record.get(s) or {}).get(k, np.nan)) for s, k in _FEAT_SCALAR]
    fft = record.get("fft") or {}
    gradient = record.get("gradient") or {}
    color = record.get("color") or {}
    v.extend(_fixed_values(fft.get("fft_band_energy"), _FFT_BANDS))
    v.extend(_fixed_values(gradient.get("gradient_hist"), _GRADIENT_BINS))
    v.extend(_fixed_values(color.get("color_hist_4x4x4"), _COLOR_BINS))
    jf = record.get("jpeg_forensics") or {}
    v.append(1.0 if jf.get("subsampling") == "4:4:4" else 0.0)
    v.append(1.0 if jf.get("progressive") else 0.0)
    pil = record.get("pil") or {}
    w, h = pil.get("width") or 0, pil.get("height") or 0
    v += [float(w * h) / 1e6, float(w) / max(h, 1)]
    fmt = record.get("content_format")
    v += [1.0 if fmt == "jpeg" else 0.0, 1.0 if fmt == "png" else 0.0]
    if schema == "v2":
        v.extend(_fixed_values(fft.get("cfa_peaks"), 2))
        dct_hist = (record.get("dct") or {}).get("dct_ac_hist")
        for position in range(_DCT_AC_POSITIONS):
            values = dct_hist[position] if isinstance(dct_hist, list) and position < len(dct_hist) else None
            v.extend(_fixed_values(values, _DCT_AC_BINS, normalize=True))
        quant_tables = jf.get("quant_tables")
        for table in range(_JPEG_QUANT_TABLES):
            values = quant_tables.get(str(table)) if isinstance(quant_tables, dict) else None
            v.extend(_fixed_values(values, _JPEG_QUANT_VALUES))
        is_jpeg = fmt == "jpeg"
        huffman_tables = jf.get("huffman_tables_hex")
        huffman_values = huffman_tables if isinstance(huffman_tables, list) else []
        v.extend(
            [
                float(len(quant_tables)) if isinstance(quant_tables, dict) else (0.0 if is_jpeg else float("nan")),
                float(len(huffman_values)) if is_jpeg else float("nan"),
                (
                    float(sum(len(value) // 2 for value in huffman_values if isinstance(value, str)))
                    if is_jpeg
                    else float("nan")
                ),
                _optional_float(jf, "scan_count"),
                _optional_float(jf, "restart_interval"),
                _optional_float(jf, "precision_bits"),
                _optional_float(jf, "adobe_transform"),
                1.0 if jf.get("jfif") else (0.0 if is_jpeg else float("nan")),
                1.0 if fmt == "webp" else 0.0,
                1.0 if isinstance(fmt, str) and fmt.startswith("isobmff:") else 0.0,
                (
                    1.0
                    if fmt is not None and fmt not in {"jpeg", "png", "webp"} and not str(fmt).startswith("isobmff:")
                    else 0.0
                ),
            ]
        )
    elif schema != "v1":
        raise ValueError(f"Unknown feature schema: {schema}")
    expected = len(feature_names(schema))
    if len(v) != expected:
        raise ValueError(f"Feature schema {schema} produced {len(v)} values; expected {expected}")
    return v


def grouped_stratified_split(
    labels: np.ndarray,
    hashes: np.ndarray,
    *,
    test_size: float = 0.3,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split records by hash while preserving label balance across hash groups."""
    if len(labels) != len(hashes):
        raise ValueError("labels and hashes must have equal length")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    groups: dict[str, list[int]] = defaultdict(list)
    group_labels: dict[str, int] = {}
    for index, (label, digest) in enumerate(zip(labels, hashes, strict=True)):
        key = str(digest) if digest else f"__row_{index}"
        int_label = int(label)
        if key in group_labels and group_labels[key] != int_label:
            raise ValueError(f"Hash {key!r} has conflicting labels")
        group_labels[key] = int_label
        groups[key].append(index)
    keys_by_label: dict[int, list[str]] = defaultdict(list)
    for key, label in group_labels.items():
        keys_by_label[label].append(key)
    rng = np.random.default_rng(random_state)
    test_set: set[str] = set()
    for class_keys in keys_by_label.values():
        if len(class_keys) < 2:
            raise ValueError("Each label needs at least two distinct hash groups")
        shuffled = np.asarray(class_keys)
        rng.shuffle(shuffled)
        test_count = min(max(round(len(shuffled) * test_size), 1), len(shuffled) - 1)
        test_set.update(shuffled[:test_count].tolist())
    train_set = set(groups) - test_set
    train = np.asarray([index for key, indices in groups.items() if key in train_set for index in indices])
    test = np.asarray([index for key, indices in groups.items() if key in test_set for index in indices])
    return np.sort(train), np.sort(test)


def temporal_holdout_split(
    dates: np.ndarray,
    hashes: np.ndarray,
    *,
    train_fraction: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split chronologically and remove later copies of training hashes."""
    if len(dates) != len(hashes):
        raise ValueError("dates and hashes must have equal length")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    unique_dates = sorted({str(date) for date in dates})
    if len(unique_dates) < 2:
        raise ValueError("Temporal holdout needs at least two distinct dates")
    cutoff_index = min(int(len(unique_dates) * train_fraction), len(unique_dates) - 1)
    cutoff = unique_dates[cutoff_index]
    train = np.flatnonzero(dates < cutoff)
    test = np.flatnonzero((dates >= cutoff) & ~np.isin(hashes, hashes[train]))
    return train, test, cutoff


def model_schema(bundle: dict[str, Any]) -> FeatureSchema:
    """Read a model's feature schema, defaulting legacy bundles to v1."""
    return parse_schema(bundle.get("feature_schema", "v1"))


def iter_records(pattern: str) -> Any:
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                yield json.loads(line)


def cmd_train(pattern: str, model_path: str, schema: FeatureSchema = "v2") -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, roc_auc_score

    # keep the feature vector, not the record: a parsed scan record is an order of
    # magnitude larger than the row it collapses to.
    labeled: dict[str, tuple[list[float], int, str]] = {}
    missing_hash_index = 0
    for r in iter_records(pattern):
        if "noise" not in r:
            continue
        lab = label_of(r)
        if lab is None:
            continue
        digest = r.get("sha256")
        if not digest:
            digest = f"__missing_{missing_hash_index}"
            missing_hash_index += 1
        date = Path(r["file"]).parent.name
        previous = labeled.get(str(digest))
        if previous is not None and previous[1] != lab:
            raise ValueError(f"Hash {digest!r} has conflicting labels")
        if previous is None or date < previous[2]:
            labeled[str(digest)] = (features_of(r, schema=schema), lab, date)

    rows, labels, row_dates = zip(*labeled.values(), strict=True)
    X = np.asarray(rows)
    y = np.asarray(labels)
    dates = np.asarray(row_dates)
    hashes = np.asarray(list(labeled))
    print(
        f"unique labeled: {len(y)} "
        f"(pos {int(y.sum())}, neg {int((1 - y).sum())}, features {X.shape[1]}, schema {schema})"
    )

    def fit(train: np.ndarray) -> tuple[Any, Any]:
        rows = X[train]
        imputer = SimpleImputer(strategy="median")
        classifier = HistGradientBoostingClassifier(
            random_state=0,
            max_iter=300,
            l2_regularization=5.0,
            early_stopping=False,
        )
        classifier.fit(imputer.fit_transform(rows), y[train])
        return imputer, classifier

    train, test = grouped_stratified_split(y, hashes, test_size=0.3, random_state=0)
    imp, clf = fit(train)
    p = clf.predict_proba(imp.transform(X[test]))[:, 1]
    print(f"SHA-grouped holdout: AUC {roc_auc_score(y[test], p):.4f} | AP {average_precision_score(y[test], p):.4f}")

    temporal_train, temporal_test, cutoff = temporal_holdout_split(dates, hashes)
    if len(temporal_train) > 100 and len(temporal_test) > 100:
        imp_t, clf_t = fit(temporal_train)
        pt = clf_t.predict_proba(imp_t.transform(X[temporal_test]))[:, 1]
        auc_t = roc_auc_score(y[temporal_test], pt)
        ap_t = average_precision_score(y[temporal_test], pt)
        print(f"temporal (<{cutoff}): AUC {auc_t:.4f} | AP {ap_t:.4f}")

    all_indices = np.arange(len(y))
    imp, clf = fit(all_indices)
    bundle = {
        "imputer": imp,
        "clf": clf,
        "feature_schema": schema,
        "feature_names": feature_names(schema),
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"model written: {model_path}")


def cmd_score(pattern: str, model_path: str, out_path: str) -> None:
    # the model file is produced locally by `train`; do not load third-party pickles
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)  # noqa: S301
    imp, clf = bundle["imputer"], bundle["clf"]
    schema = model_schema(bundle)
    stored_names = bundle.get("feature_names")
    if stored_names is not None and tuple(stored_names) != feature_names(schema):
        raise ValueError(f"Model feature names do not match schema {schema}")
    n = 0
    batch_rows: list[list[float]] = []
    batch_meta: list[dict[str, Any]] = []

    def flush(out: Any) -> None:
        """Score one batch: per-record predict_proba is dominated by call overhead."""
        nonlocal n
        if not batch_rows:
            return
        scores = clf.predict_proba(imp.transform(np.asarray(batch_rows)))[:, 1]
        for meta, score in zip(batch_meta, scores, strict=True):
            out.write(json.dumps({**meta, "ai_score": round(float(score), 4)}) + "\n")
        n += len(batch_rows)
        batch_rows.clear()
        batch_meta.clear()
        print(f"  {n}", flush=True)

    with open(out_path, "w") as out:
        for r in iter_records(pattern):
            if "noise" not in r:
                continue
            batch_rows.append(features_of(r, schema=schema))
            batch_meta.append({"file": r["file"], "sha256": r.get("sha256"), "metadata_label": label_of(r)})
            if len(batch_rows) >= _SCORE_BATCH:
                flush(out)
        flush(out)
    print(f"scored {n} -> {out_path}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "score"):
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "train" and len(sys.argv) in (4, 5):
        schema = parse_schema(sys.argv[4]) if len(sys.argv) == 5 else "v2"
        cmd_train(sys.argv[2], sys.argv[3], schema)
    elif sys.argv[1] == "score" and len(sys.argv) == 5:
        cmd_score(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
