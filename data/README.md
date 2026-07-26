# Repository data

Tracked data is organized by purpose:

```text
data/
  fixtures/
    provenance/   Real format and provenance fixtures used by tests
  calibration/
    <vendor>/     Minimal controlled inputs needed to rebuild detector assets
  synthid/
    originals/    Canonical provider-oracle fixtures, stored once
    manifest.csv  Provenance and verification record for each original
    full-pipeline-quality.csv
                  Reusable full-pipeline evaluation selection
  evaluations/
    fidelity/     Evaluation instructions and hand-verified ground truth
```

## Storage rules

1. Store each binary image once. Evaluation manifests and documentation point
   to its canonical location.
2. Put executable test fixtures in `fixtures/`.
3. Put only the minimal reproducible detector inputs in `calibration/`.
4. Put externally verified SynthID originals in `synthid/originals/` and keep
   both CSV files synchronized.
5. Keep evaluation outputs outside the repository. Record reproducible
   commands, hashes, and oracle verdicts instead of committing another corpus
   copy. A small curated before-and-after example may live in `docs/images/`
   when it is part of the public documentation.
6. Runtime detector assets belong in `src/remove_ai_watermarks/assets/`.
   Unregistered research candidates belong in
   `scripts/assets/visible-mark-candidates/` so they are not shipped in the
   wheel.

The source distribution excludes `data/`; the wheel contains only package
runtime assets.
