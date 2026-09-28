import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SIGNAL_PATH = ROOT / "data0" / "dataset0.json.gz"
LABEL_PATH = ROOT / "data0" / "data.info.labelled"

# Read only the first site from the compressed file.
with gzip.open(SIGNAL_PATH, "rt", encoding="utf-8") as handle:
    record = json.loads(next(handle))

transcript_id, positions = next(iter(record.items()))
position, sequences = next(iter(positions.items()))
sequence, reads = next(iter(sequences.items()))

features = np.asarray(reads, dtype=float)

if features.ndim != 2 or features.shape[1] != 9:
    raise ValueError(f"Expected a reads × 9 array; got {features.shape}")

if not np.isfinite(features).all():
    raise ValueError("This site contains missing or non-finite features.")

# Match this site to its label.
labels = pd.read_csv(
    LABEL_PATH,
    dtype={"transcript_id": str, "transcript_position": "int64"},
)

match = labels.loc[
    (labels["transcript_id"] == transcript_id)
    & (labels["transcript_position"] == int(position))
]

if len(match) != 1:
    raise ValueError(f"Expected one matching label; found {len(match)}")

print("Transcript:", transcript_id)
print("Position:", position)
print("7-mer:", sequence)
print("Number of reads:", features.shape[0])
print("Features per read:", features.shape[1])
print("First read:", features[0].tolist())
print("\nMatching metadata:")
print(match.to_string(index=False))