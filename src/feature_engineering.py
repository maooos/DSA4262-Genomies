"""Turn a per-read table (see `src.data_parser`) into one fixed-length
feature vector per transcript-position site.

Entry point: `build_features(df) -> X`. The same function is used for
training data and for new/unseen data: it never looks at `label` or
`gene_id`, so it has nothing to leak and nothing that forces those
columns to be present.
"""

import numpy as np
import pandas as pd

# Order defined by src.data_parser.FEATURE_NAMES.
RAW_FEATURE_NAMES = [
    "minus1_dwell",
    "minus1_signal_sd",
    "minus1_signal_mean",
    "central_dwell",
    "central_signal_sd",
    "central_signal_mean",
    "plus1_dwell",
    "plus1_signal_sd",
    "plus1_signal_mean",
]

SITE_KEY_COLUMNS = ["transcript_id", "transcript_position"]

REQUIRED_COLUMNS = [*SITE_KEY_COLUMNS, "sequence", "n_reads", *RAW_FEATURE_NAMES]

SEQUENCE_LENGTH = 7
NUCLEOTIDES = ["A", "C", "G", "T"]

# Per-read features built from the 9 raw signal measurements:
# dwell times are log-transformed (right-skewed, strictly positive), and
# the central position is contrasted against each flank to capture the
# shift a modification causes relative to its neighbours.
READ_LEVEL_FEATURE_NAMES = [
    "minus1_log_dwell",
    "central_log_dwell",
    "plus1_log_dwell",
    "minus1_signal_sd",
    "central_signal_sd",
    "plus1_signal_sd",
    "minus1_signal_mean",
    "central_signal_mean",
    "plus1_signal_mean",
    "dwell_contrast_minus1",
    "dwell_contrast_plus1",
    "sd_contrast_minus1",
    "sd_contrast_plus1",
    "mean_contrast_minus1",
    "mean_contrast_plus1",
]

# agg() statistic -> output suffix.
_SIMPLE_STAT_SUFFIXES = {"mean": "mean", "std": "sd", "min": "min", "max": "max"}

# quantile value -> output suffix.
_QUANTILE_SUFFIXES = {0.25: "p25", 0.5: "median", 0.75: "p75"}

AGG_FEATURE_COLUMNS = [
    f"{feature}_{suffix}"
    for feature in READ_LEVEL_FEATURE_NAMES
    for suffix in (*_SIMPLE_STAT_SUFFIXES.values(), *_QUANTILE_SUFFIXES.values())
]

SEQUENCE_FEATURE_COLUMNS = [
    f"seq_pos{pos}_{base}"
    for pos in range(SEQUENCE_LENGTH)
    for base in NUCLEOTIDES
]

# All columns build_features adds beyond the site keys. Use this to select
# the feature matrix out of build_features' output, e.g. X[FEATURE_COLUMNS].
FEATURE_COLUMNS = ["n_reads", *AGG_FEATURE_COLUMNS, *SEQUENCE_FEATURE_COLUMNS]


def _validate_input(df):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Input has no rows.")

    bad_sequences = ~df["sequence"].str.fullmatch(f"[ACGT]{{{SEQUENCE_LENGTH}}}")

    if bad_sequences.any():
        example = df.loc[bad_sequences, "sequence"].iloc[0]
        raise ValueError(
            f"Expected {SEQUENCE_LENGTH}-base sequences using A/C/G/T; "
            f"got {example!r}."
        )


def _derive_read_level_features(df):
    """One row per read, matching df's index, with READ_LEVEL_FEATURE_NAMES."""
    minus1_log_dwell = np.log(df["minus1_dwell"].to_numpy())
    central_log_dwell = np.log(df["central_dwell"].to_numpy())
    plus1_log_dwell = np.log(df["plus1_dwell"].to_numpy())

    return pd.DataFrame(
        {
            "minus1_log_dwell": minus1_log_dwell,
            "central_log_dwell": central_log_dwell,
            "plus1_log_dwell": plus1_log_dwell,
            "minus1_signal_sd": df["minus1_signal_sd"].to_numpy(),
            "central_signal_sd": df["central_signal_sd"].to_numpy(),
            "plus1_signal_sd": df["plus1_signal_sd"].to_numpy(),
            "minus1_signal_mean": df["minus1_signal_mean"].to_numpy(),
            "central_signal_mean": df["central_signal_mean"].to_numpy(),
            "plus1_signal_mean": df["plus1_signal_mean"].to_numpy(),
            "dwell_contrast_minus1": central_log_dwell - minus1_log_dwell,
            "dwell_contrast_plus1": central_log_dwell - plus1_log_dwell,
            "sd_contrast_minus1": (
                df["central_signal_sd"].to_numpy() - df["minus1_signal_sd"].to_numpy()
            ),
            "sd_contrast_plus1": (
                df["central_signal_sd"].to_numpy() - df["plus1_signal_sd"].to_numpy()
            ),
            "mean_contrast_minus1": (
                df["central_signal_mean"].to_numpy()
                - df["minus1_signal_mean"].to_numpy()
            ),
            "mean_contrast_plus1": (
                df["central_signal_mean"].to_numpy()
                - df["plus1_signal_mean"].to_numpy()
            ),
        },
        index=df.index,
    )


def _one_hot_sequence(sequence_by_site):
    """sequence_by_site: one 7-mer string per site, indexed by site."""
    columns = {}

    for pos in range(SEQUENCE_LENGTH):
        base_at_pos = sequence_by_site.str[pos]

        for base in NUCLEOTIDES:
            columns[f"seq_pos{pos}_{base}"] = (base_at_pos == base).astype(np.int8)

    return pd.DataFrame(columns, index=sequence_by_site.index)


def build_features(df):
    """
    Aggregate a per-read table into one feature row per site.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-read table with the columns produced by
        `src.data_parser.parse_to_parquet` (or an equivalent read from the
        resulting Parquet file): transcript_id, transcript_position,
        sequence, n_reads, and the nine raw signal features. Optional
        columns (gene_id, label, central_5mer, read_idx, ...) are ignored,
        so the same function works on labelled training data and on
        label-free new data.

    Returns
    -------
    pandas.DataFrame
        One row per (transcript_id, transcript_position) site, in first-seen
        order, with the two site-key columns followed by the feature
        columns listed in FEATURE_COLUMNS (fixed-length and identically
        named regardless of input size, so the output can be reindexed /
        concatenated across batches). Carries no label or gene_id: join
        those back in separately for modelling or evaluation. Never
        contains NaN: a site with a single read has its "_sd" columns set
        to 0 rather than left undefined.
    """
    _validate_input(df)

    read_level = _derive_read_level_features(df)
    read_level[SITE_KEY_COLUMNS] = df[SITE_KEY_COLUMNS].to_numpy()

    grouped = read_level.groupby(SITE_KEY_COLUMNS, sort=False)[READ_LEVEL_FEATURE_NAMES]

    # Note: do not use groupby.describe() here. It is dramatically slower
    # than agg()/quantile() on a wide group (minutes instead of seconds at
    # dataset scale), because it does not go through the same vectorised
    # Cython reduction path.
    simple_stats = grouped.agg(list(_SIMPLE_STAT_SUFFIXES))
    simple_stats.columns = [
        f"{feature}_{_SIMPLE_STAT_SUFFIXES[stat]}"
        for feature, stat in simple_stats.columns
    ]

    quantiles = grouped.quantile(list(_QUANTILE_SUFFIXES)).unstack(level=-1)
    quantiles.columns = [
        f"{feature}_{_QUANTILE_SUFFIXES[q]}" for feature, q in quantiles.columns
    ]

    site_stats = pd.concat([simple_stats, quantiles], axis=1)[AGG_FEATURE_COLUMNS]

    # "sd" is the only statistic that can come back undefined, and only for
    # single-read sites (sample std of one observation). Treat that as zero
    # observed spread rather than passing NaN downstream.
    site_stats = site_stats.fillna(0.0)

    by_site = df.groupby(SITE_KEY_COLUMNS, sort=False)
    n_reads = by_site["n_reads"].first()
    sequence = by_site["sequence"].first()
    seq_onehot = _one_hot_sequence(sequence)

    features = pd.concat([n_reads, site_stats, seq_onehot], axis=1)
    features = features[FEATURE_COLUMNS].reset_index()

    return features
