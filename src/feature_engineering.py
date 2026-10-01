"""Turn a per-read table (see `src.data_parser`) into one feature vector
per transcript-position site.

Main entry point: `build_features(df) -> X`. It works on both labelled
training data and label-free new data, since it never reads `label` or
`gene_id`.

Two features are motivated by biology rather than general statistics:

- `FEATURE_COLUMNS` includes a one-hot encoding of the central 5-mer as a
  single DRACH motif, in addition to the position-wise 7-mer one-hot
  (see `_one_hot_central_motif`).
- `fit_kmer_baselines` and `build_features(df, kmer_baselines=...)` add an
  optional k-mer-normalised central signal, following m6Anet's feature
  design (Hendra et al., 2022, Nature Methods,
  https://www.nature.com/articles/s41592-022-01666-1). This is opt-in:
  fit `kmer_baselines` on the training split only, then reuse it
  (without refitting) for validation, test, and new data.
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

# The 18 DRACH motifs named in the project handout: D=[A,G,T], R=[A,G],
# fixed "AC", H=[A,C,T]. The central 5-mer (sequence[1:6]) should always be
# one of these; a motif outside this set falls into "other".
DRACH_MOTIFS = sorted(
    f"{d}{r}AC{h}"
    for d in "AGT"
    for r in "AG"
    for h in "ACT"
)
CENTRAL_MOTIF_LABELS = [*DRACH_MOTIFS, "other"]
CENTRAL_MOTIF_COLUMNS = [f"motif_{label}" for label in CENTRAL_MOTIF_LABELS]

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

# The three central-position features a modification most directly shifts
# (the pore reads ~5 bases at a time, so the central 5-mer sets their
# expected baseline). Used by fit_kmer_baselines/build_features below.
KMER_BASELINE_FEATURE_NAMES = [
    "central_log_dwell",
    "central_signal_sd",
    "central_signal_mean",
]
KMER_RESID_FEATURE_NAMES = [f"{name}_kmer_resid" for name in KMER_BASELINE_FEATURE_NAMES]
_GLOBAL_BASELINE_KEY = "__global__"

# agg() statistic -> output suffix.
_SIMPLE_STAT_SUFFIXES = {"mean": "mean", "std": "sd", "min": "min", "max": "max"}

# quantile value -> output suffix.
_QUANTILE_SUFFIXES = {0.25: "p25", 0.5: "median", 0.75: "p75"}

_STAT_SUFFIXES = (*_SIMPLE_STAT_SUFFIXES.values(), *_QUANTILE_SUFFIXES.values())

AGG_FEATURE_COLUMNS = [
    f"{feature}_{suffix}"
    for feature in READ_LEVEL_FEATURE_NAMES
    for suffix in _STAT_SUFFIXES
]

KMER_RESID_AGG_COLUMNS = [
    f"{feature}_{suffix}" for feature in KMER_RESID_FEATURE_NAMES for suffix in _STAT_SUFFIXES
]

SEQUENCE_FEATURE_COLUMNS = [
    f"seq_pos{pos}_{base}"
    for pos in range(SEQUENCE_LENGTH)
    for base in NUCLEOTIDES
]

# All columns build_features adds beyond the site keys when called without
# kmer_baselines. Use this to select the feature matrix out of
# build_features' output, e.g. X[FEATURE_COLUMNS].
FEATURE_COLUMNS = [
    "n_reads",
    *AGG_FEATURE_COLUMNS,
    *SEQUENCE_FEATURE_COLUMNS,
    *CENTRAL_MOTIF_COLUMNS,
]

# Columns build_features adds in addition to FEATURE_COLUMNS when a
# kmer_baselines table is passed in.
FEATURE_COLUMNS_WITH_KMER_RESID = [*FEATURE_COLUMNS, *KMER_RESID_AGG_COLUMNS]


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


def _central_5mer(df):
    """The central 5-mer (the DRACH motif position) from the 7-mer sequence."""
    return df["sequence"].str[1:6]


def _one_hot_sequence(sequence_by_site):
    """sequence_by_site: one 7-mer string per site, indexed by site."""
    columns = {}

    for pos in range(SEQUENCE_LENGTH):
        base_at_pos = sequence_by_site.str[pos]

        for base in NUCLEOTIDES:
            columns[f"seq_pos{pos}_{base}"] = (base_at_pos == base).astype(np.int8)

    return pd.DataFrame(columns, index=sequence_by_site.index)


def _one_hot_central_motif(central_5mer_by_site):
    """
    One-hot encoding of the central 5-mer as a single DRACH motif category
    (18 DRACH motifs plus "other", 19 columns).

    The central motif's identity alone is strongly associated with the
    modification label (see notebooks/EDA_Findings.ipynb, "Main
    findings"). This encodes that signal directly, rather than leaving a
    model to recover it from a combination of the per-position columns in
    `_one_hot_sequence`.
    """
    columns = {
        f"motif_{label}": (central_5mer_by_site == label).astype(np.int8)
        for label in DRACH_MOTIFS
    }
    is_other = ~central_5mer_by_site.isin(DRACH_MOTIFS)
    columns["motif_other"] = is_other.astype(np.int8)

    return pd.DataFrame(columns, index=central_5mer_by_site.index)


def fit_kmer_baselines(df, features=KMER_BASELINE_FEATURE_NAMES):
    """
    Compute each central motif's typical signal level, used by
    `build_features` to normalise the central-position signal against it.

    Nanopore current and dwell readings depend heavily on which ~5-base
    k-mer is in the pore, largely independent of modification state. For
    this reason, m6Anet's own feature set uses signal intensity, standard
    deviation and dwell time normalised per k-mer rather than raw values
    (Hendra et al., 2022, "Detection of m6A from direct RNA sequencing
    using a multiple instance learning framework", Nature Methods,
    https://www.nature.com/articles/s41592-022-01666-1).

    Parameters
    ----------
    df : pandas.DataFrame
        A per-read table (see `build_features`), from the training split
        only. Fit this table once on training data, then reuse it
        (without refitting) for validation, test, and new data via
        `build_features(df, kmer_baselines=...)`.

    Returns
    -------
    pandas.DataFrame
        Indexed by central 5-mer motif, with one column per feature in
        `features`: the median value across all reads with that central
        motif. Includes a "__global__" row, used as the fallback for a
        motif not present in this table.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    read_level = _derive_read_level_features(df)[features]
    read_level = read_level.assign(central_5mer=_central_5mer(df).to_numpy())

    baselines = read_level.groupby("central_5mer")[features].median()
    baselines.loc[_GLOBAL_BASELINE_KEY] = read_level[features].median()

    return baselines


def _kmer_residual_features(df, kmer_baselines):
    central_5mer = _central_5mer(df)
    read_level = _derive_read_level_features(df)[KMER_BASELINE_FEATURE_NAMES]

    residuals = {}

    for feature in KMER_BASELINE_FEATURE_NAMES:
        baseline_by_motif = kmer_baselines[feature].drop(
            _GLOBAL_BASELINE_KEY, errors="ignore"
        )
        global_baseline = kmer_baselines.loc[_GLOBAL_BASELINE_KEY, feature]
        matched_baseline = (
            central_5mer.map(baseline_by_motif).fillna(global_baseline).to_numpy()
        )
        residuals[f"{feature}_kmer_resid"] = (
            read_level[feature].to_numpy() - matched_baseline
        )

    return pd.DataFrame(residuals, index=df.index)


def build_features(df, kmer_baselines=None):
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
    kmer_baselines : pandas.DataFrame, optional
        The table returned by `fit_kmer_baselines`, fit on the training
        split. When given, adds the k-mer-normalised central-signal
        features (columns `KMER_RESID_AGG_COLUMNS`), and the output has
        `FEATURE_COLUMNS_WITH_KMER_RESID` instead of `FEATURE_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        One row per (transcript_id, transcript_position) site, with the
        site-key columns followed by the feature columns (`FEATURE_COLUMNS`,
        or `FEATURE_COLUMNS_WITH_KMER_RESID` when `kmer_baselines` is
        given). Columns are fixed and identically named regardless of
        input size. Carries no label or gene_id: join those back
        separately for modelling or evaluation. Never contains NaN: a
        site with a single read has its "_sd" columns set to 0 rather
        than left undefined.
    """
    _validate_input(df)

    read_level = _derive_read_level_features(df)
    read_level_feature_names = list(READ_LEVEL_FEATURE_NAMES)

    if kmer_baselines is not None:
        read_level = pd.concat(
            [read_level, _kmer_residual_features(df, kmer_baselines)], axis=1
        )
        read_level_feature_names += KMER_RESID_FEATURE_NAMES

    read_level[SITE_KEY_COLUMNS] = df[SITE_KEY_COLUMNS].to_numpy()

    grouped = read_level.groupby(SITE_KEY_COLUMNS, sort=False)[read_level_feature_names]

    simple_stats = grouped.agg(list(_SIMPLE_STAT_SUFFIXES))
    simple_stats.columns = [
        f"{feature}_{_SIMPLE_STAT_SUFFIXES[stat]}"
        for feature, stat in simple_stats.columns
    ]

    quantiles = grouped.quantile(list(_QUANTILE_SUFFIXES)).unstack(level=-1)
    quantiles.columns = [
        f"{feature}_{_QUANTILE_SUFFIXES[q]}" for feature, q in quantiles.columns
    ]

    agg_columns = AGG_FEATURE_COLUMNS + (
        KMER_RESID_AGG_COLUMNS if kmer_baselines is not None else []
    )
    site_stats = pd.concat([simple_stats, quantiles], axis=1)[agg_columns]

    # "sd" is the only statistic that can come back undefined, and only for
    # single-read sites (sample std of one observation). Treat that as zero
    # observed spread rather than passing NaN downstream.
    site_stats = site_stats.fillna(0.0)

    by_site = df.groupby(SITE_KEY_COLUMNS, sort=False)
    n_reads = by_site["n_reads"].first()
    sequence = by_site["sequence"].first()
    seq_onehot = _one_hot_sequence(sequence)
    motif_onehot = _one_hot_central_motif(sequence.str[1:6])

    features = pd.concat([n_reads, site_stats, seq_onehot, motif_onehot], axis=1)
    output_columns = (
        FEATURE_COLUMNS_WITH_KMER_RESID if kmer_baselines is not None else FEATURE_COLUMNS
    )
    features = features[output_columns].reset_index()

    return features
