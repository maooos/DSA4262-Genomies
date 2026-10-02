import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    CENTRAL_MOTIF_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_WITH_KMER_RESID,
    KMER_RESID_AGG_COLUMNS,
    RAW_FEATURE_NAMES,
    SITE_KEY_COLUMNS,
    build_features,
    fit_kmer_baselines,
)


def _make_reads(transcript_id, position, sequence, n_reads, rng):
    """Build `n_reads` synthetic read rows for one site, parquet-shaped."""
    raw = rng.uniform(low=0.01, high=5.0, size=(n_reads, len(RAW_FEATURE_NAMES)))

    frame = pd.DataFrame(raw, columns=RAW_FEATURE_NAMES)
    frame.insert(0, "transcript_id", transcript_id)
    frame.insert(1, "transcript_position", position)
    frame.insert(2, "sequence", sequence)
    frame["n_reads"] = n_reads
    frame["read_idx"] = np.arange(n_reads)
    frame["gene_id"] = "GENE1"
    frame["label"] = 0.0
    return frame


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(0)
    return pd.concat(
        [
            _make_reads("T1", 10, "AAGACCA", n_reads=5, rng=rng),  # central AGACC (DRACH)
            _make_reads("T1", 25, "CGGACTT", n_reads=1, rng=rng),  # central GGACT (DRACH)
            _make_reads("T2", 7, "TGAACAC", n_reads=3, rng=rng),  # central GAACA (DRACH)
            _make_reads("T2", 40, "ACCCCCA", n_reads=4, rng=rng),  # central CCCCC (not DRACH)
        ],
        ignore_index=True,
    )


def test_output_shape_and_keys(synthetic_df):
    X = build_features(synthetic_df)

    assert len(X) == 4
    assert list(X.columns[: len(SITE_KEY_COLUMNS)]) == SITE_KEY_COLUMNS
    assert set(X.columns) == set(SITE_KEY_COLUMNS) | set(FEATURE_COLUMNS)
    assert not X.isna().any().any()
    assert not X.duplicated(SITE_KEY_COLUMNS).any()


def test_label_and_gene_id_not_required(synthetic_df):
    unlabelled = synthetic_df.drop(columns=["label", "gene_id"])
    X = build_features(unlabelled)

    assert "label" not in X.columns
    assert "gene_id" not in X.columns
    assert len(X) == 4


def test_single_read_site_has_zero_sd_not_nan(synthetic_df):
    """A one-read site has no observed spread: "_sd" columns should read 0,
    not NaN (sample std of a single value is undefined)."""
    X = build_features(synthetic_df).set_index(SITE_KEY_COLUMNS)
    row = X.loc[("T1", 25)]

    sd_cols = [c for c in FEATURE_COLUMNS if c.endswith("_sd")]
    assert (row[sd_cols] == 0).all()


def test_n_reads_matches_coverage(synthetic_df):
    X = build_features(synthetic_df).set_index(SITE_KEY_COLUMNS)

    assert X.loc[("T1", 10), "n_reads"] == 5
    assert X.loc[("T1", 25), "n_reads"] == 1
    assert X.loc[("T2", 7), "n_reads"] == 3


def test_aggregates_match_independent_computation(synthetic_df):
    """Recompute stats via a separate code path and compare, to catch
    group-alignment bugs between the different groupby calls inside
    build_features (n_reads / sequence / per-read stats)."""
    X = build_features(synthetic_df).set_index(SITE_KEY_COLUMNS)

    for (transcript_id, position), rows in synthetic_df.groupby(
        SITE_KEY_COLUMNS, sort=False
    ):
        x_row = X.loc[(transcript_id, position)]

        central_log_dwell = np.log(rows["central_dwell"])
        minus1_log_dwell = np.log(rows["minus1_dwell"])

        assert x_row["central_log_dwell_mean"] == pytest.approx(
            central_log_dwell.mean()
        )
        assert x_row["central_signal_mean_max"] == pytest.approx(
            rows["central_signal_mean"].max()
        )
        assert x_row["minus1_signal_sd_min"] == pytest.approx(
            rows["minus1_signal_sd"].min()
        )
        assert x_row["dwell_contrast_minus1_mean"] == pytest.approx(
            (central_log_dwell - minus1_log_dwell).mean()
        )

        if len(rows) > 1:
            assert x_row["central_signal_mean_sd"] == pytest.approx(
                rows["central_signal_mean"].std()
            )


def test_sequence_one_hot_matches_site_sequence(synthetic_df):
    X = build_features(synthetic_df).set_index(SITE_KEY_COLUMNS)

    row = X.loc[("T1", 10)]
    seq_cols = [c for c in X.columns if c.startswith("seq_pos")]
    assert row[seq_cols].sum() == 7

    for pos, base in enumerate("AAGACCA"):
        assert row[f"seq_pos{pos}_{base}"] == 1


def test_missing_required_column_raises(synthetic_df):
    broken = synthetic_df.drop(columns=["central_dwell"])

    with pytest.raises(ValueError, match="Missing required columns"):
        build_features(broken)


def test_invalid_sequence_raises(synthetic_df):
    broken = synthetic_df.copy()
    broken.loc[0, "sequence"] = "NNNNNNN"

    with pytest.raises(ValueError, match="A/C/G/T"):
        build_features(broken)


def test_empty_input_raises():
    empty = pd.DataFrame(
        columns=[*SITE_KEY_COLUMNS, "sequence", "n_reads", *RAW_FEATURE_NAMES]
    )

    with pytest.raises(ValueError, match="no rows"):
        build_features(empty)


def test_works_on_a_single_site_batch(synthetic_df):
    """build_features must also work on a one-site slice, since new/unseen
    data may be processed site-by-site or in arbitrary batches."""
    one_site = synthetic_df[
        (synthetic_df["transcript_id"] == "T2")
        & (synthetic_df["transcript_position"] == 7)
    ]
    X = build_features(one_site)

    assert len(X) == 1
    assert X.iloc[0]["transcript_id"] == "T2"


def test_central_motif_one_hot(synthetic_df):
    X = build_features(synthetic_df).set_index(SITE_KEY_COLUMNS)

    # T1/10's 7-mer is AAGACCA -> central 5-mer AGACC, a DRACH motif.
    row = X.loc[("T1", 10)]
    assert row[CENTRAL_MOTIF_COLUMNS].sum() == 1
    assert row["motif_AGACC"] == 1
    assert row["motif_other"] == 0

    # T2/40's 7-mer is ACCCCCA -> central 5-mer CCCCC, not a DRACH motif.
    row = X.loc[("T2", 40)]
    assert row[CENTRAL_MOTIF_COLUMNS].sum() == 1
    assert row["motif_other"] == 1


def test_kmer_baselines_add_opt_in_residual_columns(synthetic_df):
    baselines = fit_kmer_baselines(synthetic_df)
    X_default = build_features(synthetic_df)
    X_with_resid = build_features(synthetic_df, kmer_baselines=baselines)

    assert list(X_default.columns) == [*SITE_KEY_COLUMNS, *FEATURE_COLUMNS]
    assert list(X_with_resid.columns) == [
        *SITE_KEY_COLUMNS,
        *FEATURE_COLUMNS_WITH_KMER_RESID,
    ]
    assert not X_with_resid.isna().any().any()

    # T1/25 is the only site with a single read, and the only read sharing
    # its motif (GGACT): the motif's median is that one read's own value,
    # so its residual against the baseline is exactly 0.
    for col in KMER_RESID_AGG_COLUMNS:
        if col.endswith("_mean"):
            assert X_with_resid.loc[
                (X_with_resid["transcript_id"] == "T1")
                & (X_with_resid["transcript_position"] == 25),
                col,
            ].item() == pytest.approx(0.0, abs=1e-8)


def test_kmer_baseline_residuals_match_independent_computation(synthetic_df):
    """Recompute the motif baseline and residual via a separate code path
    (plain groupby, not fit_kmer_baselines/build_features) and compare, to
    catch bugs in either function."""
    baselines = fit_kmer_baselines(synthetic_df)
    X = build_features(synthetic_df, kmer_baselines=baselines).set_index(
        SITE_KEY_COLUMNS
    )

    central_5mer = synthetic_df["sequence"].str[1:6]
    log_dwell_baseline_by_motif = (
        np.log(synthetic_df["central_dwell"]).groupby(central_5mer).median()
    )

    site_rows = synthetic_df[
        (synthetic_df["transcript_id"] == "T1")
        & (synthetic_df["transcript_position"] == 10)
    ]
    site_motif = site_rows["sequence"].iloc[0][1:6]
    expected_resid_mean = (
        np.log(site_rows["central_dwell"]) - log_dwell_baseline_by_motif[site_motif]
    ).mean()

    assert X.loc[("T1", 10), "central_log_dwell_kmer_resid_mean"] == pytest.approx(
        expected_resid_mean
    )


def test_kmer_baselines_fall_back_to_global_for_unseen_motif(synthetic_df):
    baselines = fit_kmer_baselines(synthetic_df)
    new_site = _make_reads(
        "T3", 99, "TTTTTTT", n_reads=3, rng=np.random.default_rng(1)
    )  # central TTTTT: not in the fitted baseline table or DRACH_MOTIFS

    X = build_features(new_site, kmer_baselines=baselines)

    assert not X.isna().any().any()
    assert X.loc[0, "motif_other"] == 1
