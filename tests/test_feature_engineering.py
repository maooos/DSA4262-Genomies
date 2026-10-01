import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FEATURE_COLUMNS,
    RAW_FEATURE_NAMES,
    SITE_KEY_COLUMNS,
    build_features,
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
            _make_reads("T1", 10, "AAGACCA", n_reads=5, rng=rng),
            _make_reads("T1", 25, "CGGACTT", n_reads=1, rng=rng),
            _make_reads("T2", 7, "TGAACAC", n_reads=3, rng=rng),
        ],
        ignore_index=True,
    )


def test_output_shape_and_keys(synthetic_df):
    X = build_features(synthetic_df)

    assert len(X) == 3
    assert list(X.columns[: len(SITE_KEY_COLUMNS)]) == SITE_KEY_COLUMNS
    assert set(X.columns) == set(SITE_KEY_COLUMNS) | set(FEATURE_COLUMNS)
    assert not X.isna().any().any()
    assert not X.duplicated(SITE_KEY_COLUMNS).any()


def test_label_and_gene_id_not_required(synthetic_df):
    unlabelled = synthetic_df.drop(columns=["label", "gene_id"])
    X = build_features(unlabelled)

    assert "label" not in X.columns
    assert "gene_id" not in X.columns
    assert len(X) == 3


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
    one_site = synthetic_df[synthetic_df["transcript_id"] == "T2"]
    X = build_features(one_site)

    assert len(X) == 1
    assert X.iloc[0]["transcript_id"] == "T2"
