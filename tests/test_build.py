"""Tests for the bulk builder."""

from rebounding.data import build


def test_build_game_end_to_end(sportvu_fixture_path, pbp_cache_dir):
    rows, report = build.build_game(sportvu_fixture_path, cache_dir=pbp_cache_dir)
    assert not rows.empty
    assert report.n_paired > 0
    assert (rows.groupby("ShotID").size() == 10).all()


def test_build_many_records_failures_instead_of_swallowing_them(tmp_path, pbp_cache_dir):
    """`massunpack` used `except Exception: pass`, so losses were invisible.

    The corpus really does contain broken archives -- 5 of the 636 files are
    truncated, four of them to 32 bytes -- so this path gets exercised for real.
    """
    broken = tmp_path / "broken.7z"
    broken.write_bytes(b"not an archive")

    rows, report = build.build_many([broken], cache_dir=pbp_cache_dir)
    assert rows.empty
    assert report.n_games == 1
    assert report.n_succeeded == 0
    assert "broken.7z" in report.failures
    assert report.failures["broken.7z"]  # carries the exception text
    assert "failed games" in report.summary()


def test_build_many_aggregates_across_games(sportvu_fixture_path, pbp_cache_dir, tmp_path):
    broken = tmp_path / "broken.7z"
    broken.write_bytes(b"not an archive")

    rows, report = build.build_many(
        [sportvu_fixture_path, broken], cache_dir=pbp_cache_dir
    )
    assert report.n_games == 2
    assert report.n_succeeded == 1
    assert report.n_rows == len(rows)
    assert report.n_shots_paired <= report.n_shots_seen
    assert report.seconds >= 0


def test_build_many_writes_parquet(sportvu_fixture_path, pbp_cache_dir, tmp_path):
    output = tmp_path / "nested" / "frame.parquet"
    rows, _ = build.build_many(
        [sportvu_fixture_path], cache_dir=pbp_cache_dir, output=output
    )
    assert output.exists()

    import pandas as pd

    assert len(pd.read_parquet(output)) == len(rows)


def test_summary_is_readable_when_everything_succeeds(sportvu_fixture_path, pbp_cache_dir):
    _, report = build.build_many([sportvu_fixture_path], cache_dir=pbp_cache_dir)
    summary = report.summary()
    assert "games" in summary
    assert "shots" in summary
    assert "failed games" not in summary
