"""Command line entry points.

    python -m rebounding.cli build   --games data/7zips --out data/frame.parquet
    python -m rebounding.cli inspect --game data/7zips/01.01.2016.CHA.at.TOR.7z

``build`` needs play-by-play. ``stats.nba.com`` no longer serves it, so it comes
from Basketball-Reference (see :mod:`rebounding.data.bref`) unless a cached NBA
payload for that game is sitting in ``--pbp-cache``. Pages are cached under
``--bref-cache``, and fetching is rate limited to 20 a minute, so the first full
run over 636 games takes about 35 minutes and later ones are offline.
``inspect`` is tracking-only and has never needed the network.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rebounding.data import build, sportvu


def _build(args: argparse.Namespace) -> int:
    games = sorted(Path(args.games).glob("*.7z")) if Path(args.games).is_dir() else [Path(args.games)]
    if args.limit:
        games = games[: args.limit]
    if not games:
        print(f"no tracking archives found under {args.games}", file=sys.stderr)
        return 1

    print(f"building from {len(games)} games...")
    _, report = build.build_many(
        games, cache_dir=args.pbp_cache, output=args.out, bref_cache=args.bref_cache
    )
    print(report.summary())
    if args.out:
        print(f"wrote {args.out}")
    return 0 if report.n_succeeded else 1


def _baseline(args: argparse.Namespace) -> int:
    """Fit the baseline ladder in each feature regime and print the comparison."""
    from rebounding.eval import baseline

    if not Path(args.frame).exists():
        print(f"no frame at {args.frame}; run `build --out {args.frame}` first", file=sys.stderr)
        return 1

    results, split = baseline.run(args.frame, l2=args.l2, with_forest=args.forest)
    print(split.summary())
    print()
    print(baseline.format_results(results))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    """Tracking-side summary for one game. No play-by-play, so no network."""
    tracking = sportvu.load(args.game)
    moments = tracking.moments

    print(f"game        : {tracking.game_id}")
    print(f"moments     : {len(tracking)} (dropped {moments.attrs['dropped_no_ball']} with no ball)")
    print(f"quarters    : {moments['Quarter'].nunique()}")
    print(f"rim arrivals: {int(moments['IsRimArrival'].sum())}")
    print(f"high starts : {int(moments['IsHighStart'].sum())}  <- dribbles and passes, not just shots")
    print(f"players     : {len(tracking.roles)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rebounding")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    build_cmd = sub.add_parser("build", help="build the training frame across games")
    build_cmd.add_argument("--games", default="data/7zips", help="directory of .7z archives, or one archive")
    build_cmd.add_argument("--out", default=None, help="parquet path to write")
    build_cmd.add_argument("--pbp-cache", default="data/pbp", help="directory of cached play-by-play json")
    build_cmd.add_argument(
        "--bref-cache",
        default="data/bref",
        help="directory of cached Basketball-Reference pages, used for any game not in --pbp-cache",
    )
    build_cmd.add_argument("--limit", type=int, default=None, help="only the first N games")
    build_cmd.set_defaults(func=_build)

    baseline_cmd = sub.add_parser("baseline", help="fit the baseline ladder and report accuracy")
    baseline_cmd.add_argument("--frame", default="data/frame.parquet", help="parquet from `build`")
    baseline_cmd.add_argument("--l2", type=float, default=1.0, help="L2 penalty on the logit weights")
    baseline_cmd.add_argument(
        "--forest", action="store_true", help="also fit a random forest (needs the 'models' extra)"
    )
    baseline_cmd.set_defaults(func=_baseline)

    inspect_cmd = sub.add_parser("inspect", help="tracking-only summary for one game")
    inspect_cmd.add_argument("--game", required=True)
    inspect_cmd.set_defaults(func=_inspect)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
