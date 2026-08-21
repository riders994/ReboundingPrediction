"""Command line entry points.

    python -m rebounding.cli build          --games data/7zips --out data/frame.parquet
    python -m rebounding.cli train          --out FinalModel.pkl
    python -m rebounding.cli train-movement --out MovementModel.pkl
    python -m rebounding.cli predict        --players players.json
    python -m rebounding.cli inspect        --game data/7zips/01.01.2016.CHA.at.TOR.7z

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
from rebounding.data.features import FEATURE_REGIMES
from rebounding.models import artifact


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

    results, split = baseline.run(
        args.frame,
        l2=args.l2,
        with_forest=args.forest,
        regimes=args.regimes.split(",") if args.regimes else None,
        on=args.on,
    )
    print(split.summary())
    print()
    print(baseline.format_results(results))
    return 0


def _train(args: argparse.Namespace) -> int:
    """Fit the deployable model and write the bundle the web app loads."""
    if not Path(args.frame).exists():
        print(f"no frame at {args.frame}; run `build --out {args.frame}` first", file=sys.stderr)
        return 1

    movement = None
    if args.regime == "served+movement":
        if not Path(args.movement).exists():
            print(
                f"no movement model at {args.movement}; run `train-movement` first",
                file=sys.stderr,
            )
            return 1
        movement = artifact.load_movement(args.movement)

    built = artifact.build(
        args.frame, fit_on=args.fit_on, regime=args.regime, movement=movement
    )
    out = args.out or (
        artifact.DEFAULT_OUTPUT if args.regime == "served" else Path(f"{args.regime}-model.pkl")
    )
    path = artifact.save(built, out)
    print(built.describe())
    print()
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    # The weights are gitignored by choice, so the only record of what shipped is the
    # file itself. Say where it has to go rather than leaving it in the working tree.
    if args.regime == "served":
        print(f"scp it to the web app host; `describe --model {path}` prints this block again")
    else:
        print(f"reference model, not servable; `describe --model {path}` prints this block again")
    return 0


def _train_movement(args: argparse.Namespace) -> int:
    """Fit the movement model -- the one the app animates, and the rebounder's feeder."""
    from rebounding.models.movement import MovementConfig

    if not Path(args.frame).exists():
        print(f"no frame at {args.frame}; run `build --out {args.frame}` first", file=sys.stderr)
        return 1

    config = MovementConfig(head=args.head, epochs=args.epochs, seed=args.seed)
    built = artifact.build_movement(args.frame, fit_on=args.fit_on, config=config)
    path = artifact.save(built, args.out or artifact.DEFAULT_MOVEMENT_OUTPUT)
    print(built.describe())
    print()
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    print(
        "scp it alongside FinalModel.pkl. To fold it into the rebounder as well, "
        f"`train --regime served+movement` reads it back from {path}."
    )
    return 0


def _describe(args: argparse.Namespace) -> int:
    """Print a bundle's provenance -- what is this file, and what did it score?"""
    if not Path(args.model).exists():
        print(f"no artifact at {args.model}", file=sys.stderr)
        return 1
    print(artifact.load(args.model).describe())
    return 0


def _predict(args: argparse.Namespace) -> int:
    """Score one placed shot from a JSON file. A smoke test for a deployed artifact."""
    import json

    from rebounding.serve import PlacementError, predict

    for label, path in (("artifact", args.model), ("players", args.players)):
        if not Path(path).exists():
            print(f"no {label} at {path}", file=sys.stderr)
            return 1

    players = json.loads(Path(args.players).read_text())
    try:
        prediction = predict(artifact.load(args.model), players, basket=args.basket)
    except PlacementError as exc:
        print(f"cannot score these placements: {exc}", file=sys.stderr)
        return 1

    print(f"{'rank':>4} {'input':>5} {'slot':>4} {'p':>7}")
    for rank, (index, probability) in enumerate(prediction.ranked(), start=1):
        print(f"{rank:>4} {index:>5} {prediction.slots[index]:>4} {probability:>7.1%}")
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
    baseline_cmd.add_argument(
        "--regimes",
        default=None,
        help="comma-separated subset of release, release+derived, served, rim, all",
    )
    baseline_cmd.add_argument(
        "--on",
        default="val",
        choices=["val", "test"],
        help="which split to score on; test is for the final read, not for choosing",
    )
    baseline_cmd.set_defaults(func=_baseline)

    train_cmd = sub.add_parser("train", help="fit the deployable model and write FinalModel.pkl")
    train_cmd.add_argument("--frame", default="data/frame.parquet", help="parquet from `build`")
    train_cmd.add_argument(
        "--out",
        default=None,
        help=(
            "where to write the bundle; gitignored by design and deployed by scp. "
            f"Defaults to {artifact.DEFAULT_OUTPUT} for the served regime, "
            "<regime>-model.pkl otherwise"
        ),
    )
    train_cmd.add_argument(
        "--fit-on",
        default="train+val",
        choices=list(artifact.FIT_CHOICES),
        help="train+val is the shipping fit; train reproduces the README ladder",
    )
    train_cmd.add_argument(
        "--regime",
        default="served",
        choices=sorted(FEATURE_REGIMES),
        help=(
            "which of the models to build: 'served' is the web app's, restricted to what "
            "the UI can supply; 'all' or 'rim' is the feature-rich source of truth, which "
            "cannot be served"
        ),
    )
    train_cmd.add_argument(
        "--movement",
        default=str(artifact.DEFAULT_MOVEMENT_OUTPUT),
        help="movement bundle to generate the extra columns of the served+movement regime",
    )
    train_cmd.set_defaults(func=_train)

    movement_cmd = sub.add_parser(
        "train-movement", help="fit the movement model and write MovementModel.pkl"
    )
    movement_cmd.add_argument("--frame", default="data/frame.parquet", help="parquet from `build`")
    movement_cmd.add_argument("--out", default=None, help=f"defaults to {artifact.DEFAULT_MOVEMENT_OUTPUT}")
    movement_cmd.add_argument(
        "--head",
        default="cvae",
        choices=("point", "mixture", "cvae"),
        help=(
            "cvae samples whole coherent scenes and is what the UI should draw; "
            "mixture is multimodal but per player; point is the 2017 shape, kept to "
            "measure what averaging a bimodal target costs"
        ),
    )
    movement_cmd.add_argument("--epochs", type=int, default=120)
    movement_cmd.add_argument("--seed", type=int, default=0)
    movement_cmd.add_argument(
        "--fit-on",
        default="train",
        choices=list(artifact.FIT_CHOICES),
        help="train keeps the validation games to stop on; train+val carves a stopping slice instead",
    )
    movement_cmd.set_defaults(func=_train_movement)

    describe_cmd = sub.add_parser("describe", help="print a built artifact's provenance and scores")
    describe_cmd.add_argument("--model", default=str(artifact.DEFAULT_OUTPUT))
    describe_cmd.set_defaults(func=_describe)

    predict_cmd = sub.add_parser("predict", help="score one placed shot from a json file")
    predict_cmd.add_argument("--model", default=str(artifact.DEFAULT_OUTPUT))
    predict_cmd.add_argument(
        "--players",
        required=True,
        help="json list of ten {x, y, is_offense, is_shooter, position} objects",
    )
    predict_cmd.add_argument(
        "--basket",
        default=None,
        choices=["left", "right"],
        help="give this only for full-court coordinates; omit for folded half-court ones",
    )
    predict_cmd.set_defaults(func=_predict)

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
