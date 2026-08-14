# NBA Rebounding Model

### Rohan Vahalia, Galvanize G38DS Austin

Predicting who comes down with a rebound, from tracking data alone — no player
metadata, no reputation, just where everyone is and how they move.

## Overview

Inspired by a story about Dennis Rodman, who Isiah Thomas described in a
[2011 interview](http://www.businessinsider.com/dennis-rodman-basketball-genius-2014-3)
as studying the rotation of every teammate's shot:

>"He knew the rotation of every person that shot on our team — if it spins sideways,
>where it would bounce, how often it would bounce left or right. He had rebounding
>down to a science, and I never heard anyone think or talk about rebounding and
>defense the way he could break it down."

If Rodman could learn it, a model should be able to. This project models rebounds
purely from positional data, deliberately ignoring who the players are, to isolate
how much of rebounding is just position and movement.

## Status

The 2017 pipeline was Python 2 and no longer ran; the 2020 port on `rewrite-pd` was
abandoned half-finished. The extraction pipeline has been rebuilt as the
`rebounding` package on Python 3, with the correctness problems described below
fixed and a test suite over a committed sample game.

**The models are not in this repo.** The original `.gitignore` excluded `*.ip*` and
`*.pkl`, so the notebooks and weights were never committed here. The movement model
(`posnn.h5`) and its normaliser (`msd.pkl`) survive in the companion web app repo,
[riders994/ReboundWebApp](https://github.com/riders994/ReboundWebApp); the rebounder
(`FinalModel.pkl`) is referenced by that app's `webapp.py` but is not committed
anywhere. Rebuilding both is the next phase.

## Data

| source | what |
|---|---|
| STATS SportVU | player and ball positions at 25 Hz, 636 games of 2015-16 |
| Basketball-Reference play-by-play | shot and rebound events |

See [`data/MANIFEST.md`](data/MANIFEST.md). The SportVU feed was withdrawn in 2016
and the NBA moved to Second Spectrum in 2017-18, so this corpus cannot be extended —
about 60k rebounds is the hard ceiling, which constrains how large a model is
sensible.

The play-by-play source moved too. `stats.nba.com/stats/*` no longer answers: it
accepts the connection and holds it open until timeout, from a logged-in browser
session as readily as from a script. Its siblings are gone as well — `data.nba.net`
does not resolve and `cdn.nba.com`'s live feed 403s a 2015-16 game. Events now come
from Basketball-Reference via [`rebounding/data/bref.py`](rebounding/data/bref.py),
which emits the same frame the NBA feed did, so pairing and everything after it are
unchanged. `pbp.py` keeps the parsing and pairing but no longer fetches: its
`fetch()` reads the disk cache and raises on a miss.

One game — the sample game — still has a cached NBA payload from before the shutoff,
and `tests/test_bref.py` uses it to hold the two sources to each other. Over its 106
misses, shooter, shooting team, rebounding team, blocker and team-rebound flag agree
**100%**; clocks agree within one second; two rebounds (1.9%) are credited to
different players, both times to a teammate, which is a scoring difference between
the feeds rather than a parse error. End to end the replacement is marginally the
better source, because it states a shot distance on layups and dunks where the NBA
text omits one:

| | shots paired | median error | within 3 ft | distance stated |
|---|---|---|---|---|
| NBA feed | 78/106 | 1.24 ft | 82% | 94.3% |
| Basketball-Reference | 78/106 | 1.00 ft | 83% | 98.1% |

## Pipeline

```
rebounding/
  constants.py        court geometry, single source of truth
  data/
    sportvu.py        tracking JSON -> numpy arrays + ball features
    pbp.py            play-by-play -> shots paired with their rebounds
    bref.py           Basketball-Reference pages -> the same frame pbp.py returns
    court.py          folding full court onto the attacking half
    pairing.py        locating each play-by-play shot in the tracking data
    features.py       per-shot, per-player features in canonical slot order
    build.py          bulk build across games
  cli.py
```

```bash
uv venv && uv pip install -e ".[dev]"
pytest

python -m rebounding.cli inspect --game data/7zips/01.01.2016.CHA.at.TOR.7z
python -m rebounding.cli build --games data/7zips --out data/frame.parquet
```

`build` needs play-by-play. It reads a cached NBA payload from `--pbp-cache` when one
exists for that game and otherwise fetches from Basketball-Reference into
`--bref-cache` (default `data/bref`, gitignored). Fetching is rate limited to 20
requests a minute, which their terms ask for and they enforce with a temporary ban,
so the first full run over 636 games takes about 35 minutes. Later runs are offline.

## What was wrong with the old pipeline

Each of these is now covered by a regression test.

**Court folding destroyed left/right.** The whole of the orientation logic was
`abs(PlayerX - 47)`. Folding x alone is a *reflection*, so the same play run at
either end landed on opposite sides of the folded half court, superimposing
left-wing and right-wing shots. Worse, it was applied per player using that player's
own x, so the attacking basket was never actually determined: a defender who hadn't
crossed half court got mirrored into the front court and recorded 34.75 ft from the
rim when he was really 48.75 ft away. The fix is a 180° rotation for the left half,
with the basket resolved once per shot from the ball at the rim.

**Shot-to-rebound pairing was unvalidated and often wrong.** `findShot` searched
backward for "ball started rising", which fires on every dribble bounce, every pass
apex, and every rattle of the ball on the rim — 9,073 times in the sample game
against roughly 176 shots — with no bound on how far back it looked. When two shots
resolved to the same tracking frame, the `rimErr` fallback substituted an arbitrary
moment with no rim condition at all, and stored empty results that became NaN rows.
Nothing measured whether any of it worked.

It is measurable now, by comparing the tracking-derived release distance against the
distance stated in the play-by-play text. On the sample game's 95 unblocked misses:

| | paired | median error | within 3 ft |
|---|---|---|---|
| original logic, ported as-is | 70 | 10.06 ft | 29% |
| current | 78 | **1.31 ft** | **84%** |

**Other fixes.** 44% of shots lost their description, because only
`HOMEDESCRIPTION` was kept. 11% of rebounds — the team rebounds — were silently
discarded, biasing the training set toward clean uncontested boards. Transition
flags were encoded as "clock, or 0", so `RimStart >= t` matched an entire quarter
whenever a shot went up at 0:00. `.shift()` ran across quarter boundaries.
`massunpack` overwrote its own `run` method with a float and wrapped everything in
`except Exception: pass`, so a run that lost a third of the season looked identical
to one that lost nothing. The whole game's tracking data was held as Python lists
inside DataFrame cells.

**Velocity was never computed at all.** The pipeline extracted two isolated frames
per shot and nothing in between, so both models were being asked to reason about
movement from a pair of still photographs. This is the most likely single reason the
movement model underperformed.

## Modelling

*Being rebuilt. What follows describes the 2017 work and what replaces it.*

Two models, chained. The **rebounder model** predicts who gets the board from where
everyone stands when the ball reaches the rim; a random forest reached 86% top-1.
The **movement model** exists because the web app can only ask a user for positions
at *release*, so something has to predict where players will be a second later.

The movement model never worked well. It was a Keras dense net doing deterministic
point regression from one snapshot to another, one row per player. Three problems:
no velocity input; an L2 objective on a multimodal target, which returns the
conditional mean and so drifts every player toward the paint; and no interaction
term, so nothing stopped two predicted players occupying the same square foot and
box-outs were unrepresentable.

There is also a train/serve skew that no amount of movement-model quality fixes: the
rebounder was trained and evaluated on ground-truth rim-time positions but served
predicted ones, so 86% is a ceiling the live app never saw. The honest headline
number is top-1 accuracy **from release-time inputs**, with 86% quoted as the
ceiling and the gap stated.

The rebuild plan: a scene-level conditional VAE with a set-transformer encoder for
movement, sampling coherent futures rather than one averaged guess; and a grouped
softmax over the ten players for the rebounder, which matches the evaluation metric
directly and removes the class-weight hacks the original needed. `features.to_tensor`
already emits the `(n_shots, 10, n_features)` form in canonical slot order — offense
then defense, each nearest-to-rim first — which is what lets a plain logistic
regression or random forest consume a whole shot at once.

## Credits

Front end by Hayden ([@ht44](https://github.com/ht44)), in
[riders994/ReboundWebApp](https://github.com/riders994/ReboundWebApp).
