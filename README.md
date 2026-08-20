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
fixed and a test suite over a committed sample game. The rebounder model has been
rebuilt on top of it: 29.8% top-1 from the inputs the web app can actually supply,
against 26.8% for the model family it replaces.

**The rebounder is buildable in one command.** `python -m rebounding.cli train` fits
it and writes `FinalModel.pkl` — the model, the fitted `ShotPriors`, the served
feature list in fitted order, and the provenance to identify the file later. It takes
about eighteen seconds. The shipping fit uses train+val and scores **29.7% top-1** on
the untouched test games; every ladder number below is the train-only fit, which is
what keeps the regimes comparable.

`--regime` picks which of the project's three models to build. `served` is the web
app's, restricted to what the UI can supply; `all` is the feature-rich source of
truth, which scores **37.8%** and which `rebounding.serve` refuses to serve because
its features do not exist at prediction time. The movement model is the third and does
not live here yet. The 8.0 points between those two is its budget.

**The app calls one function.** `rebounding/serve.py::predict` takes ten placed
positions, which team is attacking and who shot, and returns a probability per player
in the caller's own order. It reproduces the training path's 27 features to 0.0 maximum
absolute difference on real shots, which is the property the web app never had: there
is now exactly one definition of every feature, and it lives here.

**The weights are deliberately not committed.** `.gitignore` still excludes `*.pkl`
and `*.h5`, and the artifact reaches the web app host by scp rather than through git.
That is a choice this time rather than the accident that lost the 2017 model, and it
is why the bundle carries its own provenance: `cli describe --model FinalModel.pkl`
answers "which model is on the box?" from the file itself. The movement model
(`posnn.h5`) and its normaliser (`msd.pkl`) survive only in the companion web app repo,
[riders994/ReboundWebApp](https://github.com/riders994/ReboundWebApp). Rebuilding that
half is the next phase.

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
    derived.py        relative and shot-context features, computed from the frame
    build.py          bulk build across games
  models/
    baselines.py      slot prior and nearest-to-rim, the floors
    conditional_logit.py  grouped softmax over the ten players
    boosted.py        the same loss with a tree ensemble for a score function
    forest.py         per-row random forest, reproducing the 2017 shape
  eval/
    split.py          chronological game-level train/val/test
    metrics.py        top-1, top-3, MRR, log loss
    baseline.py       the ladder: every model in every feature regime
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

**Pairing on the game clock let shots span a dead ball.** This one was a fault in the
*rebuilt* pipeline, not the old one, and it is the reason the corpus had a tail of
impossible shots. Release candidates were bounded by 0.3–3.0 s of **game clock**, and
the game clock stops: on a whistle, a timeout, a review. A release and a rim contact
either side of a stoppage can be thirty seconds apart in real time while the clock
shows two, so the bound accepted them and the shot was paired across the dead ball —
with the "shooter" standing at the far end of the floor in a previous possession.

Every pairing that put a shooter more than 60 ft from the rim was one of these. Over 90
games there were 13 of them, median wall-clock flight **36 s**, which the play-by-play
describes as layups and short jumpers from 1 to 26 ft. They were not full-court heaves;
there are none in the corpus, because a buzzer miss is never credited a rebound and so
never pairs.

The moments already carry a wall-clock `Timestamp`, so the fix is to apply the same
bound to real time (`MAX_WALL_FLIGHT_SECONDS`) and to derive `FlightTime` from it. The
game-clock interval stays recoverable as `ReleaseClock - RimClock`. For 97.8% of shots
the two agree within 0.1 s; where they disagree, the timestamp is the one telling the
truth about how long the ball was in the air.

| | before | after |
|---|---|---|
| shots in the corpus | 42,255 | 41,679 |
| max shot distance | 95.5 ft | **62.0 ft** |
| shooters > 60 ft | 118 | **2** |
| shooters > 40 ft | 139 | 17 |

940 shots are now dropped as `release_only_across_a_stopped_clock`; 576 of those had
been reaching training, the rest were already being caught downstream by the
substitution check. None were recovered — where the clock lies there is usually no
valid alternative release, so the shot is dropped rather than repaired. Accuracy is
unchanged within noise in every regime, which is the expected result for a fix that
removes 1.4% of the data: it buys correctness, not points.

**Flight time saturates, and no polynomial says so.** With those points gone,
`flight_hat`'s quadratic no longer had anything dragging its tail up: it peaked at
27.9 ft and fell away, predicting that a 48.7 ft shot — the furthest a user can place a
shooter on the app's canvas — hangs 1.46 s, less than a ten-footer, and going negative
past 92 ft. The measured curve climbs to 2.33 s by 30 ft and then flattens, because past
that range the shot is taken on a flatter, harder trajectory.

Raising the degree does not help, and is recorded here so nobody retries it: degrees 2,
3, 4 and 6 all sit within half a millisecond of each other on validation RMSE, every one
is non-monotone, and the higher ones are wilder in the tail — 5.17 s at 62 ft for the
quartic, 9.46 s for the sextic. Saturating forms (`log(1+d)`, `sqrt(d)`, `1 - exp(-d/k)`)
are monotone but fit no better and the first two keep climbing where the data flattens.
`predicted_flight` holds the peak instead, which is monotone, scores marginally *better*
on validation than the unclamped fit (0.4035 against 0.4037), and lands within 0.05 s of
the measured plateau.

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
movement from a pair of still photographs.

This was originally written down as the most likely single reason the movement model
underperformed. That guess is now measured and it was too strong. Velocity is worth
1.9 points of top-1 to the rebounder, and a player's release-time heading predicts his
rim-time position only about four tenths of a second into the flight — extrapolating
along it for the full flight is worse than assuming he never moves. Velocity was
missing and is worth having, but its absence does not account for the movement model's
behaviour; the L2 objective on a multimodal target does. See *Where the gains came
from* below.

## Modelling

### Results

`python -m rebounding.cli baseline --forest --on test`, on a chronological game-level
split (441 train / 95 val / 95 test games), team rebounds excluded, scored per shot as
top-1 of ten players. Feature selection and hyperparameters were chosen on validation;
the table below is the single read on the held-out test split.

| regime | what it knows | slot prior | nearest to rim | conditional logit | random forest | boosted softmax |
|---|---|---|---|---|---|---|
| **static** | ten positions | 24.0% | 23.2% | 26.4% | 26.8% | **27.6%** |
| **served** | + derived features | 24.0% | 23.2% | 27.7% | 29.2% | **29.8%** |
| release | + real velocity | 24.0% | 23.2% | 27.1% | 28.9% | 28.6% |
| release+derived | both | 24.0% | 23.2% | 29.7% | **30.9%** | 30.8% |
| rim | rim-time positions | 24.0% | 33.7% | 34.8% | 35.6% | **35.8%** |
| all | everything | 24.0% | 33.7% | 35.9% | **37.7%** | 37.4% |

`static` is what the web app can supply — a user places ten dots. `served` adds the
derived features, which need nothing further from the user. The two rows below them
add a release velocity the app has no way to collect, and are there to price it.

Four things fall out of this.

**The servable model went from 26.8% to 29.8%** without asking the UI for anything new.
Roughly a third of that is the model and two thirds the features; §*Where the gains
came from* below splits it.

**The train/serve gap is 6.0 points** (35.8% at rim time against 29.8% from what the
app can supply). That is the entire budget a movement model has to earn back. It was
8.2 points before this round of work and it shrinks every time the release-time model
improves, which is worth stating plainly: effort spent on the rebounder and effort
spent on the movement model are drawn from the same account, and the rebounder is the
cheaper of the two.

**Nothing separates the top three models by much.** Boosted softmax, random forest and
— with the derived features in hand — the conditional logit land within about 1.5
points of each other, and the boosted model does not win every row. On 5,690
validation shots an unpaired standard error is 0.6 points, so differences under about
1.2 points are not distinguishable at all; the comparisons quoted below use a paired
bootstrap over games, which is considerably tighter. The features moved the number.
The model choice mostly did not.

**The gap between top-1 and top-3 is large and stable** — 29.8% against 66.7% in the
served regime. The model is ranking sensibly and failing to separate the leaders,
which is what one would expect of an event with a genuinely random component. A tool
that narrows ten players to three is right two times in three.

**The 2017 figure of 86% top-1 does not survive contact.** Reproducing that model
family and scoring it three ways on the same data:

| metric | value |
|---|---|
| per-shot top-1 | 35.8% |
| per-row binary accuracy | 90.1% |
| per-row accuracy, predicting "nobody" for all ten | 90.0% |
| per-row AUC | 0.829 |

86% is not reachable as top-1 from these features, sits *below* the 90% that
predicting "nobody rebounds" scores by construction, and lands close to the 0.829
per-row AUC. The most likely reading is that it was a per-row metric — AUC or
accuracy over the ten-times-longer row table — rather than the per-shot question the
app actually asks. The 2017 code is gone, so this can't be confirmed directly; the
weights survive only in the web app repo. Treated as a per-row number it is
unremarkable, and nothing in the current data supports quoting 86% as top-1.

### Where the gains came from

**The features, mostly, and the relative ones above all.** Every feature in the
original pipeline described a player on his own — his coordinates, his distance to the
rim, his angle. Rebounding is a contest between ten people, so the quantities that
decide it are comparisons: who is inside whom, who has space, how a player ranks
against the other nine rather than how many feet he stands from the basket.
[`rebounding/data/derived.py`](rebounding/data/derived.py) adds those, and they are a
pure function of the ten positions, so the web app can compute every one of them from
what a user already places on the court.

The importance ranking makes the point better than the accuracy table does. In the
served regime the model puts 31% of its gain on `n_opp_inside` — how many opponents
are nearer the basket than this player — and another 11% on `dist_minus_mean`. Raw
`pre_dist`, the feature the old model leaned on, drops out of the top ten entirely
once the same information is present in relative form.

**The model change is real but second-order.** Three ways of getting a tree ensemble
to answer "which of these ten", on the same release-time features:

| model | validation top-1 |
|---|---|
| random forest, per-row binary, renormalised afterwards | 27.6% |
| LightGBM `lambdarank`, groups of ten | 28.0% |
| LightGBM, grouped softmax | 28.5% |

which is the ordering
[`conditional_logit.py`](rebounding/models/conditional_logit.py) predicts: a per-row
binary fit never sees the constraint that exactly one of the ten is the rebounder, and
`lambdarank` optimises a ranking surrogate rather than the likelihood. Matching the
loss to the metric is worth about a point.
[`boosted.py`](rebounding/models/boosted.py) is that grouped softmax with a tree
ensemble for a score function.

**Velocity is worth less than it looks, and says something.** Over a shot's flight
players travel 7.67 ft on average. Extrapolating each along his release velocity lands
8.46 ft from the truth — *worse than assuming he never moves*, at 8.41 ft. Damped by
the best-fitting factor of 0.38 it improves to 6.49 ft. Release-time velocity is worth
about four tenths of a second of movement and nothing after that, because what a
player does next is dominated by an intention his current heading does not reveal.
That is the strongest argument in this repo for a generative scene model over a
kinematic one, and equally the reason no cheap version of one will do.

### Out of scope on purpose: who the players are

The premise is predicting a rebound **from location data alone**, so per-player history
is excluded by design rather than by oversight. This is worth stating because it is not
a null result — it works.

A smoothed per-player historical rebound rate, fitted on the training games and applied
to the later ones, is worth **+1.2 points** of top-1 (29.7% → 30.9%). Only about 0.3 of
that is the position effect the model already has through `role`; the remaining ~0.9 is
individual, separating players who share a listed position. It is measuring real
rebounding — the highest fitted rates are Drummond (.338), Whiteside (.320) and DeAndre
Jordan (.308), the lowest Isaiah Thomas (.076), Redick (.072) and Wiggins (.072). It
also would not tie the app to this corpus, since any published rebound rate could be
fed at serving time.

It is still out, because "which of these ten players gets the board, knowing only where
they stand" is the question the project exists to answer. Adding a scouting prior
answers a different and easier one.

`role` — the listed-position ordinal — is the deliberate exception. It is an attribute
of the player rather than a location, and it is worth 2.1 points (3rd of 27 by gain),
but it was part of the original 2017 model concept and is kept for continuity with it.
If it is ever dropped it must be **removed and retrained**, never defaulted: serving a
placeholder 3.0 to a model trained on real positions scores 26.9%, worse than the 27.6%
of an honest model that never had the feature.

### What was tried and did not work

Recorded because the next person will otherwise try them again.

| idea | result |
|---|---|
| spatial target encoding: smoothed grid of "who rebounds from here", per shot-distance bucket and team | −0.02 pts. Boosting over `pre_x`, `pre_y`, `is_offense`, `shot_dist` already recovers it |
| neighbour context: attach the nearest opponent's and nearest teammate's own features to each row | −0.50 pts, not distinguishable from noise |
| hierarchical P(team) × P(player \| team) instead of one softmax over ten | −1.30 pts, and that one *is* significant |
| set model: permutation-equivariant net pooling over own team and opponents, 3 blocks | 30.5% against the boosted model's 30.7% — a tie |
| blending the boosted model with the conditional logit, or with the set model | +0.3 pts at best, inside the noise |
| measured flight time instead of predicted | +0.23 pts, not distinguishable — so the servable version costs nothing |

The set model is the interesting negative. It is the architecture the rebuild plan
proposes for the movement model, applied to the rebounder, and on 27k training shots
it matches gradient boosting rather than beating it. Depth tells the same story from
the other direction: 127 leaves scored a point *worse* than 63 at every learning rate
tried. The corpus cannot grow — see [`data/MANIFEST.md`](data/MANIFEST.md) — so this is
a standing constraint, not a tuning accident. Capacity is not what is missing.

What is missing is more likely information. Nothing in the current feature set knows
the ball: its arc, its entry angle, where it actually caroms. Those need a rebuild of
the frame from the tracking corpus rather than a better model, and only the parts of
them a user could plausibly specify are servable.

*What follows describes the 2017 work and what replaces it.*

Two models, chained. The **rebounder model** predicts who gets the board from where
everyone stands when the ball reaches the rim; a random forest was reported at 86%
top-1, which the section above shows was almost certainly a per-row metric. The
**movement model** exists because the web app can only ask a user for positions at
*release*, so something has to predict where players will be a second later.

The movement model never worked well. It was a Keras dense net doing deterministic
point regression from one snapshot to another, one row per player. Three problems:
no velocity input; an L2 objective on a multimodal target, which returns the
conditional mean and so drifts every player toward the paint; and no interaction
term, so nothing stopped two predicted players occupying the same square foot and
box-outs were unrepresentable.

There is also a train/serve skew that no amount of movement-model quality fixes: the
rebounder was trained and evaluated on ground-truth rim-time positions but served
predicted ones. That skew is now measured rather than argued about — 6.0 points of
top-1 — and the honest headline number is accuracy **from what the app can supply**,
29.8% for the train-only fit the ladder above compares and **29.7% for the weights
that actually ship**, both on the same held-out test games.

The rebounder half of the rebuild plan is done: the grouped softmax matches the
evaluation metric directly and removes the class-weight hacks the original needed, and
`features.to_tensor` emits the `(n_shots, 10, n_features)` form in canonical slot order
— offense then defense, each nearest-to-rim first — which is what lets any of these
models consume a whole shot at once.

What remains is the movement half: a scene-level conditional VAE with a
set-transformer encoder, sampling coherent futures rather than one averaged guess. Two
results above bear on it. Its accuracy budget is 6.0 points and shrinking, so it should
be justified as a UI feature — the predicted movement is what the app draws — rather
than as an accuracy device; that argument is made in
[`docs/webapp-handoff.md`](docs/webapp-handoff.md). And the fact that constant-velocity
extrapolation is worse than assuming nobody moves says the problem it has to solve is
real, not a matter of arithmetic on a heading.

## Credits

Front end by Hayden ([@ht44](https://github.com/ht44)), in
[riders994/ReboundWebApp](https://github.com/riders994/ReboundWebApp).
