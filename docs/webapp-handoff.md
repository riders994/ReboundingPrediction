# ReboundWebApp: handoff for the rebuild

Written from `ReboundingPrediction` on 2026-08-14, for a separate project working on
[`riders994/ReboundWebApp`](https://github.com/riders994/ReboundWebApp).

The data pipeline behind the app has been rebuilt and the model has been re-measured
on the full 631-game corpus. This document records what that changed, what is
verifiably broken in the app today, and what the app has to do to match the pipeline
it is meant to serve.

Everything under "Verified" was checked directly against the repo on the date above:
`webapp.py` was read in full, and file existence was confirmed by HTTP status. Items
under "Needs checking" are inferences that a project with the repo checked out should
confirm before acting on.

---

## 1. The app cannot currently start

**Verified.** `webapp.py` line 3 of its startup block does:

```python
Model = joblib.load(open('./FinalModel.pkl', 'rb'))
```

`FinalModel.pkl` is **not in the repository** — `raw.githubusercontent.com/.../FinalModel.pkl`
returns **404**. The cause is in the repo's own `.gitignore`, whose entire contents are:

```
*.pkl
```

So the rebounder model — the thing the whole app exists to serve — was never
committed. What did survive:

| file | status | what it is |
|---|---|---|
| `posnn.h5` | present, 2.6 MB | Keras movement model (release positions → rim positions) |
| `msd.pkl` | present, 563 B | normalisation constants (mean/std) for the movement model's input |
| `FinalModel.pkl` | **404 — lost** | the random forest that actually predicts the rebounder |

`msd.pkl` survives despite the ignore rule, so it was committed before the rule
existed or force-added. `FinalModel.pkl` was not so lucky. Rohan believes a copy may
exist on an old EC2 box.

**Recovering it is not worth much even if the EC2 copy turns up.** It was trained on
the old pipeline, whose court folding used `abs(x - 47)` — a reflection rather than a
rotation — so it learned from coordinates where left-wing and right-wing shots were
superimposed. A recovered `FinalModel.pkl` would be a model fitted on mirrored
features, and serving it correctly would mean reproducing the bug on the app side.
Retrain from the current pipeline instead; §5 gives the contract. Recover the file
only if you want it as a historical baseline to compare against.

---

## 2. Nothing in the app runs on a modern stack

**Verified**, from the source:

| what | where | why it breaks |
|---|---|---|
| `print 'received'` | `predict()` | Python 2 syntax, `SyntaxError` on any Python 3 |
| `import cPickle as pickle` | imports | removed in Python 3 |
| `xrange(5)` | `boxgen` | removed in Python 3 |
| `d.keys()[0]` | `predict()` | Python 3 `dict_keys` is not subscriptable |
| `from sklearn.externals import joblib` | imports | removed in scikit-learn 0.23 |
| `from keras.layers.core import ...` | imports | moved in Keras 2.4 / TF 2.x |

There is also **no `requirements.txt`, no `Procfile`, and no pinned environment**
(all confirmed 404). The 2016 dependency set cannot be reconstructed from the repo.

Treat the port as a rewrite of `webapp.py`, not a 2to3 pass. It is roughly 130 lines
and most of the feature code should be deleted rather than ported — see §5.

---

## 3. Correctness bugs, with the pipeline's ground truth

Each of these is a place where the app computes a feature differently from the
pipeline that trained the model. Every one of them is train/serve skew: the model was
fitted on one definition and served another.

### 3.1 The hoop is in the wrong place — **verified**

```python
hoop = np.array([41.65, 25])          # webapp.py, in features() and boxgen()
```

The correct folded rim is `(41.75, 25)`. From `rebounding/constants.py`:

```python
HOOP = (HALF_COURT_X - RIM_INSET, CENTER_Y)  # (41.75, 25.0)
# The original code used 41.65, which is 0.1 ft short.
```

Small, but it shifts every distance and angle feature the model consumes, and it is
free to fix.

### 3.2 The angle convention has its arguments swapped — **verified**

```python
angle = np.arctan2(diff[:,0], diff[:,1])   # webapp.py
```

`arctan2` takes `(y, x)`. Passing `(dx, dy)` measures the bearing off the sideline
axis instead of off the baseline axis. The pipeline's `rebounding/data/court.py`
documents this exact bug and fixes it:

```python
return np.arctan2(dy, -dx)   # -dx so "out from the basket" is the zero bearing
```

In the old code this was *self-consistent* — the angle was only ever used inside
`cos(angle - shooter_angle)`, so a rotated frame cancels. It matters now because the
new features use the signed angle directly, and because the sign is what distinguishes
the two sides of the floor.

### 3.3 The movement direction feature is sign-inverted — **verified**

```python
closer = ((self.pre['HDist'] < self.pos['HDist']).astype(int) - 0.5) * 2
self.pos['MoveV'] = np.sqrt((move ** 2).sum(axis = 1)) * closer
```

`pre_HDist < pos_HDist` is true when the player ends up **farther** from the rim, so
the variable named `closer` is `+1` for moving *away*. The pipeline's equivalent:

```python
"closed_on_rim": np.where(pos_dist < pre_dist, 1.0, -1.0)   # features.py
```

`+1` for moving *toward*. The two are exact opposites, so `MoveV` carries the wrong
sign on every player.

### 3.4 Rendered positions are transposed against modelled positions — **verified**

```python
self.pos['x']    = self.posArr[:,0]
self.pos['y']    = self.posArr[:,1]
self.pos['newy'] = self.posArr[:,0]     # <- newy gets column 0
self.pos['newx'] = self.posArr[:,1]     # <- newx gets column 1
```

`x`/`y` feed the model; `newx`/`newy` are what `Modeling()` returns and what the
browser draws. They are assigned from opposite columns. **The predicted positions
drawn on screen are transposed relative to the ones the prediction was computed
from.**

This is the most damaging bug in the app, because the drawn movement is the product
(§5) — not a debug overlay. Whatever the movement model's real quality, the user has
never seen it: they have been watching its output reflected about the line `x = y`.
Any judgement anyone has formed about how good the movement predictions look is
based on transposed coordinates. Fix this before evaluating `posnn.h5` at all.

### 3.5 There is no court folding at all — **needs checking**

The app takes canvas coordinates and computes distance to a single hardcoded hoop.
The training data is *folded* onto one half court by a 180° rotation, with the
attacking basket resolved per shot from the ball at the rim. The old pipeline folded
with `abs(x - 47)`, which is a reflection, not a rotation — so left-wing and
right-wing shots were superimposed.

What needs confirming with the repo checked out: what coordinate frame `public/script.js`
emits, its origin and axis directions, and whether it matches the folded frame the
pipeline produces. If the handedness is flipped, every left/right-asymmetric feature
is mirrored at serve time. `rebounding/data/court.py::describe_side_convention` exists
to make this checkable — feed it a known position and see which side it reports.

### 3.6 Feature order is positional and undocumented — **verified**

```python
self.modIn = np.concatenate([self.pre.values, self.pos[[...]].values], axis=1)
```

`self.pre.values` takes whatever column order pandas happens to hold, which depends
on the insertion order inside `features()`. Both `msd.pkl`'s normalisation constants
and the model's feature indices are bound to that incidental order. Any edit that
adds or reorders a column silently corrupts every prediction, with no error.

Replace with an explicit named list. §5 gives it.

### 3.7 Minor — **verified**

- `inputDecode(model='Model')` passes the **string** `'Model'`, not the model. It only
  works because `Modeling(fitModel=Model)` overwrites `self.model` before use.
- `requests`, `random`, `Sequential`, `Dense`, `Dropout`, `Activation`, `Flatten`,
  `SGD`, `Adadelta`, `Adagrad`, and `RandomForestClassifier` are all imported and
  never used.
- `test = res.values` in `predict()` is dead.
- `CORS(app)` is unrestricted.

---

## 4. The accuracy claim has to change

The app's lineage carries a headline of **86% top-1**. That number does not survive
re-measurement, and the site should not quote it.

Reproducing the same model family (random forest) on the rebuilt 631-game corpus,
scored three ways on a held-out chronological split:

| metric | value |
|---|---|
| per-shot top-1 — *the question the app asks* | **35.8%** |
| per-row binary accuracy | 90.1% |
| per-row accuracy predicting "nobody" for all ten players | **90.0%** |
| per-row AUC | 0.829 |

86% is not reachable as top-1 from these features; it sits *below* the 90% that
predicting "nobody rebounds" scores by construction, and lands beside the 0.829
per-row AUC. The most likely reading is that it was a per-row metric over the
ten-times-longer row table, not a per-shot one. The 2017 code is gone, so this cannot
be confirmed directly — state it as the reading the arithmetic supports.

**The honest number for the app is top-1 from what a user can actually supply: 30.1%**
on the held-out test split. Against a 10-player field, chance is 10% and the positional
prior alone is 23.8%.

That is the score of the weights in `FinalModel.pkl`, and it is the figure to put on
the site. The ladders in this document all read 29.2%, which is the same model fitted
on the training games only — the comparison basis that keeps the regimes honest
against each other. The shipped artifact adds the 95 validation games to the fit once
the hyperparameters are settled, which is worth 0.9 points. The test games are held
out of both. Quote 30.1%: it belongs to the model actually being served.

That figure moved after this brief was first written. It was 27.6%, from a random
forest on release-time features. Two changes since, both in the parent repo and
neither requiring anything new from the UI:

| step | test top-1 |
|---|---|
| positions only, per-row random forest (the old shape) | 26.7% |
| positions only, grouped-softmax gradient boosting | 27.5% |
| **+ relative and shot-context features (`rebounding/data/derived.py`)** | **29.2%** |

The features are the larger half of that, and the important thing about them is that
they are a **pure function of the ten dots the user already places** — who is inside
whom, how far each player is from the rim relative to the other nine, where each stands
relative to the shot. Nothing new is asked of the user; the serving code computes them
from the same input it already has. §5 gives the list.

If the UI displays a confidence or accuracy figure anywhere, it needs to become 29%
top-1, phrased as "picks the right rebounder just under 3 times in 10".

---

## 5. What the app should serve instead

### The movement model stays — it is a product feature, not an accuracy device

The app's whole interaction is: place ten players at release, press Run, watch what
happens. The predicted movement **is** the thing being shown. So `posnn.h5` (or its
replacement) is required regardless of what it does for accuracy, and the Keras
dependency stays in the stack either way.

That reframes it rather than removing it. Two decisions that look like one:

**Decision A — what the rebounder consumes.** Independent of the visual. Measured on
the rebuilt corpus, test split:

| regime | best top-1 | meaning |
|---|---|---|
| what the app can supply, with derived features | 29.2% | ten static dots |
| the same plus real release velocity | 31.1% | needs a UI the app does not have |
| rim-time features | 36.3% | ceiling; requires knowing the future |

A *perfect* movement model buys **7.1 points** over what the app can serve today. That
budget has shrunk twice — it was 8.2 points when this brief was written, then 5.3
against the velocity-carrying variant — because the release-time model keeps
improving while the ceiling barely moves. Every point added to the served model is a
point subtracted from the movement model's justification.

There is a second, blunter measurement pointing the same way. Over a shot's flight,
players travel 7.67 ft on average. Extrapolating each player along his release velocity
lands **8.46 ft** from the truth — *worse than assuming he never moves*, which lands
8.41 ft away. Damping the step to the best-fitting 0.38 brings it to 6.49 ft. So
release-time velocity is worth about four tenths of a second of NBA movement and
nothing beyond that; a player's next second is dominated by an intention the velocity
does not reveal. Anything that predicts rim-time positions well has to model that
intention, which is the argument for a generative scene model rather than a kinematic
one — and equally the reason a cheap version of it will not do.

The current movement model is biased in a known
direction, so feeding its output to the rebounder plausibly scores *below* the 29.2%
that ignoring it entirely gets. **Recommendation: run the rebounder on release-time
features**, and let the movement model drive the animation beside it rather than
upstream of it. If a later movement model measurably beats release-time features on
held-out data, move it back into the path then — that is a one-line swap, and §7 step 4
is where to test it.

**Decision B — the quality bar for the movement model.** This is where it gets more
demanding, not less. As a hidden intermediate, a mediocre movement model costs a few
points of an already-imperfect metric. As the visible output, its failure modes are
things a user watches happen:

- **L2 on a multimodal target returns the conditional mean.** A player who might crash
  the offensive glass or might leak out in transition gets drawn splitting the
  difference — drifting toward the paint, which is where the average of those two
  futures lies. Every player drifting paint-ward is immediately legible as wrong to
  anyone who has watched basketball.
- **No interaction term.** Nothing stops two predicted players occupying the same
  square foot. Overlapping dots read as a broken app, and box-outs — the thing the
  model is supposed to illustrate — are literally unrepresentable.

So the scene-level conditional VAE in the rebuild plan is motivated by **the UI**, not
by the 7.1 points. It samples coherent whole-scene futures instead of averaging them,
which is exactly what a visual needs.

It also unlocks something the current model cannot do: sampling several futures and
drawing the *spread* — a cloud or a set of ghosts per player — is more honest than one
confident dot, and communicates uncertainty a coach would want to see. A generative
model gives that for free; a point regression cannot express it at all.

**Evaluate it accordingly.** Do not tune the movement model on L2 error — that metric
is what produces the drift. Judge it on whether real rim-time positions fall inside the
predicted distribution (calibration), and on whether sampled scenes are physically
plausible: no overlapping players, speeds within human range, box-out relationships
preserved.

### The feature contract

Train and serve exactly these, in this order. This is
`rebounding/data/features.py::SERVED_FEATURES`, and the app should import that list
rather than retype it:

```
pre_x, pre_y, pre_dist, pre_angle, pre_cos_shooter, pre_box,
is_offense, is_shooter, role,                                    # positions
d_nearest_opp, d_nearest_any, n_within_6, n_within_10,
inside_gap, opp_boxes_me, n_opp_inside, n_team_inside,
dist_minus_best, dist_minus_nearest_teammate, dist_minus_mean,
closeness_share,                                                 # the contest
flight_hat, shot_dist, rel_bearing, abs_rel_bearing,
sin_shooter, dist_x_shot                                         # the shot
```

Notes for the serving side:

- **The eighteen derived columns need no new input.** Every one is computed from the
  ten positions, which team is attacking, and who shot — the app has all three
  already. Call `rebounding.data.derived.ShotPriors.transform` on the same frame the
  first nine features come from. They are worth 1.7 points of top-1, which is more
  than the model change was.
- `flight_hat` is a *predicted* flight time, from a quadratic in shot distance fitted
  on the training games and carried on the `ShotPriors` object. Do not substitute a
  measured flight time — at serving time the shot has not landed yet, and the model
  was trained on the prediction.
- **Velocity is out, and that decision still holds.** `pre_vx`, `pre_vy`, `pre_speed`
  are absent above because a user places static dots. Re-measured with the derived
  features in place, on the test split:

  | feature set | boosted softmax |
  |---|---|
  | with velocity | 31.1% |
  | without (the list above) | 29.2% |

  Roughly 1.9 points, still not worth asking a user to drag a direction vector for
  each of ten players. Whatever else happens, **do not pass zeros for velocity to a
  model trained with real ones** — that is train/serve skew, and strictly worse than
  the honest no-velocity model.
- `role` is an ordinal listed position from `constants.POSITION_MAP`, defaulting to
  `3.0` when unknown.
- `pre_box` is the box-out count from `features.boxgen`, which asserts a `(10, 2)`
  shape rather than silently miscounting on a bad split — the old `boxgen` did not.
- **Player order matters.** The pipeline sorts offense first, then defense, each
  ordered nearest-to-rim *at release*. The app must reproduce that ordering exactly
  before building the input array.

### The output shape

The current app takes per-row random-forest probabilities and divides by their sum.
The rebuilt model is a **grouped softmax over the ten players** — probabilities
already sum to 1 across the shot by construction, and no renormalisation is needed or
wanted. The model to serve is `rebounding/models/boosted.py::BoostedSoftmax`, which is
LightGBM under that same grouped loss;
`rebounding/models/conditional_logit.py` is the linear version of the same thing and
is the fallback if a LightGBM dependency is unwelcome in the serving image, at a cost
of about 1.6 points.

### Loading the artifact

`FinalModel.pkl` is a `rebounding.models.artifact.ModelArtifact`, not a bare
estimator. It holds the model, the **fitted `ShotPriors`**, the served feature list in
fitted order, and a provenance block. Load it through the package rather than with a
bare `joblib.load`, which will give a clearer error when a dependency is missing:

```python
from rebounding.models.artifact import load

artifact = load("FinalModel.pkl")
probabilities = artifact.predict_proba(x)   # (n_shots, 10, 27) -> (n_shots, 10)
```

Three things follow from the shape of that bundle:

- **The priors travel with the model.** `flight_hat` and the shot-context features
  need the damping constant and the flight-time regression fitted on the training
  games. Use `artifact.priors.transform(rows)`, never a freshly constructed
  `ShotPriors` — an unfitted one raises, and one refitted on serving data would be a
  different feature.
- **`artifact.features` is authoritative, not the imported `SERVED_FEATURES`.** It is
  the order the trees were fitted in. If the package moves on and the deployed bundle
  does not, `load` warns; build the input array from `artifact.features` and that
  drift cannot silently reorder your columns. This is the fix for §3.6.
- **The serving host needs `rebounding` importable**, plus lightgbm. The bundle
  references its classes by import path, so unpickling imports them.

`python -m rebounding.cli describe --model FinalModel.pkl` prints the commit, the
corpus hash, what it was fitted on and what it scored. Since the weights are not in
git, that block is the only way to tell which model is on the box — worth capturing in
the deploy log after each scp.

### The serving entry point — use this instead of building features in the app

`rebounding/serve.py` turns ten placed dots into probabilities. **This replaces
`features()` and `boxgen()` in `webapp.py` entirely**, and it is what step 2 of §7 means
by importing the pipeline's versions. Do not reimplement folding, slot ordering or
box-out counts in the app.

```python
from rebounding.models.artifact import load
from rebounding.serve import predict, PlacementError

artifact = load("FinalModel.pkl")
prediction = predict(artifact, players)      # players: ten dicts, straight from JSON

prediction.probabilities   # np.ndarray, indexed exactly as `players` was
prediction.ranked()        # [(input index, probability), ...] most likely first
prediction.most_likely()   # index into `players`
```

Each player is `{"x": …, "y": …, "is_offense": bool, "is_shooter": bool,
"position": "G"|"F-C"|… }`, with optional `player_id`. Coordinates are the **folded**
half-court frame — `x` 0 at half court to 47 at the baseline, `y` in `[0, 50]`, rim at
`(41.75, 25)` — which is the frame `webapp.py` was already almost using with its
`41.65` hoop. Pass `basket="left"|"right"` instead if you are handing over full-court
coordinates.

Four things worth knowing:

- **The result is indexed the way you passed the players in.** The ten are reordered
  internally into canonical slots to be scored and the probabilities are mapped back,
  with `prediction.slots` recording where each one went. You never have to think about
  slot order — and you must not assume the output is in it.
- **Supply `position` if the UI can.** Leaving it unset defaults every player to `3.0`
  and costs about **1.4 points of top-1**, measured on the test split. That is a bigger
  loss than several of the fixes in §3 are worth gaining.
- **Invalid placements raise `PlacementError`**, not a bad prediction: nine players,
  six on offense, two shooters, a defensive shooter, or coordinates that look like an
  unfolded frame. Catch it and surface it to the user — the pipeline drops such shots
  rather than featurising them, so there is no sensible answer to return.
- **An artifact wanting velocity is refused outright.** If the served feature list ever
  grows `pre_vx`, `v_radial` or an `ext_*` column, `predict` raises rather than quietly
  passing zeros, because zeros to a velocity-trained model is the skew §5 warns about.

Verified against the pipeline: replaying real shots from the corpus through
`serve.feature_frame` reproduces all 27 columns the training path computed to **0.0
maximum absolute difference**, and scoring test shots through `predict` matches the
tensor path exactly. There is no train/serve skew left in the feature layer.

`python -m rebounding.cli predict --model FinalModel.pkl --players players.json` runs
one shot from the command line, which is the quickest way to confirm a freshly scp'd
artifact works on the host before pointing the app at it.

---

## 6. Deployment

- `app.run(host='0.0.0.0', port=80)` is the Flask **development server**. The parent
  repo's last commit message ("Need to add GUnicorn capabilities to load balance")
  says this was already known. Use gunicorn behind a real proxy.
- `serve_static` builds paths with `os.path.dirname(os.getcwd())` and a hardcoded
  `'ReboundWebApp'` segment, so it breaks under any working directory but one. Flask's
  `static_folder` does this properly.
- Front end is plain JS with **d3 v4** loaded from `d3js.org` — pinned to a CDN, no
  build step, no `package.json`. Fine to leave alone for v1; worth knowing it is
  unversioned.

---

## 7. Suggested order

1. ~~**Retrain a rebounder.**~~ **Done — the artifact exists.** `python -m
   rebounding.cli train` in the parent repo writes `FinalModel.pkl` (2.53 MB) and it
   reaches the app host by **scp**, not through git: `*.pkl` stays ignored by choice.
   It is a bundle, not a bare estimator — see "loading the artifact" at the end of §5,
   and do not expect a plain `RandomForestClassifier` at the other end of the load.
2. **Rewrite `webapp.py` for Python 3** around the explicit feature contract in §5.
   Delete `features()` and `boxgen()` from the app and call `rebounding.serve.predict`
   instead — it exists now, it is tested against the pipeline's own output, and it
   leaves exactly one definition of every feature. See "the serving entry point" in §5.
3. **Fix §3.4** (the transposed render) — one line, and it gates step 4.
4. **Look at `posnn.h5` honestly, for the first time.** With the transpose fixed and
   the coordinate frame confirmed, run some real plays through it and watch. It may be
   adequate as a visual, or the paint-drift may be obvious on sight. That observation
   decides whether the CVAE rebuild is urgent or can wait — and it is unavailable until
   step 3 lands.
5. **Confirm the coordinate frame** (§3.5) with a known-position round trip before
   trusting any number or any drawn position the app produces.
6. **Correct the accuracy copy** (§4).
7. **Deploy under gunicorn** (§6).

Steps 1–2 make the app runnable; 3 and 5 make it correct; 4 tells you how much
modelling work is actually left; 6 makes it honest.

---

## 8. Where things live

| what | where |
|---|---|
| **the serving entry point the app calls** | **`rebounding/serve.py::predict`** |
| the deployable bundle, and `load` | `rebounding/models/artifact.py` |
| feature definitions, regimes, slot ordering | `rebounding/data/features.py` |
| the served feature list | `rebounding/data/features.py::SERVED_FEATURES` |
| relative and shot-context features, and the fitted priors | `rebounding/data/derived.py` |
| court geometry, folding, `HOOP`, `rim_angle` | `rebounding/data/court.py`, `rebounding/constants.py` |
| the model to serve | `rebounding/models/boosted.py` |
| its linear fallback | `rebounding/models/conditional_logit.py` |
| baseline numbers reproduced | `python -m rebounding.cli baseline --forest` |
| the same on the held-out test split | `python -m rebounding.cli baseline --on test` |
| built training frame | `data/frame.parquet` (gitignored, rebuild with `cli build`) |
| the shipped weights | `FinalModel.pkl` (gitignored by choice, built with `cli train`, deployed by scp) |

One caveat worth carrying: the two data sources behind all of this now disagree on
1.9% of rebounds — the NBA feed and Basketball-Reference credit some boards to
different players on the same team. That is label noise below the level any of these
decisions turn on, but it is a floor on achievable accuracy.
