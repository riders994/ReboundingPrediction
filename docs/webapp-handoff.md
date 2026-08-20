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

### 3.4 Rendered positions are transposed against modelled positions — **withdrawn**

```python
self.pos['x']    = self.posArr[:,0]
self.pos['y']    = self.posArr[:,1]
self.pos['newy'] = self.posArr[:,0]     # <- newy gets column 0
self.pos['newx'] = self.posArr[:,1]     # <- newx gets column 1
```

`x`/`y` feed the model; `newx`/`newy` are what `Modeling()` returns and what the
browser draws. They are assigned from opposite columns.

**This section was wrong, and acting on it would break the app.** It is not a
transposition bug. It is the conversion from model coordinates back to canvas
coordinates, and the two frames genuinely do have their axes swapped — see §3.5. `x`
and `y` stay in the model frame because they are about to be fed back through
`features()`; `newx` and `newy` are the same points in the frame the browser draws in.
The matching swap on the way in is the innocuous-looking line

```python
df.columns = ['Off', 'isShoot', 'y', 'x']
```

which renames the incoming `x` column to `y` and vice versa. Both swaps are correct and
they compose to the identity. Reversing either one — as the original version of this
section instructed — introduces exactly the reflection about `x = y` that it warned
about.

**A real trap does live in that line, for anyone porting it.** It works only because
Python 2's pandas sorted dict keys alphabetically, giving `isOffense, isShooter, x, y`.
Modern pandas preserves insertion order, so the same statement now names the incoming
`x` column `Off` and `y` column `isShoot`. It produces nonsense silently, with no
error. The port replaces the whole thing with an explicit conversion in
`webapp_port/coordinates.py`.

There is also a genuine bug nearby that this section missed. The original `boxgen`
builds `dbox` from `o`, whose values index the *second* five-player block, then
concatenates it first — so each team is assigned the other team's box-out counts.
`rebounding/data/features.py::boxgen` does not have this problem. The 2017 movement
model was trained on the broken version, so the port reproduces it deliberately as that
model's input contract and nowhere else.

### 3.5 The canvas frame — **resolved, except for handedness**

`public/script.js` draws a 500×470 SVG and posts `xy / 10`, so it emits feet: `x` in
`[0, 50]` across the screen and `y` in `[0, 47]` down it. The canvas shows **only the
basket half of the court, with the basket at the bottom** (confirmed by Rohan).

That fixes the frame completely on the length axis. The model's `x` is the 47 ft
half-court length and its `y` is the 50 ft width, so **screen-across is model `y` and
screen-down is model `x`** — the ranges admit no other reading. SVG `y` grows downward
and the basket is at the bottom, so canvas-down and model-`x` agree with no flip.

The width axis is settled too. SportVU is a true bird's-eye view of the whole floor —
which is *why* the data has to be folded at all, both baskets being recorded — so the
model frame carries real-world chirality, and `court.fold` folds with a 180° rotation,
which preserves it. The canvas swap preserves it as well, though the arithmetic looks
like it should not: `(cx, cy) → (cy, cx)` has determinant −1, but SVG `y` points down,
so the canvas is already left-handed as drawn and the two reversals cancel.

**And the question turns out not to matter.** Mirroring every shot in the test split
about the length axis and rescoring gives 30.1% top-1 either way — identical to a tenth
of a point against a 0.60 point standard error, log loss 1.857 against 1.859, the same
player picked on 85.2% of shots, and a mean probability change of 0.012. The model has
not learned a usable left/right asymmetry. Getting the width convention wrong would be
undetectable; getting the length convention wrong would put every player at the wrong
distance from the rim. Only one of these two was ever worth worrying about.

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
happens. The predicted movement **is** the thing being shown. So the movement model is
required regardless of what it does for accuracy.

**This section's recommendation has since been confirmed by measurement, and the
heading is now literally true rather than a judgement call.** `posnn.h5` has been
replaced by a set transformer (see the parent repo's README, *The movement model*), and
the replacement was fed back to the rebounder exactly as Decision A below imagines. It
is worth **0.1 points of top-1** against an 8.1-point oracle budget. Read the rest of
this section as the argument that turned out to be right; the numbers in it are the
older, smaller estimates.

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
upstream of it.

*Settled, 2026-08-20.* The replacement movement model was built and the swap measured
properly — the rebounder trained on its *predictions* rather than on rim-time truth,
which is the only version of the experiment worth running. On the test split: 30.0%
from release-time features, 30.1% with the movement model's rim-time columns added,
29.8% marginalising over sixteen sampled scenes, and 38.1% from rim-time truth. The
forecast captures roughly one percent of its own budget, and a "stay put" control scores
29.7%, so the extra columns are not carrying anything on their own either. The reading
that survives every control: the predictable part of a player's next 1.9 seconds is the
part already implied by where everyone is standing, which the rebounder can see for
itself. **Serve `FinalModel.pkl` on `SERVED_FEATURES` and animate beside it.** The
`served+movement` regime exists, works, and is not worth deploying.

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

*Built, 2026-08-20.* That is now `rebounding/eval/movement.py`, and both predictions in
this subsection held. Trained on L2, the same network sends defenders rimward on 84.4%
of shots against a real 69.0% and produces 40% more player collisions than the corpus
contains — the drift and the overlap, exactly as described. Changing only the loss to a
density removes both. The deployed head samples whole scenes from a per-scene latent and
lands on 0.56 collisions per scene against the corpus's 0.55; "no overlapping players"
turned out to be the metric the scene latent actually earns its place on.

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

### The movement entry point — `posnn.h5` is retired

The movement model has been rebuilt and ships the same way the rebounder does:

```bash
python -m rebounding.cli train-movement --out MovementModel.pkl   # in this repo
scp MovementModel.pkl <host>:<app dir>/                           # alongside FinalModel.pkl
```

That build is deterministic: seeded at 0 and rebuilt from a clean tree, it reproduces
the shipped weights **bitwise**, all 801,946 of them. So `cli describe --model
MovementModel.pkl` is checkable rather than merely informative — if the commit it
names is not the one you expect, rebuild at that commit and compare, and the answer
is exact rather than approximate.

```python
from rebounding.models.artifact import load_movement
from rebounding.serve import animate

movement = load_movement("MovementModel.pkl")
result = animate(movement, players, n=12)   # same ten dicts `predict` takes

result.scenes      # (12, 10, 2) sampled futures, indexed as `players` was
result.scene(0)    # one of them, (10, 2)
result.mean        # the conditional mean. For the rebounder. Do not draw it.
```

Four things follow, and the first is the one that changes the front end:

- **It returns several futures, not one.** The model predicts a distribution over
  whole scenes. `scenes` is that distribution sampled, and each draw is internally
  consistent — one latent commits all ten players at once, so a sampled scene is a
  scenario rather than ten independent guesses.
- **Do not average the scenes before drawing them.** That collapses the model back to
  the point estimate it replaced, and the point estimate is what put every player in
  the paint. Animate one draw; fan the rest out as ghosts if you want to show the
  spread. `mean` exists for the rebounder's features and for tests.
- **Every returned scene is physically possible, and that is not free.** The decoder
  is Gaussian, so its support is unbounded and a thin tail of draws asks a player to
  cross the court in a second and a half — 0.16% of scenes against the predicted
  flight time. `animate` redraws those before returning them, and reports what it had
  to do as `result.redrawn` and `result.clamped`. Both are normally 0; a placement
  that pushes them up is one the model finds strange, which is worth surfacing rather
  than hiding. Nothing is needed on the front end for this.
- **`msd.pkl` is gone.** The normalisation constants are buffers inside the network,
  so they cannot drift away from the weights they belong to. So is TensorFlow: the
  bundle is a torch state dict, and the CPU wheel is the right one to install — this
  is a small model over ten tokens and a request is one forward pass.
- **It eats `SERVED_FEATURES`**, exactly as the rebounder does. There is no second
  input contract to reproduce, no 41.65 hoop, no swapped `arctan2`. Everything §3 said
  about the 2017 eight-column input is now historical.

To fold it back into the rebounder as well, build the `served+movement` regime:

```bash
python -m rebounding.cli train --regime served+movement --movement MovementModel.pkl
```

and pass the movement bundle to `predict(artifact, players, movement=movement)`. An
artifact that wants those ten columns without one **raises** rather than defaulting,
for the same reason a velocity-hungry artifact does. Note what that build does with
the training rows: it generates their rim-time columns from the movement model's
*predictions*, never from the corpus's real `pos_*`. Training on truth and serving
forecasts is precisely the skew that produced the 2017 number, and `cli baseline`
now refuses this regime outright rather than fitting it off the frame.

---

## 6. Deployment

### Installing on the app host

```bash
pip install 'rebounding[serve]' --extra-index-url https://download.pytorch.org/whl/cpu
```

The `serve` extra is exactly what `predict` and `animate` need. Two things about that
line are load-bearing:

- **The CPU index is not optional.** Without it pip resolves torch's default Linux
  wheel, which carries a bundled CUDA runtime — roughly 2.5 GB onto a host that will
  never see a GPU. The movement model is 802k parameters over ten tokens and one
  request is a single forward pass; the CPU wheel is not a compromise here.
- **scikit-learn is a serving dependency, not a training one.** It looks like it should
  be droppable, and it is not: `BoostedSoftmax` fits a `lightgbm.LGBMRegressor`, so
  `FinalModel.pkl` unpickles an sklearn estimator and `load` fails without it. The only
  thing `serve` drops relative to `models` is xgboost.

### Serving

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
3. ~~**Fix §3.4**~~ **Do not.** That section has been withdrawn: the assignment it
   flags is a coordinate-frame conversion, not a transposition, and reversing it breaks
   the render. Nothing gates step 4 any more.
4. ~~**Look at `posnn.h5` honestly**~~ **Superseded — it has been replaced.** The
   question that step asked was whether the 2017 movement model's paint-drift was bad
   enough to justify a rebuild. It was not answerable as written (there is no
   TensorFlow in the parent repo's environment, and a 2017 Keras HDF5 may not load in
   current Keras at all), and it is now moot: the model has been retrained as a set
   transformer over the ten players, and the drift was measured on the *new* model's
   own point-estimate head rather than guessed at from the old one's animation. See
   "the movement entry point" in §5 for the API and the parent repo's README for the
   numbers. What is left on the front end is drawing the sampled scenes rather than
   one dot.
5. ~~**Confirm the coordinate frame**~~ **Done** (§3.5). The canvas is the basket half
   with the basket at the bottom, so screen-down is model `x`; SportVU's bird's-eye view
   and the rotational fold settle the width axis. Neither flip is set, and mirroring the
   test split shows the width convention is worth 0.0 points anyway.
6. **Correct the accuracy copy** (§4).
7. **Deploy under gunicorn** (§6).

Steps 1–2 make the app runnable; 3 and 5 make it correct; 4 is done and its output is
a second bundle to scp; 6 makes it honest.

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
