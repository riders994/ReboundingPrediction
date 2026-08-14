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
from.** This is the most user-visible bug in the app: the dots are in the wrong
places, and it would look like a bad model rather than a bad index.

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

**The honest number for the app is top-1 from release-time inputs: 27.6%**, because
release is all a user can supply. Against a 10-player field, chance is 10% and the
positional prior alone is 23.1%.

If the UI displays a confidence or accuracy figure anywhere, it needs to become 27.6%
top-1, phrased as "picks the right rebounder about 1 time in 4".

---

## 5. What the app should serve instead

### The architecture can get simpler

Today the app chains two models: `posnn.h5` predicts where players will be when the
ball arrives, then the rebounder model runs on those predicted positions. That chain
is the source of the app's worst behaviour — the movement model is an L2 regression
onto a multimodal target, so it returns the conditional mean and drifts every player
toward the paint, with no term preventing two predicted players from occupying the
same spot.

**That chain is now known to be worth 8.2 points of top-1.** Measured on the rebuilt
corpus:

| regime | best top-1 | meaning |
|---|---|---|
| release-time features only | 27.6% | what a user can actually supply |
| rim-time features | 35.8% | ceiling; requires knowing the future |

A *perfect* movement model buys 8.2 points. The current one is far from perfect and
demonstrably biased. **Recommendation: drop the movement model for v1** and serve a
single model trained directly on release-time features. That removes Keras, `posnn.h5`,
`msd.pkl`, and every bug in §3.3 and §3.4 from the stack, at a cost that is bounded
above by 8 points and is probably negative in practice.

If the app wants to keep showing predicted movement as a *visual*, it can — but it
should not be in the prediction path.

### The feature contract

Train and serve exactly these, in this order. From
`rebounding/data/features.py::RELEASE_FEATURES`:

```
pre_x, pre_y, pre_dist, pre_angle, pre_vx, pre_vy, pre_speed,
pre_cos_shooter, pre_box, is_offense, is_shooter, role
```

Notes for the serving side:

- `pre_vx`, `pre_vy`, `pre_speed` are velocity at release. **The app has no velocity** —
  a user places static dots. Either extend the UI to let a user drag a direction
  vector, or train the served model on the subset without velocity and accept the
  loss. Do not pass zeros for velocity to a model trained with real ones; that is
  another train/serve skew. Measure the no-velocity variant before choosing.
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
wanted. See `rebounding/models/conditional_logit.py`.

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

1. **Retrain and commit a model.** Nothing else can be tested until the app has
   weights. Use the release-time regime; remove `*.pkl` from `.gitignore` or use Git
   LFS, or the same loss happens again.
2. **Rewrite `webapp.py` for Python 3** around the explicit feature contract in §5.
   Delete `features()` and `boxgen()` from the app and import the pipeline's versions
   so there is exactly one definition of every feature.
3. **Fix §3.4** (the transposed render) — it is one line and the most visible defect.
4. **Confirm the coordinate frame** (§3.5) with a known-position round trip before
   trusting any number the app produces.
5. **Correct the accuracy copy** (§4).
6. **Deploy under gunicorn** (§6).

Steps 1 and 2 make the app runnable; 3 and 4 make it correct; 5 makes it honest.

---

## 8. Where things live

| what | where |
|---|---|
| feature definitions, regimes, slot ordering | `rebounding/data/features.py` |
| court geometry, folding, `HOOP`, `rim_angle` | `rebounding/data/court.py`, `rebounding/constants.py` |
| the grouped-softmax model | `rebounding/models/conditional_logit.py` |
| baseline numbers reproduced | `python -m rebounding.cli baseline --forest` |
| built training frame | `data/frame.parquet` (gitignored, rebuild with `cli build`) |

One caveat worth carrying: the two data sources behind all of this now disagree on
1.9% of rebounds — the NBA feed and Basketball-Reference credit some boards to
different players on the same team. That is label noise below the level any of these
decisions turn on, but it is a floor on achievable accuracy.
