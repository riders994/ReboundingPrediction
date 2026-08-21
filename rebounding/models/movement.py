"""The movement model: where the ten players will be when the ball reaches the rim.

This is model three of the three. It exists for two reasons that pull in different
directions, and the design follows from taking both seriously.

**It is the product.** The web app's whole interaction is placing ten players and
watching what happens next, so whatever this predicts is drawn on screen. Its failure
modes are things a user watches happen rather than a number in a table.

**It is the feeder.** Its predicted rim-time positions unlock the ``pos_*`` and
``move_*`` families for the served rebounder. The gap between what the app can supply
today and rim-time truth is the budget this model has to earn back.

Three things about the target decide the architecture.

*It is a set.* Ten players with no natural order, whose interaction is the phenomenon
being modelled -- a box-out is a pairwise relation, and a model that sees one player at
a time cannot represent one. The encoder here is a stack of self-attention layers with
no positional encoding, so it is permutation equivariant by construction: shuffle the
input and the output shuffles with it. The pipeline's canonical slot order is a
convenience for tensor layout, not information this model is allowed to lean on.

*It is joint.* Ten independently plausible players can be a collectively impossible
scene -- two of them in the same square foot, nobody near the rim. Sampling has to
happen once per scene, not once per player, which is what the latent in
:class:`SceneVAEHead` is for.

*It is multimodal, and this is the one that broke the 2017 model.* An offensive player
fourteen feet out either crashes the glass or leaks out, and those are opposite
futures. Squared error on a target like that returns the conditional mean, which is
neither -- every player drifts toward the average of two intentions. That is a
property of the loss, not of the network: a deeper model trained on L2 produces a more
confident version of the same wrong answer. So the point head below exists to
*reproduce* that failure and measure it, and the two density heads are the actual
candidates.

The three heads, all on the same encoder so the comparison is about the objective:

===============  ==========================================================
head             what it predicts
===============  ==========================================================
``point``        one displacement per player. Trained on L2. The 2017 shape.
``mixture``      K Gaussians per player. Trained on NLL. Multimodal, but
                 each player is sampled independently, so scenes are not
                 coherent -- a diagnostic rather than a candidate.
``cvae``         one latent per *scene*, decoded to ten correlated
                 displacements. Trained on the ELBO. Samples whole futures.
===============  ==========================================================

Inputs are :data:`rebounding.data.features.SERVED_FEATURES` exactly -- the same 27
columns the served rebounder eats. That is not a coincidence to be tidied away later:
this model's whole purpose is to run on what the app has, so letting it see a velocity
or a rim-time position would make it unservable in the only place it is wanted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

N_PLAYERS = 10

# Displacements are divided by this before the loss sees them, so the network's
# outputs sit near unit scale whatever the head. Players travel 7.5 ft over a flight,
# so this puts a typical move at about 0.75.
MOVE_SCALE = 10.0

# Latent draws averaged to estimate the VAE's conditional mean. The decoder is not
# linear in z, so decoding the prior's mean is not the mean of the decodings -- and it
# is the latter the rebounder wants. Fixed rather than tunable, and seeded, so
# `predict` stays deterministic.
LATENT_MEAN_DRAWS = 32
LATENT_MEAN_SEED = 20250820

# Floor on any predicted standard deviation, in scaled units. Without it the mixture
# head will collapse a component onto a single training point and send the NLL to
# minus infinity. 0.02 is 0.2 ft, comfortably inside tracking noise.
MIN_SIGMA = 0.02


@dataclass
class MovementConfig:
    """Architecture and optimisation, kept together so an artifact can carry it."""

    head: str = "cvae"
    d_model: int = 96
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1

    # mixture head
    n_components: int = 6

    # cvae head
    d_latent: int = 16
    kl_weight: float = 1.0
    kl_warmup_epochs: int = 10

    # optimisation
    epochs: int = 120
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    seed: int = 0

    # Threads for the fitting loop. The tensors here are tiny -- ten tokens a sample --
    # so torch's default of one thread per core spends more time synchronising than
    # computing: on this twelve-thread box the default oversubscribes badly under any
    # concurrent load. None leaves torch's default alone.
    threads: int | None = 6

    def __post_init__(self) -> None:
        if self.head not in ("point", "mixture", "cvae"):
            raise ValueError(f"head must be point, mixture or cvae, got {self.head!r}")


def _torch():
    """Import torch on demand, with a message that says what to install.

    The rest of the package runs without it -- the boosted rebounder is numpy and
    LightGBM -- so torch stays an optional dependency of this module alone.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "the movement model needs pytorch: pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu"
        ) from exc
    return torch


def _build(config: MovementConfig, n_features: int):
    """Construct the network. Kept out of the class so the import stays lazy."""
    torch = _torch()
    nn = torch.nn

    class SetEncoder(nn.Module):
        """Self-attention over the ten players, permutation equivariant.

        No positional encoding on purpose. The slot index carries real information in
        this corpus -- offence occupies the first five slots, sorted nearest-to-rim --
        but all of it is already present as features (``is_offense``,
        ``dist_minus_best``, ``dist_minus_nearest_teammate``). Letting the model read
        the index as well would let it learn "slot 0 crashes" instead of "the nearest
        offensive player crashes", which is the same thing until the app hands over
        players in a different order.
        """

        def __init__(self, d_in: int) -> None:
            super().__init__()
            self.project = nn.Linear(d_in, config.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(
                layer, num_layers=config.n_layers, enable_nested_tensor=False
            )
            self.norm = nn.LayerNorm(config.d_model)

        def forward(self, x):
            return self.norm(self.blocks(self.project(x)))

    class PointHead(nn.Module):
        """One displacement per player, trained on squared error. The 2017 shape."""

        def __init__(self) -> None:
            super().__init__()
            self.out = nn.Linear(config.d_model, 2)

        def forward(self, h, target=None):
            return {"mean": self.out(h)}

        def conditional_mean(self, out):
            return out["mean"]

        def loss(self, out, target, **_):
            return {"loss": ((out["mean"] - target) ** 2).sum(-1).mean()}

        def sample(self, out, n: int):
            return out["mean"].unsqueeze(0).expand(n, *out["mean"].shape)

    class MixtureHead(nn.Module):
        """K correlated 2-D Gaussians per player, trained on the exact NLL.

        Per player, which is the limitation: the K components describe *this* player's
        futures and nothing ties his choice of component to anyone else's. It answers
        "is the target multimodal, and does modelling that help" without answering
        "can we draw a coherent scene".
        """

        def __init__(self) -> None:
            super().__init__()
            k = config.n_components
            # per component: weight, mean_x, mean_y, log_sx, log_sy, tanh-corr
            self.out = nn.Linear(config.d_model, k * 6)

        def forward(self, h, target=None):
            k = config.n_components
            raw = self.out(h).view(*h.shape[:-1], k, 6)
            return {
                "logits": raw[..., 0],
                "mean": raw[..., 1:3],
                "sigma": torch.nn.functional.softplus(raw[..., 3:5]) + MIN_SIGMA,
                "rho": torch.tanh(raw[..., 5]) * 0.99,
            }

        def component_logprob(self, out, target):
            """Log N(target | component) for every component. (..., K)"""
            delta = (target.unsqueeze(-2) - out["mean"]) / out["sigma"]
            rho = out["rho"]
            one_minus = 1.0 - rho**2
            quad = (
                delta[..., 0] ** 2 + delta[..., 1] ** 2
                - 2 * rho * delta[..., 0] * delta[..., 1]
            ) / one_minus
            log_det = torch.log(out["sigma"]).sum(-1) + 0.5 * torch.log(one_minus)
            return -0.5 * quad - log_det - math.log(2 * math.pi)

        def log_prob(self, out, target):
            weights = torch.log_softmax(out["logits"], dim=-1)
            return torch.logsumexp(weights + self.component_logprob(out, target), dim=-1)

        def conditional_mean(self, out):
            weights = torch.softmax(out["logits"], dim=-1).unsqueeze(-1)
            return (weights * out["mean"]).sum(-2)

        def loss(self, out, target, **_):
            return {"loss": -self.log_prob(out, target).mean()}

        def sample(self, out, n: int):
            """Draw ``n`` scenes. Each player picks his own component, independently.

            That independence is the head's whole limitation, stated in code: nothing
            couples one player's choice to another's, so two sampled players can both
            crash to the same spot.
            """
            shape = out["logits"].shape  # (batch, players, K)
            weights = torch.softmax(out["logits"], dim=-1)
            flat = weights.reshape(-1, config.n_components)
            picked = torch.multinomial(flat, n, replacement=True)  # (batch*players, n)
            picked = picked.view(*shape[:-1], n).permute(2, 0, 1)  # (n, batch, players)

            gather = picked.unsqueeze(-1)
            mean = torch.gather(
                out["mean"].expand(n, *out["mean"].shape), -2,
                gather.unsqueeze(-1).expand(*gather.shape, 2),
            ).squeeze(-2)
            sigma = torch.gather(
                out["sigma"].expand(n, *out["sigma"].shape), -2,
                gather.unsqueeze(-1).expand(*gather.shape, 2),
            ).squeeze(-2)
            rho = torch.gather(out["rho"].expand(n, *out["rho"].shape), -1, gather).squeeze(-1)

            z1, z2 = torch.randn_like(mean[..., 0]), torch.randn_like(mean[..., 1])
            dx = sigma[..., 0] * z1
            dy = sigma[..., 1] * (rho * z1 + torch.sqrt(1 - rho**2) * z2)
            return mean + torch.stack([dx, dy], dim=-1)

    class SceneVAEHead(nn.Module):
        """One latent per scene, decoded to ten correlated displacements.

        The latent is pooled over all ten players and broadcast back to all ten, so a
        single draw commits the whole scene at once: if ``z`` says "this is a
        transition possession", every player is decoded under that reading rather than
        each flipping a coin of his own. That shared commitment is the difference
        between ten plausible players and one plausible future.

        The decoder emits a mean and a per-player scale rather than a bare point, so
        the reconstruction term is a real log-likelihood and the sampler has residual
        spread once the scene is chosen.
        """

        def __init__(self, d_in: int) -> None:
            super().__init__()
            self.posterior = SetEncoder(d_in + 2)
            self.to_latent = nn.Linear(config.d_model, config.d_latent * 2)
            self.decode = nn.Sequential(
                nn.Linear(config.d_model + config.d_latent, config.d_ff),
                nn.GELU(),
                nn.Linear(config.d_ff, config.d_ff),
                nn.GELU(),
            )
            self.out = nn.Linear(config.d_ff, 4)

        def _decode(self, h, z):
            broadcast = z.unsqueeze(1).expand(-1, h.shape[1], -1)
            raw = self.out(self.decode(torch.cat([h, broadcast], dim=-1)))
            return {
                "mean": raw[..., :2],
                "sigma": torch.nn.functional.softplus(raw[..., 2:4]) + MIN_SIGMA,
            }

        def forward(self, h, target=None, x=None):
            if target is None:
                # The prior *mean*, not a draw, so `predict` is deterministic. It is
                # also the least interesting thing this head can do -- decoding z = 0
                # is a point estimate again, and reintroduces exactly the averaging the
                # latent exists to avoid. Use it for comparison; draw for the UI.
                z = torch.zeros(h.shape[0], config.d_latent, device=h.device)
                return {**self._decode(h, z), "encoded": h}
            pooled = self.posterior(torch.cat([x, target], dim=-1)).mean(dim=1)
            mu, log_var = self.to_latent(pooled).chunk(2, dim=-1)
            log_var = log_var.clamp(-8.0, 8.0)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
            return {**self._decode(h, z), "mu": mu, "log_var": log_var, "encoded": h}

        def conditional_mean(self, out):
            """Averaged over latent draws, not decoded at the prior's mean.

            Decoding ``z = 0`` is a point in the decoder's output space and nothing
            says it is the average of the points ``z`` actually reaches. Seeded so
            two calls on the same input agree, which the rebounder's features need.
            """
            h = out["encoded"]
            generator = torch.Generator(device=h.device).manual_seed(LATENT_MEAN_SEED)
            total = None
            for _ in range(LATENT_MEAN_DRAWS):
                z = torch.randn(
                    h.shape[0], config.d_latent, generator=generator, device=h.device
                )
                decoded = self._decode(h, z)["mean"]
                total = decoded if total is None else total + decoded
            return total / LATENT_MEAN_DRAWS

        def loss(self, out, target, kl_weight: float = 1.0, **_):
            delta = (target - out["mean"]) / out["sigma"]
            nll = (0.5 * delta**2 + torch.log(out["sigma"]) + 0.5 * math.log(2 * math.pi))
            reconstruction = nll.sum(-1).mean()
            kl = -0.5 * (1 + out["log_var"] - out["mu"] ** 2 - out["log_var"].exp()).sum(-1).mean()
            # Per-player reconstruction against a per-scene KL: the ten players share
            # one latent, so the KL is paid once for the whole scene.
            return {
                "loss": reconstruction + kl_weight * kl / N_PLAYERS,
                "reconstruction": reconstruction.detach(),
                "kl": kl.detach(),
            }

        def sample(self, out, n: int):
            h = out["encoded"]
            draws = []
            for _ in range(n):
                z = torch.randn(h.shape[0], config.d_latent, device=h.device)
                decoded = self._decode(h, z)
                draws.append(decoded["mean"] + torch.randn_like(decoded["mean"]) * decoded["sigma"])
            return torch.stack(draws)

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = SetEncoder(n_features)
            self.head = {
                "point": PointHead,
                "mixture": MixtureHead,
                "cvae": lambda: SceneVAEHead(n_features),
            }[config.head]()
            self.register_buffer("centre", torch.zeros(n_features))
            self.register_buffer("scale", torch.ones(n_features))

        def standardise(self, x):
            return (x - self.centre) / self.scale

        def forward(self, x, target=None):
            standardised = self.standardise(x)
            h = self.encoder(standardised)
            if config.head == "cvae":
                return self.head(h, target=target, x=standardised)
            return self.head(h, target=target)

    return Network()


class MovementModel:
    """Fit, sample and score the set model. Same tensor conventions as the ladder.

    ``x`` is ``(n_shots, 10, n_features)`` of release-time features and ``y`` is
    ``(n_shots, 10, 2)`` of rim-time *displacement* in feet. Displacement rather than
    absolute position because it is centred near zero and because "stay put" is then
    the origin rather than an arbitrary point the network has to learn to hit --
    staying put is the baseline this has to beat, so it should cost nothing to express.
    """

    def __init__(self, config: MovementConfig | None = None) -> None:
        self.config = config or MovementConfig()
        self.network = None
        self.n_features: int | None = None
        self.history: list[dict[str, float]] = []
        self.best_epoch: int | None = None
        self.best_val: float | None = None

    # -- fitting ---------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> MovementModel:
        torch = _torch()
        config = self.config
        torch.manual_seed(config.seed)
        if config.threads:
            torch.set_num_threads(config.threads)

        if x.shape[1] != N_PLAYERS:
            raise ValueError(f"expected {N_PLAYERS} players per shot, got {x.shape[1]}")
        if y.shape != (x.shape[0], N_PLAYERS, 2):
            raise ValueError(f"target must be (n_shots, {N_PLAYERS}, 2), got {y.shape}")

        self.n_features = x.shape[2]
        self.network = _build(config, self.n_features)

        features = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
        target = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32) / MOVE_SCALE

        # Standardisation is fitted here and stored as buffers, so it travels with the
        # weights and cannot drift away from them at serving time.
        flat = features.reshape(-1, self.n_features)
        centre = flat.mean(0)
        scale = flat.std(0).clamp(min=1e-6)
        self.network.centre.copy_(centre)
        self.network.scale.copy_(scale)

        optimiser = torch.optim.AdamW(
            self.network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)

        val_tensors = None
        if validation is not None:
            vx, vy = validation
            val_tensors = (
                torch.as_tensor(np.ascontiguousarray(vx), dtype=torch.float32),
                torch.as_tensor(np.ascontiguousarray(vy), dtype=torch.float32) / MOVE_SCALE,
            )

        n = len(features)
        best = (math.inf, None, -1)
        generator = torch.Generator().manual_seed(config.seed)

        for epoch in range(config.epochs):
            self.network.train()
            # KL annealing: a VAE handed its full KL from epoch zero collapses the
            # latent to the prior and degenerates into the point head it replaced.
            kl_weight = config.kl_weight * min(1.0, (epoch + 1) / max(1, config.kl_warmup_epochs))
            order = torch.randperm(n, generator=generator)
            running: dict[str, float] = {}

            for start in range(0, n, config.batch_size):
                batch = order[start : start + config.batch_size]
                bx, by = features[batch], target[batch]
                out = self.network(bx, target=by)
                terms = self.network.head.loss(out, by, kl_weight=kl_weight)
                optimiser.zero_grad(set_to_none=True)
                terms["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                optimiser.step()
                for name, value in terms.items():
                    running[name] = running.get(name, 0.0) + float(value.detach()) * len(batch)

            schedule.step()
            record = {"epoch": epoch, "kl_weight": kl_weight}
            record.update({f"train_{k}": v / n for k, v in running.items()})

            if val_tensors is not None:
                record["val_loss"] = self._evaluate(*val_tensors, kl_weight=config.kl_weight)
                if record["val_loss"] < best[0] - 1e-5:
                    best = (
                        record["val_loss"],
                        {k: v.detach().clone() for k, v in self.network.state_dict().items()},
                        epoch,
                    )
            self.history.append(record)

            if val_tensors is not None and epoch - best[2] >= config.patience:
                break

        if best[1] is not None:
            self.network.load_state_dict(best[1])
            self.best_epoch = best[2]
            self.best_val = best[0]
        return self

    def _evaluate(self, features, target, kl_weight: float) -> float:
        torch = _torch()
        self.network.eval()
        total, n = 0.0, len(features)
        with torch.no_grad():
            for start in range(0, n, 1024):
                bx, by = features[start : start + 1024], target[start : start + 1024]
                out = self.network(bx, target=by)
                total += float(self.network.head.loss(out, by, kl_weight=kl_weight)["loss"]) * len(bx)
        return total / n

    # -- prediction ------------------------------------------------------------

    def _forward(self, x: np.ndarray, batch: int = 1024):
        torch = _torch()
        if self.network is None:
            raise RuntimeError("model is not fitted")
        self.network.eval()
        features = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, len(features), batch):
                chunk = features[start : start + batch]
                yield chunk, self.network(chunk)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Conditional-mean displacement in feet, ``(n_shots, 10, 2)``.

        Provided for comparison and for the downstream rebounder, **not** for the
        animation. On a multimodal target this is exactly the quantity that drifts;
        :meth:`sample` is what the UI should draw.
        """
        torch = _torch()
        chunks = [self.network.head.conditional_mean(out) for _, out in self._forward(x)]
        return torch.cat(chunks).numpy() * MOVE_SCALE

    def sample(self, x: np.ndarray, n: int = 20, seed: int | None = None) -> np.ndarray:
        """``n`` sampled futures per shot, ``(n, n_shots, 10, 2)``, in feet."""
        torch = _torch()
        if seed is not None:
            torch.manual_seed(seed)
        chunks = []
        for _, out in self._forward(x):
            chunks.append(self.network.head.sample(out, n))
        return torch.cat(chunks, dim=1).numpy() * MOVE_SCALE

    def log_prob(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-player log density of the truth, ``(n_shots, 10)``, in feet^-2.

        Only the mixture head has a tractable one; the VAE's marginal needs importance
        sampling and the point head has no density at all. Raises rather than
        returning a proxy, because a proxy would end up in a table next to real ones.
        """
        torch = _torch()
        if self.config.head != "mixture":
            raise NotImplementedError(
                f"{self.config.head!r} has no closed-form density; use sample-based metrics"
            )
        target = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32) / MOVE_SCALE
        chunks, at = [], 0
        for chunk, out in self._forward(x):
            piece = target[at : at + len(chunk)]
            chunks.append(self.network.head.log_prob(out, piece))
            at += len(chunk)
        # Change of variables back to feet: the density was over scaled coordinates.
        return torch.cat(chunks).numpy() - 2 * math.log(MOVE_SCALE)

    # -- persistence -----------------------------------------------------------

    def state(self) -> dict[str, Any]:
        if self.network is None:
            raise RuntimeError("model is not fitted")
        return {
            "config": self.config,
            "n_features": self.n_features,
            "weights": {k: v.cpu().numpy() for k, v in self.network.state_dict().items()},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> MovementModel:
        model = cls(state["config"])
        model.n_features = state["n_features"]
        model.network = _build(model.config, model.n_features)
        as_tensor = _torch().as_tensor
        model.network.load_state_dict(
            {k: as_tensor(v) for k, v in state["weights"].items()}
        )
        model.network.eval()
        return model
