"""Post-hoc XY displacement diagnostics, independent of matching decisions.

The diagnostic assumes a dominant coherent reference population. Its exponential
distance score is neither a match probability nor a calibrated p-value.
"""

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from numbers import Integral
from typing import Optional
import warnings

import numpy as np
from scipy.stats import median_abs_deviation
from sklearn.covariance import MinCovDet


@dataclass(frozen=True)
class DisplacementConsistencyResult:
    score: Optional[float]
    distance: Optional[float]
    residual_um: Optional[float]
    reference_count: int
    reason: str


@dataclass(frozen=True)
class _ReferenceFit:
    centre: Optional[np.ndarray]
    cholesky: Optional[np.ndarray]
    reason: str = ""


class DisplacementConsistency:
    """An input snapshot supporting endpoint-excluded displacement scoring.

    ``raw_centroids`` has shape (3, n, 2); rows 1:3 are XY in micrometres
    and the final axis contains independent waveform halves. Both halves must
    be finite. References share a session pair, probe, and, if provided, shank.
    Without shanks the grouping is explicitly probe-only.

    Accepted pairs are canonicalized and deduplicated. Within each scope, all
    accepted pairs involving an ambiguously assigned endpoint are discarded,
    even if another conflicting pair is nonfinite or touches the candidate.
    Candidate endpoints are then excluded from every remaining reference.

    MCD estimates centre and scatter. The diagonal covariance floor is
    ``(MAD_normal(half_displacement_0 - half_displacement_1) / 2)**2``.
    Under independent, equal-variance halves this estimates the variance of
    the mean pair displacement. Adding it is deliberately conservative: the
    empirical scatter already includes measurement noise. Zero MAD does not
    establish zero uncertainty; no artificial epsilon is substituted.

    The default minimum of 20 references is configurable (at least 3), not calibrated.
    Fits are cached in a bounded, instance-local LRU; replace the snapshot
    whenever decisions or scientific inputs change. No input is modified.
    """

    _CACHE_SIZE = 128
    _MAX_CONDITION = 1e12

    def __init__(
        self,
        raw_centroids,
        session_ids,
        probe_ids,
        accepted_pairs,
        *,
        shank_ids=None,
        min_references=20,
    ):
        if (
            isinstance(min_references, (bool, np.bool_))
            or not isinstance(min_references, Integral)
            or min_references < 3
        ):
            raise ValueError("min_references must be an integer >= 3")
        self._min_references = int(min_references)
        raw = np.asarray(raw_centroids, dtype=float)
        if raw.ndim != 3 or raw.shape[0] != 3 or raw.shape[2] != 2:
            raise ValueError("raw_centroids must have shape (3, n, 2)")
        self._n_units = raw.shape[1]
        self._halves = np.array(raw[1:3].transpose(1, 0, 2), copy=True)
        self._halves.setflags(write=False)
        self._finite = np.isfinite(self._halves).all(axis=(1, 2))
        with np.errstate(invalid="ignore", over="ignore"):
            self._means = self._halves[:, :, 0] / 2 + self._halves[:, :, 1] / 2
        self._means.setflags(write=False)
        self._sessions = self._label_codes(session_ids, "session_ids")
        self._probes = self._label_codes(probe_ids, "probe_ids")
        self._shanks = (
            None if shank_ids is None else self._label_codes(shank_ids, "shank_ids")
        )

        grouped_pairs = defaultdict(set)
        for pair in accepted_pairs:
            if np.shape(pair) != (2,):
                raise ValueError("accepted_pairs must contain two-index pairs")
            a, b = (self._unit_index(index) for index in pair)
            scope, a, b, reason = self._scope(a, b)
            if not reason:
                grouped_pairs[scope].add((a, b))

        self._references = {}
        for scope, pairs in grouped_pairs.items():
            counts = Counter(endpoint for pair in pairs for endpoint in pair)
            independent = [
                (a, b)
                for a, b in sorted(pairs)
                if counts[a] == counts[b] == 1 and self._finite[a] and self._finite[b]
            ]
            self._references[scope] = tuple(independent)
        self._fit_cache = OrderedDict()

    def _label_codes(self, labels, name):
        labels = np.asarray(labels)
        if labels.shape != (self._n_units,):
            raise ValueError(f"{name} must have one value per unit")
        mapping = {}
        codes = np.full(self._n_units, -1, dtype=int)
        for index, label in enumerate(labels):
            if label is None:
                continue
            try:
                if label != label:  # NaN identifiers cannot define a scope.
                    continue
                if isinstance(label, (float, np.floating)) and not np.isfinite(label):
                    continue
                if label not in mapping:
                    mapping[label] = len(mapping)
                codes[index] = mapping[label]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain scalar, hashable identifiers") from exc
        codes.setflags(write=False)
        return codes

    def _unit_index(self, index):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
            raise ValueError("unit indices must be integers")
        index = int(index)
        if not 0 <= index < self._n_units:
            raise IndexError(f"unit index {index} is outside [0, {self._n_units})")
        return index

    def _scope(self, a, b):
        for codes, name in (
            (self._sessions, "session"),
            (self._probes, "probe"),
            (self._shanks, "shank"),
        ):
            if codes is not None and (codes[a] < 0 or codes[b] < 0):
                return None, a, b, f"Missing {name} identifier"
        if self._sessions[a] == self._sessions[b]:
            return None, a, b, "Candidate must span two sessions"
        if self._probes[a] != self._probes[b]:
            return None, a, b, "Candidate units belong to different probes"
        if self._shanks is not None and self._shanks[a] != self._shanks[b]:
            return None, a, b, "Candidate units belong to different shanks"
        if self._sessions[a] > self._sessions[b]:
            a, b = b, a
        scope = (
            self._sessions[a],
            self._sessions[b],
            self._probes[a],
            None if self._shanks is None else self._shanks[a],
        )
        return scope, a, b, ""

    @staticmethod
    def _unavailable(reason, reference_count=0):
        return DisplacementConsistencyResult(None, None, None, reference_count, reason)

    def score(self, unit_a, unit_b):
        """Return a frozen diagnostic; invalid indices raise, unsuitable pairs do not.

        ``distance`` is sqrt(residual.T @ covariance_inverse @ residual).
        ``score`` is 100 * exp(-distance**2 / 2), and ``residual_um`` is
        the unnormalized Euclidean distance from the robust displacement centre.
        All three are None when unavailable, with an explanatory ``reason``.
        """
        a, b = self._unit_index(unit_a), self._unit_index(unit_b)
        scope, a, b, reason = self._scope(a, b)
        if reason:
            return self._unavailable(reason)
        if not self._finite[a] or not self._finite[b]:
            return self._unavailable("Candidate XY centroids require two finite halves")

        pairs = tuple(
            pair for pair in self._references.get(scope, ()) if a not in pair and b not in pair
        )
        if pairs:
            first, second = np.asarray(pairs, dtype=int).T
            with np.errstate(over="ignore", invalid="ignore"):
                displacements = self._means[second] - self._means[first]
                half_differences = (
                    self._halves[second, :, 0] - self._halves[first, :, 0]
                ) - (self._halves[second, :, 1] - self._halves[first, :, 1])
            finite = np.isfinite(displacements).all(axis=1)
            finite &= np.isfinite(half_differences).all(axis=1)
            pairs = tuple(pair for pair, keep in zip(pairs, finite) if keep)
            displacements, half_differences = displacements[finite], half_differences[finite]

        count = len(pairs)
        if count < self._min_references:
            return self._unavailable(
                f"Insufficient independent references ({count} < {self._min_references})",
                count,
            )
        fit = self._fit_cache.get(pairs)
        if fit is None:
            fit = self._fit_references(displacements, half_differences)
            self._fit_cache[pairs] = fit
            if len(self._fit_cache) > self._CACHE_SIZE:
                self._fit_cache.popitem(last=False)
        self._fit_cache.move_to_end(pairs)
        if fit.reason:
            return self._unavailable(fit.reason, count)

        with np.errstate(over="ignore", invalid="ignore"):
            residual = self._means[b] - self._means[a] - fit.centre
            normalized = np.linalg.solve(fit.cholesky, residual)
            distance = float(np.hypot(*normalized))
            residual_um = float(np.hypot(*residual))
        if not np.isfinite(distance) or not np.isfinite(residual_um):
            return self._unavailable("Candidate residual exceeds numerical range", count)
        # Squaring a very large finite distance can overflow, but the score is
        # already below floating-point range there.
        score = 0.0 if distance > 40 else float(100 * np.exp(-0.5 * distance**2))
        return DisplacementConsistencyResult(score, distance, residual_um, count, "")

    @staticmethod
    def _robust_scatter(displacements):
        origin = np.median(displacements, axis=0)
        centered = displacements - origin
        _, singular_values, directions = np.linalg.svd(centered, full_matrices=False)
        tolerance = max(centered.shape) * np.finfo(float).eps * singular_values[0]
        rank = int(np.count_nonzero(singular_values > tolerance))
        if rank == 0:
            return origin, np.zeros((2, 2))

        # An exact point population large enough to fill MCD's support has
        # zero scatter. sklearn raises for this legitimate degenerate case;
        # only independently measured split-half noise can make it usable.
        unique, counts = np.unique(displacements, axis=0, return_counts=True)
        support_size = int(np.ceil((len(displacements) + rank + 1) / 2))
        if counts.max() >= support_size:
            return unique[counts.argmax()].copy(), np.zeros((2, 2))

        # Fit collinear data in its identifiable subspace rather than asking
        # MCD to invert a singular 2D cloud. Restore the physical XY basis
        # before adding the split-half measurement floor.
        basis = directions[:rank].T
        scale = singular_values[0]
        projected = (centered / scale) @ basis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            estimator = MinCovDet(random_state=0).fit(projected)
        centre = origin + (estimator.location_ @ basis.T) * scale
        covariance = (basis @ estimator.covariance_ @ basis.T) * scale * scale
        return centre, covariance

    def _fit_references(self, displacements, half_differences):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                centre, covariance = self._robust_scatter(displacements)
                noise_std = median_abs_deviation(half_differences, axis=0, scale="normal") / 2
                covariance = covariance + np.diag(noise_std**2)
                covariance = (covariance + covariance.T) / 2
                if not np.isfinite(centre).all() or not np.isfinite(covariance).all():
                    raise ValueError("nonfinite covariance")
                eigenvalues = np.linalg.eigvalsh(covariance)
                if (
                    eigenvalues[0] <= 0
                    or eigenvalues[0] < eigenvalues[-1] / self._MAX_CONDITION
                ):
                    return _ReferenceFit(
                        None, None,
                        "Singular or ill-conditioned covariance; split-half noise "
                        "does not support reliable two-dimensional uncertainty",
                    )
                cholesky = np.linalg.cholesky(covariance)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return _ReferenceFit(
                None, None, "Robust reference covariance could not be estimated reliably"
            )
        centre.setflags(write=False)
        cholesky.setflags(write=False)
        return _ReferenceFit(centre, cholesky)
