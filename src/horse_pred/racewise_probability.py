"""Transparent and nonlinear utilities for supervised race-wise probability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np

from horse_pred.modeling import (
    GroupStructure,
    binary_win_targets,
    race_balanced_weights,
    race_softmax,
    validate_grouped_rows,
    validate_prediction_feature_columns,
    winner_mass_targets,
)


@dataclass(frozen=True)
class FrozenNumericTransform:
    """Train-only numeric imputation and scaling for linear utilities."""

    feature_names: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    constant_mask: np.ndarray
    z_clip: float = 10.0

    @classmethod
    def fit(
        cls,
        values: Any,
        feature_names: Sequence[str],
        *,
        z_clip: float = 10.0,
    ) -> FrozenNumericTransform:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
            raise ValueError("numeric transform matrix shape does not match features")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("numeric transform requires a non-empty matrix")
        finite = np.isfinite(matrix)
        medians = np.zeros(matrix.shape[1], dtype=np.float64)
        for index in range(matrix.shape[1]):
            column = matrix[finite[:, index], index]
            medians[index] = float(np.median(column)) if len(column) else 0.0
        imputed = np.where(finite, matrix, medians)
        means = imputed.mean(axis=0)
        raw_scales = imputed.std(axis=0)
        constant = (~np.isfinite(raw_scales)) | (raw_scales < 1e-12)
        scales = raw_scales.copy()
        scales[constant] = 1.0
        return cls(
            feature_names=tuple(feature_names),
            medians=medians,
            means=means,
            scales=scales,
            constant_mask=constant,
            z_clip=float(z_clip),
        )

    def transform(self, values: Any) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("numeric transform matrix shape does not match fitted features")
        matrix = np.where(np.isfinite(matrix), matrix, self.medians)
        transformed = np.clip(
            (matrix - self.means) / self.scales,
            -self.z_clip,
            self.z_clip,
        )
        transformed[:, self.constant_mask] = 0.0
        return transformed.astype(np.float32, copy=False)

    def audit(self) -> dict[str, Any]:
        digest = sha256()
        for values in (self.medians, self.means, self.scales, self.constant_mask):
            digest.update(np.asarray(values).tobytes())
        return {
            "feature_count": len(self.feature_names),
            "constant_feature_count": int(self.constant_mask.sum()),
            "z_clip": self.z_clip,
            "transform_sha256": digest.hexdigest(),
        }


def grouped_softmax_numpy(
    utilities: np.ndarray, structure: GroupStructure
) -> np.ndarray:
    """Stable grouped softmax for a validated contiguous choice-set vector."""

    values = np.asarray(utilities, dtype=np.float64)
    if values.ndim != 1 or len(values) != structure.row_count:
        raise ValueError("utilities do not match grouped rows")
    if not np.isfinite(values).all():
        raise ValueError("utilities must be finite")
    result = np.empty_like(values)
    for start, end in structure.row_slices:
        shifted = values[start:end] - np.max(values[start:end])
        exponentials = np.exp(shifted)
        result[start:end] = exponentials / exponentials.sum()
    return result


def conditional_logit_loss_gradient(
    coefficients: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    structure: GroupStructure,
    *,
    l2: float,
) -> tuple[float, np.ndarray]:
    """Mean choice-set cross-entropy and exact linear-utility gradient."""

    beta = np.asarray(coefficients, dtype=np.float64)
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if matrix.shape != (structure.row_count, len(beta)):
        raise ValueError("linear feature matrix does not match groups/coefficients")
    if target.shape != (structure.row_count,):
        raise ValueError("target vector does not match groups")
    if l2 < 0 or not np.isfinite(l2):
        raise ValueError("l2 must be finite and non-negative")
    probabilities = grouped_softmax_numpy(matrix @ beta, structure)
    positive = target > 0
    loss = -float(np.sum(target[positive] * np.log(probabilities[positive])))
    loss /= len(structure.group_sizes)
    loss += 0.5 * l2 * float(beta @ beta)
    gradient = matrix.T @ (probabilities - target)
    gradient /= len(structure.group_sizes)
    gradient += l2 * beta
    return loss, np.asarray(gradient, dtype=np.float64)


def _binary_loss_gradient(
    coefficients: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    l2: float,
) -> tuple[float, np.ndarray]:
    beta = np.asarray(coefficients, dtype=np.float64)
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    logits = matrix @ beta
    probabilities = np.empty_like(logits)
    positive = logits >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    negative_exp = np.exp(logits[~positive])
    probabilities[~positive] = negative_exp / (1.0 + negative_exp)
    loss_terms = np.logaddexp(0.0, logits) - target * logits
    denominator = float(weight.sum())
    loss = float(weight @ loss_terms) / denominator + 0.5 * l2 * float(beta @ beta)
    gradient = matrix.T @ (weight * (probabilities - target)) / denominator + l2 * beta
    return loss, np.asarray(gradient, dtype=np.float64)


@dataclass
class LinearUtilityModel:
    """Fitted conditional-logit or capacity-matched linear Binary utility."""

    kind: str
    feature_names: tuple[str, ...]
    transform: FrozenNumericTransform
    coefficients: np.ndarray
    l2: float
    optimization: dict[str, Any]

    def predict_utility(self, values: Any) -> np.ndarray:
        return np.asarray(self.transform.transform(values) @ self.coefficients, dtype=float)


def _fit_linear_candidate(
    train_x: np.ndarray,
    train_targets: np.ndarray,
    train_structure: GroupStructure,
    validation_x: np.ndarray,
    validation_targets: np.ndarray,
    validation_structure: GroupStructure,
    *,
    l2: float,
    max_iterations: int,
    ftol: float,
    gtol: float,
    kind: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.optimize import minimize  # type: ignore[import-not-found]

    initial = np.zeros(train_x.shape[1], dtype=np.float64)
    if kind == "conditional_logit":
        objective = lambda beta: conditional_logit_loss_gradient(  # noqa: E731
            beta, train_x, train_targets, train_structure, l2=l2
        )
    elif kind == "linear_binary":
        weights = np.asarray(
            race_balanced_weights(
                [race for race, size in zip(train_structure.race_ids, train_structure.group_sizes) for _ in range(size)]
            ),
            dtype=float,
        )
        objective = lambda beta: _binary_loss_gradient(  # noqa: E731
            beta, train_x, train_targets, weights, l2=l2
        )
    else:
        raise ValueError("unknown linear utility kind")
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": ftol, "gtol": gtol},
    )
    coefficients = np.asarray(result.x, dtype=np.float64)
    validation_utility = validation_x @ coefficients
    validation_probability = grouped_softmax_numpy(validation_utility, validation_structure)
    positive = validation_targets > 0
    validation_loss = -float(
        np.sum(validation_targets[positive] * np.log(validation_probability[positive]))
    ) / len(validation_structure.group_sizes)
    return coefficients, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "final_objective": float(result.fun),
        "gradient_norm": float(np.linalg.norm(result.jac)),
        "coefficient_norm": float(np.linalg.norm(coefficients)),
        "validation_native_race_log_loss": validation_loss,
        "l2": float(l2),
    }


def fit_linear_utility(
    train_values: Any,
    train_race_ids: Sequence[Any],
    train_finish_positions: Sequence[int],
    validation_values: Any,
    validation_race_ids: Sequence[Any],
    validation_finish_positions: Sequence[int],
    feature_names: Sequence[str],
    *,
    kind: str = "conditional_logit",
    l2_grid: Sequence[float] = (1e-4, 1e-3, 1e-2),
    z_clip: float = 10.0,
    max_iterations: int = 250,
    ftol: float = 1e-10,
    gtol: float = 1e-6,
) -> LinearUtilityModel:
    """Fit a train-transformed linear utility and select L2 on validation LL."""

    validate_prediction_feature_columns(feature_names)
    train_structure = validate_grouped_rows(train_race_ids)
    validation_structure = validate_grouped_rows(validation_race_ids)
    transform = FrozenNumericTransform.fit(train_values, feature_names, z_clip=z_clip)
    train_x = transform.transform(train_values)
    validation_x = transform.transform(validation_values)
    if kind == "conditional_logit":
        train_targets = np.asarray(
            winner_mass_targets(train_finish_positions, train_race_ids), dtype=float
        )
        validation_targets = np.asarray(
            winner_mass_targets(validation_finish_positions, validation_race_ids), dtype=float
        )
    elif kind == "linear_binary":
        train_targets = np.asarray(binary_win_targets(train_finish_positions), dtype=float)
        validation_targets = np.asarray(
            winner_mass_targets(validation_finish_positions, validation_race_ids), dtype=float
        )
    else:
        raise ValueError("kind must be conditional_logit or linear_binary")
    candidates: list[tuple[np.ndarray, dict[str, Any]]] = []
    for l2 in l2_grid:
        candidates.append(
            _fit_linear_candidate(
                train_x,
                train_targets,
                train_structure,
                validation_x,
                validation_targets,
                validation_structure,
                l2=float(l2),
                max_iterations=max_iterations,
                ftol=ftol,
                gtol=gtol,
                kind=kind,
            )
        )
    if not all(record[1]["success"] for record in candidates):
        failures = [record[1] for record in candidates if not record[1]["success"]]
        raise RuntimeError(f"linear utility optimizer failed: {failures}")
    coefficients, selected = min(
        candidates,
        key=lambda item: (
            item[1]["validation_native_race_log_loss"],
            -item[1]["l2"],
        ),
    )
    return LinearUtilityModel(
        kind=kind,
        feature_names=tuple(feature_names),
        transform=transform,
        coefficients=coefficients,
        l2=float(selected["l2"]),
        optimization={
            "selected": selected,
            "candidates": [record for _, record in candidates],
            "transform": transform.audit(),
        },
    )


def grouped_softmax_objective(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray | None,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact grouped-softmax gradient and diagonal Hessian approximation."""

    del weights
    structure = validate_grouped_rows(
        [group for group, size in enumerate(np.asarray(groups, dtype=int)) for _ in range(size)]
    )
    probabilities = grouped_softmax_numpy(np.asarray(predictions), structure)
    targets = np.asarray(labels, dtype=float)
    return probabilities - targets, np.maximum(probabilities * (1.0 - probabilities), 1e-6)


def grouped_softmax_metric(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray | None,
    groups: np.ndarray,
) -> tuple[str, float, bool]:
    del weights
    structure = validate_grouped_rows(
        [group for group, size in enumerate(np.asarray(groups, dtype=int)) for _ in range(size)]
    )
    probabilities = grouped_softmax_numpy(np.asarray(predictions), structure)
    targets = np.asarray(labels, dtype=float)
    positive = targets > 0
    loss = -float(np.sum(targets[positive] * np.log(probabilities[positive])))
    loss /= len(structure.group_sizes)
    return "race_log_loss", loss, False


def train_nonlinear_racewise(
    frame: Any,
    *,
    feature_columns: Sequence[str],
    train_split: str = "train",
    model_validation_split: str = "model_validation",
    race_id_column: str = "race_id",
    finish_position_column: str = "model_finish_position",
    split_column: str = "rolling_role",
    params: Mapping[str, Any] | None = None,
    early_stopping_rounds: int = 50,
) -> Any:
    """Fit nonlinear utility with grouped softmax cross-entropy."""

    import lightgbm as lgb  # type: ignore[import-not-found]

    validate_prediction_feature_columns(feature_columns)
    train = frame.loc[frame[split_column].eq(train_split)]
    validation = frame.loc[frame[split_column].eq(model_validation_split)]
    if train.empty or validation.empty:
        raise ValueError("nonlinear race-wise train/validation split is empty")
    train_groups = validate_grouped_rows(train[race_id_column].tolist())
    validation_groups = validate_grouped_rows(validation[race_id_column].tolist())
    train_targets = winner_mass_targets(
        train[finish_position_column].astype(int).tolist(), train[race_id_column].tolist()
    )
    validation_targets = winner_mass_targets(
        validation[finish_position_column].astype(int).tolist(),
        validation[race_id_column].tolist(),
    )
    model_params = dict(params or {})
    model_params.pop("objective", None)
    model_params.pop("metric", None)
    model_params.pop("class_weight", None)
    model_params["objective"] = grouped_softmax_objective
    model_params["metric"] = "None"
    model = lgb.LGBMRanker(**model_params)
    model.fit(
        train.loc[:, list(feature_columns)].astype("float32", copy=False),
        train_targets,
        group=list(train_groups.group_sizes),
        feature_name=list(feature_columns),
        eval_set=[
            (
                validation.loc[:, list(feature_columns)].astype("float32", copy=False),
                validation_targets,
            )
        ],
        eval_group=[list(validation_groups.group_sizes)],
        eval_metric=grouped_softmax_metric,
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model


def predict_racewise_utility(
    model: Any,
    frame: Any,
    feature_columns: Sequence[str],
) -> np.ndarray:
    """Return raw utilities from a fitted linear or nonlinear model."""

    validate_prediction_feature_columns(feature_columns)
    values = frame.loc[:, list(feature_columns)].astype("float32", copy=False)
    if isinstance(model, LinearUtilityModel):
        return model.predict_utility(values)
    return np.asarray(model.predict(values), dtype=float)


def native_race_probability(utilities: Sequence[float], race_ids: Sequence[Any]) -> np.ndarray:
    return np.asarray(race_softmax(utilities, race_ids), dtype=float)
