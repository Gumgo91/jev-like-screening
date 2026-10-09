"""Dense screening and reversal diagnostics for development experiments.

Scores are lower-is-better; predictions are higher-is-better. Quadruplet
indices refer to rows (targets) and columns (ligands) of the dense arrays.
Null controls use the same public evaluator as real predictions. Reversal
quadruplets share targets and ligands: their count is never used as an
independent sample size for an uncertainty interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_INDEX_COLS = ("target_a", "target_b", "ligand_i", "ligand_j")
_QUAD_COLS = (*_INDEX_COLS, "d_a", "d_b")
_NULL_METRICS = ("joint_acc", "predicted_reversal_rate", "orientation_accuracy",
                 "sideA_acc", "sideB_acc", "tie_frac", "prediction_coverage")


def _dense(values, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty targets x ligands array")
    return array


def _mask(mask, shape):
    if mask is None:
        return np.ones(shape, dtype=bool)
    array = np.asarray(mask, dtype=bool)
    if array.shape != shape:
        raise ValueError("valid_mask must have the same shape as the dense array")
    return array


def _positive_count(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def sample_reversal_quads(scores, *, n_candidates=100_000, delta=1.0,
                          seed=0, valid_mask=None):
    """Sample candidates independently of predictions, then retain reversals.

    Sampling is with replacement, uniformly over ordered target/ligand indices.
    All four scores must be finite and unmasked, targets/ligands distinct,
    signs opposite, and both margins at least delta. This is label-conditioned,
    model-blind sampling, rather than score-blind sampling. Repeated quads are
    allowed; no independent-observation uncertainty claim follows from them.
    """
    score = _dense(scores, "scores")
    n_candidates = _positive_count(n_candidates, "n_candidates")
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    if min(score.shape) < 2:
        raise ValueError("reversal sampling needs at least two targets and ligands")
    valid = np.isfinite(score) & _mask(valid_mask, score.shape)
    rng = np.random.default_rng(seed)
    a = rng.integers(score.shape[0], size=n_candidates)
    b = rng.integers(score.shape[0], size=n_candidates)
    i = rng.integers(score.shape[1], size=n_candidates)
    j = rng.integers(score.shape[1], size=n_candidates)
    distinct = (a != b) & (i != j)
    complete = valid[a, i] & valid[a, j] & valid[b, i] & valid[b, j]
    # Avoid subtracting invalid infinities, and compare signs without products
    # that can overflow for large finite scores.
    safe = np.where(valid, score, 0.0)
    da, db = safe[a, i] - safe[a, j], safe[b, i] - safe[b, j]
    keep = (distinct & complete & np.isfinite(da) & np.isfinite(db)
            & (np.sign(da) == -np.sign(db)) & (da != 0) & (db != 0)
            & (np.minimum(np.abs(da), np.abs(db)) >= delta))
    frame = pd.DataFrame({"target_a": a[keep], "target_b": b[keep],
                          "ligand_i": i[keep], "ligand_j": j[keep],
                          "d_a": da[keep], "d_b": db[keep]})
    frame.attrs.update(n_candidates=n_candidates,
                       n_nondegenerate=int(distinct.sum()),
                       n_valid_candidates=int((distinct & complete).sum()),
                       n_reversals=int(keep.sum()), delta=float(delta), seed=seed)
    return frame


def reversal_details(predictions, quads, *, valid_mask=None):
    """Evaluate indexed quads, with exact ties wrong and missingness explicit.

    Reference-invalid rows (non-reversals, nonfinite margins, repeated target
    or ligand) are excluded from the eligible pool. A missing prediction makes
    the entire eligible quad unevaluable and incorrect under the conservative
    all-eligible denominator. Index/schema errors raise rather than silently
    changing the pool. No tolerance is used for prediction ties.
    """
    pred = _dense(predictions, "predictions")
    if not isinstance(quads, pd.DataFrame) or not set(_QUAD_COLS) <= set(quads):
        raise ValueError(f"quads must be a DataFrame containing {_QUAD_COLS}")
    raw = quads.loc[:, list(_INDEX_COLS)].to_numpy(dtype=float)
    if not np.isfinite(raw).all() or not (raw == np.floor(raw)).all():
        raise ValueError("quad indices must be finite integers")
    if (raw < 0).any() or (raw[:, :2] >= pred.shape[0]).any() \
            or (raw[:, 2:] >= pred.shape[1]).any():
        raise ValueError("quad index outside the dense prediction shape")
    a, b, i, j = raw.astype(np.int64).T
    da = quads["d_a"].to_numpy(dtype=float)
    db = quads["d_b"].to_numpy(dtype=float)
    reference_valid = (np.isfinite(da) & np.isfinite(db) & (da != 0) & (db != 0)
                       & (np.sign(da) == -np.sign(db)) & (a != b) & (i != j))
    available = np.isfinite(pred) & _mask(valid_mask, pred.shape)
    evaluated = (reference_valid & available[a, i] & available[a, j]
                 & available[b, i] & available[b, j])
    safe = np.where(available, pred, 0.0)
    # Ordering comparisons avoid overflow and make exact tie semantics clear.
    sign_a = (safe[a, i] > safe[a, j]).astype(int) \
        - (safe[a, i] < safe[a, j]).astype(int)
    sign_b = (safe[b, i] > safe[b, j]).astype(int) \
        - (safe[b, i] < safe[b, j]).astype(int)
    correct_a = evaluated & (sign_a == -np.sign(da))
    correct_b = evaluated & (sign_b == -np.sign(db))
    predicted_reversal = evaluated & (sign_a != 0) & (sign_a == -sign_b)
    result = quads.copy()
    result["reference_valid"] = reference_valid
    result["predictions_valid"] = evaluated
    result["sideA_correct"] = correct_a
    result["sideB_correct"] = correct_b
    result["joint_correct"] = correct_a & correct_b
    result["predicted_reversal"] = predicted_reversal
    result["tie"] = evaluated & ((sign_a == 0) | (sign_b == 0))
    return result


def reversal_diagnostics(predictions, quads, *, valid_mask=None):
    """Return joint fidelity, reversal incidence, and conditional orientation.

    Main rates use eligible reference quads; missing predictions count wrong
    rather than disappearing. ``joint_acc_scored`` additionally shows the
    complete-prediction subset. ``orientation_accuracy`` is conditional on
    actually predicting opposite strict orders, and is NaN when none occur.
    Positive joint accuracy alone does not establish above-chance target use:
    independent continuous random rankings per target have expectation 0.25.
    """
    details = reversal_details(predictions, quads, valid_mask=valid_mask)
    n = int(details.reference_valid.sum())
    evaluated = int(details.predictions_valid.sum())
    reversed_count = int(details.predicted_reversal.sum())
    joint_count = int(details.joint_correct.sum())

    def rate(count, denominator=n):
        return float(count / denominator) if denominator else float("nan")

    return {"n_quads": len(details), "n_eligible": n, "n_evaluated": evaluated,
            "n_invalid_reference": len(details) - n,
            "n_missing_predictions": n - evaluated,
            "n_predicted_reversals": reversed_count,
            "n_joint_correct": joint_count,
            "joint_acc": rate(joint_count),
            "joint_acc_scored": rate(joint_count, evaluated),
            "sideA_acc": rate(int(details.sideA_correct.sum())),
            "sideB_acc": rate(int(details.sideB_correct.sum())),
            "predicted_reversal_rate": rate(reversed_count),
            "orientation_accuracy": rate(joint_count, reversed_count),
            "tie_frac": rate(int(details.tie.sum())),
            "prediction_coverage": rate(evaluated),
            "denominator_policy": "eligible reference quads; missing predictions incorrect"}


def screening_diagnostics(scores, predictions, *, valid_mask=None,
                          hit_quantiles=(0.01, 0.05),
                          top_fractions=(0.01, 0.05, 0.10, 0.20),
                          ligand_ids=None):
    """Per-target screening recall, enrichment, ranking correlation, R95.

    Hits and budgets use ALL finite, unmasked labels. Missing predictions
    cannot be selected and do not shrink the hit denominator. Predictions tied
    at the budget boundary use ascending ligand_id (column index by default).
    Hit score ties are included, with their actual prevalence reported. Rank
    correlation alone uses the complete-prediction subset. Budgets follow the
    existing project's max(1, round(fraction*N)) convention.
    """
    score = _dense(scores, "scores")
    pred = _dense(predictions, "predictions")
    if score.shape != pred.shape:
        raise ValueError("scores and predictions must have identical shapes")
    qs, budgets = tuple(hit_quantiles), tuple(top_fractions)
    if not qs or not budgets or any(not np.isfinite(x) or not 0 < x <= 1
                                   for x in (*qs, *budgets)):
        raise ValueError("hit quantiles and top fractions must be in (0, 1]")
    ids = np.arange(score.shape[1]) if ligand_ids is None else np.asarray(ligand_ids)
    if ids.ndim != 1 or len(ids) != score.shape[1] or len(np.unique(ids)) != len(ids):
        raise ValueError("ligand_ids must contain one unique id per column")
    label_valid = np.isfinite(score) & _mask(valid_mask, score.shape)
    rows = []
    for target in range(score.shape[0]):
        valid = label_valid[target]
        n = int(valid.sum())
        scored = np.flatnonzero(valid & np.isfinite(pred[target]))
        ranked = scored[np.lexsort((ids[scored], -pred[target, scored]))]
        row = {"target_id": target, "n_ligands": n, "n_scored": len(scored),
               "n_missing_predictions": n - len(scored),
               "prediction_coverage": float(len(scored) / n) if n else np.nan}
        for q in qs:
            q_name = f"{q * 100:g}"
            threshold = float(np.quantile(score[target, valid], q)) if n else np.nan
            hits = valid & (score[target] <= threshold)
            n_hits = int(hits.sum())
            row[f"hit_rate_{q_name}"] = n_hits / n if n else np.nan
            row[f"hit_threshold_{q_name}"] = threshold
            for budget in budgets:
                k = max(1, int(round(budget * n))) if n else 0
                selected = ranked[:k]
                found = int(hits[selected].sum())
                suffix = f"{q_name}@{budget * 100:g}"
                recall = found / n_hits if n_hits else np.nan
                row[f"recall_{suffix}"] = recall
                # k is the intended budget, so failed predictions do not make
                # a reduced selection appear to have a better enrichment.
                row[f"ef{suffix}"] = (found / k) / (n_hits / n) if n_hits and k else np.nan
            need = int(np.ceil(0.95 * n_hits))
            cumulative = np.cumsum(hits[ranked])
            reached = np.flatnonzero(cumulative >= need) if need else np.array([])
            row[f"recall95_budget_{q_name}"] = ((int(reached[0]) + 1) / n
                                                if len(reached) else
                                                (np.inf if n_hits else np.nan))
        if len(scored) > 1:
            true_rank = pd.Series(-score[target, scored]).rank().to_numpy()
            pred_rank = pd.Series(pred[target, scored]).rank().to_numpy()
            row["spearman"] = (float(np.corrcoef(true_rank, pred_rank)[0, 1])
                               if np.std(true_rank) > 0 and np.std(pred_rank) > 0
                               else np.nan)
        else:
            row["spearman"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _null_summary(samples, observed=None):
    summary = {}
    for key in _NULL_METRICS:
        values = np.asarray([sample[key] for sample in samples], dtype=float)
        finite = values[np.isfinite(values)]
        entry = {"n_defined": len(finite),
                 "mean": float(finite.mean()) if len(finite) else np.nan,
                 "sd": float(finite.std(ddof=1)) if len(finite) > 1 else
                       (0.0 if len(finite) else np.nan),
                 "randomization_interval": np.quantile(finite, [0.025, 0.975]).tolist()
                                           if len(finite) else [np.nan, np.nan]}
        if observed is not None:
            entry["observed_minus_null_mean"] = float(observed[key] - entry["mean"])
        summary[key] = entry
    return summary


def independent_random_target_null(scores, quads, *, n_repeats=100,
                                   seed=0, valid_mask=None):
    """Continuous random scores independently drawn for each target/ligand.

    With complete predictions and eligible quads, expected joint=0.25,
    reversal frequency=0.5, orientation=0.5. These are mathematical null
    expectations, NOT a confidence interval from treating quads as independent.
    Each repetition is a whole random dense prediction matrix.
    """
    score = _dense(scores, "scores")
    n_repeats = _positive_count(n_repeats, "n_repeats")
    available = np.isfinite(score) & _mask(valid_mask, score.shape)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_repeats):
        prediction = rng.standard_normal(score.shape)
        samples.append(reversal_diagnostics(prediction, quads, valid_mask=available))
    coverage = samples[0]["prediction_coverage"]
    return {"seed": seed, "n_repeats": n_repeats,
            "expected_joint_acc_complete": 0.25,
            "expected_joint_acc": 0.25 * coverage,
            "expected_predicted_reversal_rate": 0.5 * coverage,
            "expected_orientation_accuracy": 0.5,
            "null_summary": _null_summary(samples), "samples": samples,
            "uncertainty_unit": "whole independently generated prediction matrix"}


def target_mapping_permutation_null(predictions, quads, *, n_repeats=100,
                                    seed=0, valid_mask=None, target_groups=None):
    """Permute whole prediction rows against fixed true target identities.

    Predictions, ligand rankings, and missingness within each row are preserved;
    the correct target-to-prediction correspondence is disrupted. Optional
    target_groups restrict permutations within groups. Identity permutations
    are allowed. Paired differences share the same eligible quad denominator.
    Intervals describe the chosen finite-set randomization, not population CIs;
    significance requires a justified target-exchangeability assumption.
    """
    pred = _dense(predictions, "predictions")
    n_repeats = _positive_count(n_repeats, "n_repeats")
    available = _mask(valid_mask, pred.shape)
    groups = np.zeros(pred.shape[0], dtype=int) if target_groups is None \
        else np.asarray(target_groups)
    if groups.ndim != 1 or len(groups) != pred.shape[0] or pd.isna(groups).any():
        raise ValueError("target_groups must have one nonmissing group per target")
    blocks = [np.flatnonzero(groups == group) for group in pd.unique(groups)]
    observed = reversal_diagnostics(pred, quads, valid_mask=available)
    rng = np.random.default_rng(seed)
    samples, permutations = [], []
    for _ in range(n_repeats):
        permutation = np.arange(pred.shape[0])
        for block in blocks:
            permutation[block] = rng.permutation(block)
        null = reversal_diagnostics(pred[permutation], quads,
                                    valid_mask=available[permutation])
        null["paired_joint_delta"] = observed["joint_acc"] - null["joint_acc"]
        null["paired_reversal_rate_delta"] = (observed["predicted_reversal_rate"]
                                               - null["predicted_reversal_rate"])
        null["paired_orientation_delta"] = (observed["orientation_accuracy"]
                                             - null["orientation_accuracy"])
        samples.append(null)
        permutations.append(permutation.tolist())
    return {"seed": seed, "n_repeats": n_repeats, "observed": observed,
            "n_permutable_targets": sum(len(block) for block in blocks if len(block) > 1),
            "null_summary": _null_summary(samples, observed),
            "samples": samples, "permutations": permutations,
            "uncertainty_unit": "whole target mapping, conditional on supplied targets",
            "interpretation": "finite-set randomization; population inference requires exchangeability"}


def paired_target_bootstrap_ci(values_a, values_b, *, n_bootstrap=2_000,
                               confidence=0.95, seed=0, target_ids=None):
    """Paired target-row resampling interval for a mean screening difference.

    Inputs must contain ONE summary per target per condition; do not pass quads,
    target pairs, or separate seed rows as independent targets. Average seeds
    within each target before calling. A and B are resampled together. This is
    a descriptive finite-set target-resampling interval: it neither establishes
    equivalence nor adjusts family, ligand/scaffold, or dyadic dependence, and
    is not a population confidence claim for biologically independent targets.
    """
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError("values_a/values_b must be equally sized nonempty target vectors")
    if target_ids is not None:
        ids = np.asarray(target_ids)
        if ids.ndim != 1 or len(ids) != len(a) or len(pd.unique(ids)) != len(ids) \
                or pd.isna(ids).any():
            raise ValueError("target_ids must be unique and nonmissing; one row per target")
    n_bootstrap = _positive_count(n_bootstrap, "n_bootstrap")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    complete = np.isfinite(a) & np.isfinite(b)
    differences = a[complete] - b[complete]
    if not len(differences):
        raise ValueError("at least one target needs both finite condition values")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, 1_024):
        stop = min(start + 1_024, n_bootstrap)
        draws = rng.integers(len(differences), size=(stop - start, len(differences)))
        means[start:stop] = differences[draws].mean(axis=1)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(means, [alpha, 1 - alpha])
    return {"estimate": float(differences.mean()), "lower": float(lower),
            "upper": float(upper), "confidence": confidence,
            "n_targets": len(differences), "n_dropped_targets": int((~complete).sum()),
            "n_bootstrap": n_bootstrap, "seed": seed, "difference": "a minus b",
            "resampling_unit": "paired target summary",
            "degenerate": bool(len(differences) == 1 or np.ptp(differences) == 0),
            "interpretation": "descriptive finite-set target-resampling interval",
            "limitations": "does not adjust family/scaffold/dyadic dependence; not an equivalence test"}
