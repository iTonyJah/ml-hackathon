# Action Plan

## Case Summary

Build a service that receives a shift and returns TOP-10 `user_id` values that are most similar to users who will actually apply to the shift.

The metric is calculated by days, capacity groups, fixed TOP-10 pool size, and FPR limit `min(1.0, capacity / 10)`.

Key files:

- `README.md`
- `REGLAMENT.md`
- `hackaton/train/training.py`

Main growth area: offline training exists, but online inference is still almost rule-based.

## Plan

1. Prepare data:
   create `data/train` and `data/validation`, and put `user.csv`, `shift.csv`, `event.csv`, `apply.csv` there according to schemas from `DATA.md`.

2. Run baseline and get the starting metric:
   run `make install`, `make migrate`, `make run`, then training and eval from `README.md`. This gives the baseline comparison point: `artifacts/eval_run/eval_report.md`.

3. Fix the architecture gap:
   make `prepare` build or update features and load the model/artifacts, while `predict` uses scoring instead of only a SQL filter.

4. Improve candidate generation:
   for each shift, use an expanded pool instead of only same-location users: location, `has_mk`, `VIEW`/`APPLY`/`FINISHED` history, `task_type`, `employer_id`, `workplace_id` matches, user activity, cancellations, and reward/hour preferences.

5. Improve the model:
   replace `LogisticRegression` with a stronger tabular algorithm if dependencies allow it. Without new packages, use `HistGradientBoostingClassifier` plus careful categorical encodings. With dependency changes, consider CatBoost or LightGBM.

6. Synchronize offline and online features:
   move feature engineering out of `training.py` into a shared module so training and `predict` compute the same features. Otherwise offline metric will not reflect online behavior.

7. Optimize for the regulation metric:
   the metric depends on TOP-10 ordering, so the goal is ranking quality, not just classification. Validate through `hackaton/eval`, inspect quality by day and capacity, and separately check small capacity groups.

8. Add tests:
   cover `predict` for `need_mk`, location fallback, event history usage, stable TOP-10 size and ordering, and no 503 after `prepare`.

9. Check performance:
   `predict` must stay fast under `predict_max_rpm <= 200`, so heavy aggregations belong in `prepare`; `predict` should only load ready features and score a bounded candidate pool.

10. Final verification:
    run `make test`, `make precommit`, `make load-test`, then produce the final `eval_report.md` with overall metric, latency p95, and `stop_reason`.

## Current Notes

- `data/train`, `data/train_split`, and `data/validation` currently exist.
- Baseline eval reports currently exist under `artifacts/eval_baseline` and `artifacts/eval_strict_split`.
- `hackaton/train/training.py` still uses `LogisticRegression`.
- `hackaton/service/repositories.py` now has prepare-time aggregate rebuild and scored candidate selection.
- `hackaton/service/prepare_manager.py` can run async prepare callbacks and is wired to rebuild online features.
- Online inference now uses rule score plus ML reranker, expanded candidate pool, and ratio features for
  `APPLY/VIEW` and `FINISHED/APPLY` history by user, task, employer, and workplace.
- Best checked configuration for the presumed TOP-10 ROC-AUC@0.1 path is now the default:
  `ML_RERANKER_WEIGHT=0.5`, `CANDIDATE_POOL_LIMIT=1000`.
- Latest checked presumed TOP-10 ROC-AUC@0.1 on the local validation split: `0.815874`
  (`+0.061319` vs notebook baseline `0.754555`).
