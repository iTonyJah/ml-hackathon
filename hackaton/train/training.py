## ИМПОРТЫ
 # Стандартная библиотека
from __future__ import annotations          # аннотации типов как строки
import json, logging, pickle                # сериализация, логи, сохранение модели
from dataclasses import asdict, dataclass   # конфигурация как класс
from pathlib import Path                    # работа с путями

 # Библиотеки для анализа и визуализации
import matplotlib.pyplot as plt             # графики SHAP
import numpy as np                          # математика
import pandas as pd                         # работа с табличными данными
import shap                                 # интерпретация моделей

 # Scikit-learn: препроцессинг и пайплайны
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

 # Модели машинного обучения
from lightgbm import LGBMClassifier

 # Локальный модуль: расчёт метрики хакатона
from hackaton.eval.metric import calculate_target_metric

LOGGER = logging.getLogger(__name__)


## КОНСТАНТЫ
 # Какие колонки ОБЯЗАТЕЛЬНО должны быть в каждом файле
REQUIRED_USER_COLUMNS = [
    "location_id", 
    "is_strict_location", 
    "id", 
    "has_mk"
]
REQUIRED_SHIFT_COLUMNS = [
    "id",
    "start_at",
    "location_id",
    "task_type",
    "employer_id",
    "workplace_id",
    "need_mk",
    "id_differential",
    "hours",
    "reward",
    "capacity",
]
REQUIRED_EVENT_COLUMNS = [
    "id", 
    "shift_id", 
    "user_id", 
    "interaction", 
    "ts"
]
 # Какие значения взаимодействия считаются валидными
VALID_INTERACTIONS = {"VIEW", "APPLY", "FINISHED", "USER_CANCEL", "SYSTEM_CANCEL"}


## КОНФИГУРАЦИЯ ОБУЧЕНИЯ - TrainConfig
@dataclass(frozen=True, slots=True)
class TrainConfig:
    user_path: str
    shift_path: str
    event_path: str
    output_dir: str
    random_state: int = 42
    max_iter: int = 1000
    test_ratio: float = 0.2
    skip_shap: bool = False
    shap_sample_size: int = 1000
    # >>> НОВЫЕ ПОЛЯ <<<
    use_grid_search: bool = False
    cv_folds: int = 3
    scoring_metric: str = "roc_auc"
    n_jobs_grid: int = 2
    grid_search_verbose: int = 1


## ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

 # нормализация булевых значений 
 # преобразует "true"/"1"/"yes" → True, "false"/"0"/"no" → False
def _to_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(mapping)

 # проверка схемы данных - 
 # Проверяет: все ли обязательные колонки на месте? Считает пропуски (NaN) в каждой колонке
 # Возвращает словарь с отчётом для логирования.
def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> dict[str, object]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")
    null_counts = {c: int(df[c].isna().sum()) for c in required}
    return {
        "rows": int(len(df)),
        "required_columns_ok": True,
        "null_counts": null_counts,
    }

 # 1 ЭТАП - загрузка и очистка
def _load_and_validate_data(
    cfg: TrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    users = pd.read_csv(cfg.user_path)
    shifts = pd.read_csv(cfg.shift_path)
    events = pd.read_csv(cfg.event_path)

    checks = {
        "user": _validate_columns(users, REQUIRED_USER_COLUMNS, "user.csv"),
        "shift": _validate_columns(shifts, REQUIRED_SHIFT_COLUMNS, "shift.csv"),
        "event": _validate_columns(events, REQUIRED_EVENT_COLUMNS, "event.csv"),
    }

    users = users[REQUIRED_USER_COLUMNS].copy()
    shifts = shifts[REQUIRED_SHIFT_COLUMNS].copy()
    events = events[REQUIRED_EVENT_COLUMNS].copy()

    users["id"] = users["id"].astype(str)
    users["location_id"] = users["location_id"].astype(str)
    users["has_mk"] = _to_bool(users["has_mk"])
    users["is_strict_location"] = _to_bool(users["is_strict_location"])

    shifts["id"] = shifts["id"].astype(str)
    shifts["location_id"] = shifts["location_id"].astype(str)
    shifts["task_type"] = shifts["task_type"].astype(str)
    shifts["employer_id"] = shifts["employer_id"].astype(str)
    shifts["workplace_id"] = shifts["workplace_id"].astype(str)
    shifts["need_mk"] = _to_bool(shifts["need_mk"])
    shifts["id_differential"] = _to_bool(shifts["id_differential"])
    shifts["hours"] = pd.to_numeric(shifts["hours"], errors="coerce")
    shifts["reward"] = pd.to_numeric(shifts["reward"], errors="coerce")
    shifts["capacity"] = pd.to_numeric(shifts["capacity"], errors="coerce")
    shifts["start_at"] = pd.to_datetime(shifts["start_at"], utc=True, errors="coerce")

    events["id"] = events["id"].astype(str)
    events["shift_id"] = events["shift_id"].astype(str)
    events["user_id"] = events["user_id"].astype(str)
    events["interaction"] = events["interaction"].astype(str).str.upper()
    events["ts"] = pd.to_datetime(events["ts"], utc=True, errors="coerce")
    events = events[events["interaction"].isin(VALID_INTERACTIONS)]

    critical = {
        "users": ["id", "location_id", "has_mk", "is_strict_location"],
        "shifts": [
            "id",
            "start_at",
            "location_id",
            "need_mk",
            "id_differential",
            "hours",
            "reward",
            "capacity",
        ],
        "events": ["id", "shift_id", "user_id", "interaction", "ts"],
    }
    users = users.dropna(subset=critical["users"]).drop_duplicates(subset=["id"])
    shifts = shifts.dropna(subset=critical["shifts"]).drop_duplicates(subset=["id"])
    events = events.dropna(subset=critical["events"]).drop_duplicates(subset=["id"])

    checks["post_clean_rows"] = {
        "user": int(len(users)),
        "shift": int(len(shifts)),
        "event": int(len(events)),
    }
    return users, shifts, events, checks


 # 2 ЭТАП - Feature Engineering 
def _build_training_frame(
    users: pd.DataFrame, shifts: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    shifts_for_join = shifts.rename(columns={"id": "shift_id"}).copy()
    merged_events = events.merge(
        shifts_for_join[["shift_id", "start_at", "employer_id", "workplace_id"]],
        on="shift_id",
        how="inner",
    )
    # Отфильтровываем события ПОСЛЕ начала смены
    merged_events = merged_events[merged_events["ts"] <= merged_events["start_at"]].copy()

    # Агрегируем: сколько раз пользователь VIEW/APPLY/FINISHED и т.д
    grouped = merged_events.groupby(["user_id", "shift_id"], as_index=False).agg(
        first_ts=("ts", "min"),
        view_cnt=("interaction", lambda s: int((s == "VIEW").sum())),
        apply_cnt=("interaction", lambda s: int((s == "APPLY").sum())),
        finished_cnt=("interaction", lambda s: int((s == "FINISHED").sum())),
        user_cancel_cnt=("interaction", lambda s: int((s == "USER_CANCEL").sum())),
        system_cancel_cnt=("interaction", lambda s: int((s == "SYSTEM_CANCEL").sum())),
    )
    # Создаём target: 1 если было APPLY или FINISHED, иначе 0
    grouped["target"] = ((grouped["apply_cnt"] + grouped["finished_cnt"]) > 0).astype(int)

    #Считаем историю пользователя БЕЗ текущей смены (user_hist_*)
    user_totals = grouped.groupby("user_id", as_index=False).agg(
        user_total_views=("view_cnt", "sum"),
        user_total_applies=("apply_cnt", "sum"),
        user_total_finished=("finished_cnt", "sum"),
    )
    grouped = grouped.merge(user_totals, on="user_id", how="left")
    grouped["user_hist_views"] = grouped["user_total_views"] - grouped["view_cnt"]
    grouped["user_hist_applies"] = grouped["user_total_applies"] - grouped["apply_cnt"]
    grouped["user_hist_finished"] = grouped["user_total_finished"] - grouped["finished_cnt"]

    base = grouped.merge(shifts_for_join, on="shift_id", how="inner")
    base = base.merge(
        users, left_on="user_id", right_on="id", how="inner", suffixes=("_shift", "_user")
    )

    # Добавляем признаки совпадения: location_match, need_mk_match
    base["location_match"] = (base["location_id_shift"] == base["location_id_user"]).astype(int)
    base["need_mk_match"] = (base["need_mk"] == base["has_mk"]).astype(int)

    # Добавляем: сколько смен завершено у этого работодателя/места
    finished = merged_events[merged_events["interaction"] == "FINISHED"][
        ["user_id", "employer_id", "workplace_id"]
    ].copy()
    emp_finished = (
        finished.groupby(["user_id", "employer_id"], as_index=False)
        .size()
        .rename(columns={"size": "user_finished_employer"})
    )
    wp_finished = (
        finished.groupby(["user_id", "workplace_id"], as_index=False)
        .size()
        .rename(columns={"size": "user_finished_workplace"})
    )
    base = base.merge(emp_finished, on=["user_id", "employer_id"], how="left")
    base = base.merge(wp_finished, on=["user_id", "workplace_id"], how="left")
    base["user_finished_employer"] = base["user_finished_employer"].fillna(0)
    base["user_finished_workplace"] = base["user_finished_workplace"].fillna(0)
    return base

 # 3 ЭТАП - временное разделение на train/test
def _time_split(frame: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        raise ValueError("Training frame is empty after preprocessing.")
    unique_ts = np.array(sorted(frame["start_at"].dropna().unique()))
    if unique_ts.size < 2:
        raise ValueError("Not enough temporal points for 80/20 split.")
    split_idx = max(1, int(unique_ts.size * (1 - test_ratio)))
    split_idx = min(split_idx, unique_ts.size - 1)
    split_border = unique_ts[split_idx]
    train = frame[frame["start_at"] < split_border].copy()
    test = frame[frame["start_at"] >= split_border].copy()
    if train.empty or test.empty:
        raise ValueError("Time split produced empty train or test set.")
    return train, test

 # 4 ЭТАП - сборка пайплайна
def _build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    max_iter: int,
    use_grid_search: bool = False,
    cv_folds: int = 3,
    scoring_metric: str = "roc_auc",
    n_jobs_grid: int = 2,
    grid_search_verbose: int = 1,
) -> tuple[Pipeline | GridSearchCV, dict[str, object]]:
    """Строит пайплайн с LGBMClassifier + опциональный GridSearch."""
    
    # 1. Препроцессинг
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="passthrough"
    )
    
    # 2. >>> ВАЖНО: cv определяем СРАЗУ, до любых условий <<<
    cv = TimeSeriesSplit(n_splits=cv_folds)
    
    # 3. Базовая модель: LGBM
    base_model = LGBMClassifier(
        random_state=random_state,
        verbose=-1,
        n_jobs=-1,
        force_col_wise=True,
    )
    
    # 4. Сетка параметров
    param_grid = {
        "model__n_estimators": [50, 100, 150],
        "model__learning_rate": [0.05, 0.1, 0.15],
        "model__max_depth": [4, 6, 8, -1],
        "model__min_data_in_leaf": [10, 20, 30],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }
    
    # 5. >>> Если без GridSearch — возвращаем простой пайплайн <<<
    if not use_grid_search:
        base_model.set_params(
            n_estimators=min(max_iter, 100),
            learning_rate=0.1,
            max_depth=6,
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", base_model)])
        grid_info = {"grid_search_used": False}
        return pipeline, grid_info  # ← ранний возврат, cv здесь не нужен
    
    # 6. >>> GridSearch путь (cv уже определён выше) <<<
    base_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", base_model)])
    
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=cv,  # ← теперь cv точно определён
        scoring=scoring_metric,
        n_jobs=n_jobs_grid,
        verbose=grid_search_verbose,
        return_train_score=True,
    )
    
    grid_info = {
        "grid_search_used": True,
        "cv_folds": cv_folds,
        "scoring": scoring_metric,
        "param_grid": param_grid,
        "n_candidates": int(np.prod([len(v) for v in param_grid.values()])),
    }
    
    return grid_search, grid_info  # type: ignore

def _generate_shap_plots(
    pipeline: Pipeline | GridSearchCV,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    output_dir: Path,
    sample_size: int,
) -> dict[str, str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Извлекаем модель: если это GridSearch, берём best_estimator_
    if hasattr(pipeline, "best_estimator_"):
        actual_pipeline = pipeline.best_estimator_  # type: ignore
    else:
        actual_pipeline = pipeline  # type: ignore

    preprocessor = actual_pipeline.named_steps["preprocessor"]
    model = actual_pipeline.named_steps["model"]    

    # TreeExplainer работает с исходными данными (DataFrame)
    n = min(sample_size, len(x_test))
    x_test_sample = x_test.head(n).copy()
    
    # Выбираем explainer по типу модели
    model_name = model.__class__.__name__
    if model_name in ["LGBMClassifier", "RandomForestClassifier", "GradientBoostingClassifier"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test_sample)
        # Для бинарной классификации: берём класс 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_vals_array = shap_values
    else:
        # Линейные модели: нужно трансформировать данные
        x_test_arr = preprocessor.transform(x_test_sample)
        if hasattr(x_test_arr, "toarray"):
            x_test_arr = x_test_arr.toarray()
        explainer = shap.LinearExplainer(model, preprocessor.transform(x_train.head(min(sample_size, len(x_train)))))
        shap_vals = explainer(x_test_arr)
        shap_vals_array = shap_vals.values if hasattr(shap_vals, "values") else shap_vals
    
    # Получаем feature_names после OneHotEncoding
    feature_names = preprocessor.get_feature_names_out()
           
    # Сохранение графиков (без изменений)
    summary_path = plots_dir / "shap_summary.png"
    bar_path = plots_dir / "shap_bar.png"
    
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_vals_array, x_test_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=140)
    plt.close()
    
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_vals_array, x_test_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=140)
    plt.close()
    
    return {"shap_summary": str(summary_path), "shap_bar": str(bar_path)}


def run_training(cfg: TrainConfig) -> dict[str, object]:

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("ЭТАП 1: Загрузка и валидация данных")
    users, shifts, events, checks = _load_and_validate_data(cfg)
    LOGGER.info(
        "Loaded rows after cleanup: users=%s shifts=%s events=%s",
        len(users),
        len(shifts),
        len(events),
    )

    LOGGER.info("ЭТАП 2: Построение обучающей таблицы")
    frame = _build_training_frame(users, shifts, events)
    LOGGER.info("Built training frame rows=%s", len(frame))

    LOGGER.info("ЭТАП 3: Временной split на train/test")
    train_frame, test_frame = _time_split(frame, cfg.test_ratio)
    LOGGER.info("Split rows: train=%s test=%s", len(train_frame), len(test_frame))

    feature_columns = [
        "has_mk",
        "is_strict_location",
        "need_mk",
        "id_differential",
        "hours",
        "reward",
        "capacity",
        "location_match",
        "need_mk_match",
        "view_cnt",
        "user_cancel_cnt",
        "system_cancel_cnt",
        "user_hist_views",
        "user_hist_applies",
        "user_hist_finished",
        "user_finished_employer",
        "user_finished_workplace",
        "task_type",
    ]
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns after preprocessing: {missing}")

    numeric_features = [
        "hours",
        "reward",
        "capacity",
        "location_match",
        "need_mk_match",
        "view_cnt",
        "user_cancel_cnt",
        "system_cancel_cnt",
        "user_hist_views",
        "user_hist_applies",
        "user_hist_finished",
        "user_finished_employer",
        "user_finished_workplace",
        "has_mk",
        "is_strict_location",
        "need_mk",
        "id_differential",
    ]
    categorical_features = ["task_type"]

    x_train = train_frame[feature_columns].copy()
    x_test = test_frame[feature_columns].copy()
    for col in ["has_mk", "is_strict_location", "need_mk", "id_differential"]:
        x_train[col] = x_train[col].astype(int)
        x_test[col] = x_test[col].astype(int)
    y_train = train_frame["target"].astype(int)

    LOGGER.info("ЭТАП 4: Подготовка признаков")
    LOGGER.info("Feature columns: %s", ", ".join(feature_columns))
    LOGGER.info("Feature sample:\n%s", x_train.head(5).to_string(index=False))

    # """ EXTENSION POINT: swap baseline model/pipeline while keeping artifact contract stable. """

    LOGGER.info("ЭТАП 5: Обучение модели %s pipeline", "GridSearch+" if cfg.use_grid_search else "Baseline")
    
    pipeline_or_grid, grid_info = _build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=cfg.random_state,
        max_iter=cfg.max_iter,
        use_grid_search=cfg.use_grid_search,
        cv_folds=cfg.cv_folds,
        scoring_metric=cfg.scoring_metric,
        n_jobs_grid=cfg.n_jobs_grid,
        grid_search_verbose=cfg.grid_search_verbose,
    )
    
    # Fit: если это GridSearchCV, он сам обучит и выберет лучшую модель
    if cfg.use_grid_search:
        LOGGER.info("Starting GridSearchCV: %d candidates, %d folds", 
                   grid_info["n_candidates"], cfg.cv_folds)
        pipeline_or_grid.fit(x_train, y_train)  # type: ignore
        best_pipeline = pipeline_or_grid.best_estimator_  # type: ignore
        grid_results = {
            "best_params": pipeline_or_grid.best_params_,  # type: ignore
            "best_score": float(pipeline_or_grid.best_score_),  # type: ignore
            "best_index": int(pipeline_or_grid.best_index_),  # type: ignore
            "cv_results": {
                "mean_test_score": [float(s) for s in pipeline_or_grid.cv_results_["mean_test_score"]],  # type: ignore
                "std_test_score": [float(s) for s in pipeline_or_grid.cv_results_["std_test_score"]],  # type: ignore
                "params": pipeline_or_grid.cv_results_["params"],  # type: ignore
                "rank_test_score": [int(r) for r in pipeline_or_grid.cv_results_["rank_test_score"]],  # type: ignore
            },
            **grid_info
        }
        LOGGER.info("✅ GridSearch best score: %.4f", grid_results["best_score"])
        LOGGER.info("✅ GridSearch best params: %s", grid_results["best_params"])
    else:
        pipeline_or_grid.fit(x_train, y_train)  # type: ignore
        best_pipeline = pipeline_or_grid  # type: ignore
        grid_results = grid_info
    
    LOGGER.info("ЭТАП 5 завершен: model=%s, grid_search=%s", cfg.use_grid_search)


    LOGGER.info("ЭТАП 6: Инференс и расчёт метрики")
    
    # predict_proba работает одинаково для обычного пайплайна и best_estimator_
    proba = best_pipeline.predict_proba(x_test)[:, 1]  # type: ignore
    
    metric_df = test_frame[["shift_id", "start_at", "capacity", "target"]].copy()
    metric_df["score"] = proba
    metric_result = calculate_target_metric(metric_df)
    
    metrics = {
        "target_metric": metric_result.target_metric,
        "evaluated_days": metric_result.evaluated_days,
        "evaluated_groups": metric_result.evaluated_groups,
        "evaluated_shifts": metric_result.evaluated_shifts,
        "day_metrics": metric_result.day_metrics,
        "test_rows": int(len(test_frame)),
        "train_rows": int(len(train_frame)),
        "grid_search_used": cfg.use_grid_search,
    }
    if cfg.use_grid_search:
        metrics["grid_best_score"] = grid_results["best_score"]
        metrics["grid_best_params"] = {k: str(v) for k, v in grid_results["best_params"].items()}

    LOGGER.info("ЭТАП 7: Сохранение артефактов в %s", output_dir)
  
    # Сохраняем финальную модель (всегда best_pipeline)
    with (output_dir / "model.pkl").open("wb") as f:
        pickle.dump(best_pipeline, f)  # type: ignore
    
    # Если использовался грид-серч — сохраняем детали
    if cfg.use_grid_search:
        (output_dir / "grid_search_results.json").write_text(
            json.dumps(grid_results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8"
        )
        # Топ-5 комбинаций параметров для отчёта
        cv_df = pd.DataFrame({
            "mean_test_score": grid_results["cv_results"]["mean_test_score"],
            "std_test_score": grid_results["cv_results"]["std_test_score"],
            "rank": grid_results["cv_results"]["rank_test_score"],
            "params": [str(p) for p in grid_results["cv_results"]["params"]]
        }).sort_values("rank").head(5)
        (output_dir / "grid_search_top5.csv").write_text(
            cv_df.to_csv(index=False), encoding="utf-8"
        )
    
    # Остальные артефакты (без изменений)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "examples": x_train.head(5).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8"
    )
    (output_dir / "train_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "data_contract_check.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )


    report_lines = [
        "# Train Report",
        "",
        "## Data",
        "",
        f"- train_rows: {len(train_frame):,}",
        f"- test_rows: {len(test_frame):,}",
        "",
        "## Target metric (by regulation)",
        "",
        f"- target_metric: {metrics['target_metric']}",
        f"- evaluated_days: {metrics['evaluated_days']}",
        f"- evaluated_groups: {metrics['evaluated_groups']}",
        f"- evaluated_shifts: {metrics['evaluated_shifts']}",    
    ]
    if cfg.use_grid_search:
        report_lines.extend([
            "",
            "## Grid Search Results",
            "",
            f"- **Best CV Score** ({cfg.scoring_metric}): `{grid_results['best_score']:.4f}`",
            f"- **Best Parameters**:",
            "```json",
            json.dumps(grid_results["best_params"], indent=2, default=str),
            "```",
            "",
            "### Top 5 Parameter Combinations",
            "",
            "| Rank | Score | Std | Parameters |",
            "|------|-------|-----|------------|",
        ])
        cv_df = pd.DataFrame({
            "score": grid_results["cv_results"]["mean_test_score"],
            "std": grid_results["cv_results"]["std_test_score"],
            "rank": grid_results["cv_results"]["rank_test_score"],
            "params": [str(p) for p in grid_results["cv_results"]["params"]]
        }).sort_values("rank").head(5)
        for _, row in cv_df.iterrows():
            params_short = row["params"][:80] + "..." if len(row["params"]) > 80 else row["params"]
            report_lines.append(
                f"| {int(row['rank'])} | {row['score']:.4f} | ±{row['std']:.4f} | `{params_short}` |"
            )


    shap_result: dict[str, str] = {}
    if cfg.skip_shap:
        report_lines.extend(["", "## SHAP", "", "- SHAP skipped by config (--skip-shap)."])
    else:
        try:
            shap_result = _generate_shap_plots(
                best_pipeline, x_train, x_test, output_dir, cfg.shap_sample_size
            )
            report_lines.extend(
                [
                    "",
                    "## SHAP",
                    "",
                    f"- shap_summary: {shap_result['shap_summary']}",
                    f"- shap_bar: {shap_result['shap_bar']}",
                ]
            )
        except Exception as exc:  # noqa: BLE001
            skip_path = output_dir / "plots" / "shap_skipped.txt"
            skip_path.parent.mkdir(parents=True, exist_ok=True)
            skip_path.write_text(f"SHAP generation failed: {exc}", encoding="utf-8")
            report_lines.extend(["", "## SHAP", "", f"- SHAP generation failed: {exc}"])

    (output_dir / "train_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    LOGGER.info("ЭТАП 8: Отчёт и SHAP")
    return {"metrics": metrics, "shap": shap_result}
