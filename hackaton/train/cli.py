from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from hackaton.train.training import TrainConfig, run_training


@click.group()
def cli() -> None:
    """CLI для baseline обучения."""
    pass


@cli.command("train")
@click.option(
    "--user-path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True
)
@click.option(
    "--shift-path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True
)
@click.option(
    "--event-path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True
)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--random-state", type=int, default=42, show_default=True)
@click.option("--max-iter", type=int, default=1000, show_default=True)
@click.option("--test-ratio", type=float, default=0.2, show_default=True)
@click.option("--skip-shap", is_flag=True, default=False)
@click.option("--shap-sample-size", type=int, default=1000, show_default=True)
# >>> НОВЫЕ ОПЦИИ <<<
@click.option("--model-type", default="lgbm", type=click.Choice(["lgbm", "rf", "logreg"]), show_default=True)
@click.option("--use-grid-search", is_flag=True, default=False, show_default=True)
@click.option("--cv-folds", default=3, type=int, show_default=True)
@click.option("--scoring-metric", default="roc_auc", type=str, show_default=True)
@click.option("--n-jobs-grid", default=2, type=int, show_default=True)
@click.option("--grid-search-verbose", default=1, type=int, show_default=True)
def train_cmd(
    user_path: Path, shift_path: Path, event_path: Path, output_dir: Path,
    random_state: int, max_iter: int, test_ratio: float, skip_shap: bool,
    shap_sample_size: int, model_type: str, use_grid_search: bool,
    cv_folds: int, scoring_metric: str, n_jobs_grid: int, grid_search_verbose: int,
) -> None:
    """Запуск train pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = TrainConfig(
        user_path=str(user_path), shift_path=str(shift_path), event_path=str(event_path),
        output_dir=str(output_dir), random_state=random_state, max_iter=max_iter,
        test_ratio=test_ratio, skip_shap=skip_shap, shap_sample_size=shap_sample_size,
        model_type=model_type, use_grid_search=use_grid_search, cv_folds=cv_folds,
        scoring_metric=scoring_metric, n_jobs_grid=n_jobs_grid, grid_search_verbose=grid_search_verbose,
    )
    result = run_training(cfg)
    click.echo("\n✅ Training finished.")
    click.echo(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
