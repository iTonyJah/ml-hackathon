from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from hackaton.train.cv import run_cv
from hackaton.train.training import TrainConfig, run_training


@click.group()
def cli() -> None:
    """CLI для baseline обучения."""


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
def train_cmd(
    user_path: Path,
    shift_path: Path,
    event_path: Path,
    output_dir: Path,
    random_state: int,
    max_iter: int,
    test_ratio: float,
    skip_shap: bool,
    shap_sample_size: int,
) -> None:
    """Запуск baseline train pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = TrainConfig(
        user_path=str(user_path),
        shift_path=str(shift_path),
        event_path=str(event_path),
        output_dir=str(output_dir),
        random_state=random_state,
        max_iter=max_iter,
        test_ratio=test_ratio,
        skip_shap=skip_shap,
        shap_sample_size=shap_sample_size,
    )
    result = run_training(cfg)
    click.echo("Обучение завершено успешно.")
    click.echo(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


@cli.command("cv")
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
@click.option(
    "--val-days", type=int, default=30, show_default=True, help="Размер валидационного окна в днях"
)
@click.option("--candidate-limit", type=int, default=300, show_default=True)
def cv_cmd(
    user_path: Path,
    shift_path: Path,
    event_path: Path,
    output_dir: Path,
    val_days: int,
    candidate_limit: int,
) -> None:
    """Time-based cross-validation без запуска сервиса."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    result = run_cv(
        user_path=str(user_path),
        shift_path=str(shift_path),
        event_path=str(event_path),
        val_days=val_days,
        candidate_limit=candidate_limit,
        output_dir=str(output_dir),
    )
    click.echo(
        f"CV завершен. overall_metric={result.overall_metric:.4f}"
        f"  evaluated_days={result.evaluated_days}"
        f"  evaluated_shifts={result.evaluated_shifts}"
    )


if __name__ == "__main__":
    cli()
