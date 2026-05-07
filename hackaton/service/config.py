from dataclasses import dataclass
from os import getenv


def _env_float(name: str, default: float) -> float:
    value = getenv(name)
    if value is None:
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class Settings:
    app_host: str = getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(getenv("APP_PORT", "8000"))
    db_path: str = getenv("DB_PATH", "./data/hackaton.db")
    prepare_sleep_seconds: int = int(getenv("PREPARE_SLEEP_SECONDS", "10"))
    enable_ml_reranker: bool = _env_bool("ENABLE_ML_RERANKER", True)
    ml_reranker_weight: float = min(1.0, max(0.0, _env_float("ML_RERANKER_WEIGHT", 0.5)))
    candidate_pool_limit: int = max(10, int(getenv("CANDIDATE_POOL_LIMIT", "1000")))


settings = Settings()
