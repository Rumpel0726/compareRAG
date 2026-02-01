# -*- coding: utf-8 -*-
"""
Настройка логирования с использованием Loguru.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "./logs",
    rotation: str = "500 MB",
    retention: str = "10 days",
    console_output: bool = True
) -> None:
    """
    Настраивает систему логирования.

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_dir: Директория для логов
        rotation: Условие ротации логов
        retention: Время хранения логов
        console_output: Выводить ли логи в консоль
    """
    # Удаляем стандартный handler
    logger.remove()

    # Создаем директорию для логов если нужно
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Console handler
    if console_output:
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True
        )

    # File handler - основной лог
    logger.add(
        log_path / "rag_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        encoding="utf-8"
    )

    # File handler - лог ошибок
    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
        level="ERROR",
        rotation=rotation,
        retention=retention,
        compression="zip",
        encoding="utf-8"
    )

    # File handler - метрики в JSON формате
    logger.add(
        log_path / "metrics_{time:YYYY-MM-DD}.json",
        format="{message}",
        level="INFO",
        rotation="1 day",
        retention=retention,
        filter=lambda record: "metrics" in record["extra"],
        serialize=True
    )

    logger.info(f"Logging настроен: уровень={log_level}, директория={log_dir}")


def get_logger(name: Optional[str] = None):
    """
    Возвращает логгер с указанным именем.

    Args:
        name: Имя логгера (обычно __name__ модуля)

    Returns:
        Логгер
    """
    if name:
        return logger.bind(name=name)
    return logger


def log_metrics(metric_name: str, value: float, **kwargs):
    """
    Логирует метрику в JSON формат.

    Args:
        metric_name: Название метрики
        value: Значение метрики
        **kwargs: Дополнительные поля
    """
    logger.bind(metrics=True).info({
        "metric": metric_name,
        "value": value,
        **kwargs
    })


# Создаем глобальный логгер для удобства
rag_logger = logger.bind(name="RAG")
