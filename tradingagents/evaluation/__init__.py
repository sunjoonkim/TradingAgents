"""
TradingAgents Evaluation Framework

This module provides tools for evaluating trading agent performance through
backtesting, metrics calculation, and comprehensive analysis.
"""

from .metrics import (
    TradingMetrics,
    calculate_accuracy,
    calculate_precision_recall,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    build_confusion_matrix,
    calculate_returns_from_decisions,
)
from .backtester import Backtester
from .evaluator import Evaluator

__all__ = [
    "TradingMetrics",
    "calculate_accuracy",
    "calculate_precision_recall",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_profit_factor",
    "build_confusion_matrix",
    "calculate_returns_from_decisions",
    "Backtester",
    "Evaluator",
]
