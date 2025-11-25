"""
Trading Metrics Module

This module provides various metrics for evaluating trading agent performance,
including accuracy, precision, recall, Sharpe ratio, and other financial metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class TradingMetrics:
    """Container for trading evaluation metrics."""
    
    # Decision accuracy metrics
    total_decisions: int = 0
    correct_decisions: int = 0
    accuracy: float = 0.0
    
    # Per-class metrics
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    f1_score: Dict[str, float] = field(default_factory=dict)
    
    # Financial performance metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    
    # Confusion matrix
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "decision_accuracy": {
                "total_decisions": self.total_decisions,
                "correct_decisions": self.correct_decisions,
                "accuracy": self.accuracy,
            },
            "per_class_metrics": {
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
            },
            "financial_performance": {
                "total_return": self.total_return,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "profit_factor": self.profit_factor,
                "win_rate": self.win_rate,
            },
            "trade_statistics": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "average_win": self.average_win,
                "average_loss": self.average_loss,
            },
            "confusion_matrix": self.confusion_matrix,
        }


def calculate_accuracy(
    predictions: List[str], 
    actual_outcomes: List[str]
) -> Tuple[float, int, int]:
    """
    Calculate decision accuracy.
    
    Args:
        predictions: List of predicted decisions (BUY, SELL, HOLD)
        actual_outcomes: List of what the optimal decision would have been
        
    Returns:
        Tuple of (accuracy, correct_count, total_count)
    """
    if len(predictions) != len(actual_outcomes):
        raise ValueError("Predictions and actual outcomes must have the same length")
    
    if len(predictions) == 0:
        return 0.0, 0, 0
    
    correct = sum(
        1 for pred, actual in zip(predictions, actual_outcomes) 
        if pred.upper() == actual.upper()
    )
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0
    
    return accuracy, correct, total


def calculate_precision_recall(
    predictions: List[str],
    actual_outcomes: List[str],
    labels: Optional[List[str]] = None
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Calculate precision, recall, and F1 score for each class.
    
    Args:
        predictions: List of predicted decisions
        actual_outcomes: List of actual optimal decisions
        labels: List of class labels (default: ["BUY", "SELL", "HOLD"])
        
    Returns:
        Tuple of (precision_dict, recall_dict, f1_dict)
    """
    if labels is None:
        labels = ["BUY", "SELL", "HOLD"]
    
    # Normalize to uppercase
    predictions = [p.upper() for p in predictions]
    actual_outcomes = [a.upper() for a in actual_outcomes]
    
    precision = {}
    recall = {}
    f1_score = {}
    
    for label in labels:
        # True positives: predicted label and actual is label
        tp = sum(1 for p, a in zip(predictions, actual_outcomes) 
                 if p == label and a == label)
        
        # False positives: predicted label but actual is not label
        fp = sum(1 for p, a in zip(predictions, actual_outcomes) 
                 if p == label and a != label)
        
        # False negatives: actual is label but predicted something else
        fn = sum(1 for p, a in zip(predictions, actual_outcomes) 
                 if p != label and a == label)
        
        # Calculate metrics
        precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[label] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 score
        if precision[label] + recall[label] > 0:
            f1_score[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label])
        else:
            f1_score[label] = 0.0
    
    return precision, recall, f1_score


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: List of period returns (as decimals, e.g., 0.05 for 5%)
        risk_free_rate: Annual risk-free rate (default: 0)
        periods_per_year: Number of trading periods per year (default: 252 for daily)
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    
    # Calculate excess returns
    period_risk_free = risk_free_rate / periods_per_year
    excess_returns = returns_array - period_risk_free
    
    # Calculate mean and std of excess returns
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)  # Sample std
    
    if std_excess == 0:
        return 0.0
    
    # Annualize the Sharpe ratio
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    
    return float(sharpe)


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate the maximum drawdown from an equity curve.
    
    Args:
        equity_curve: List of portfolio values over time
        
    Returns:
        Maximum drawdown as a decimal (e.g., 0.15 for 15%)
    """
    if len(equity_curve) < 2:
        return 0.0
    
    equity_array = np.array(equity_curve)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_array)
    
    # Calculate drawdowns
    drawdowns = (running_max - equity_array) / running_max
    
    # Return maximum drawdown
    max_dd = np.max(drawdowns)
    
    return float(max_dd)


def calculate_profit_factor(
    winning_trades: List[float],
    losing_trades: List[float]
) -> float:
    """
    Calculate the profit factor (gross profits / gross losses).
    
    Args:
        winning_trades: List of profits from winning trades (positive values)
        losing_trades: List of losses from losing trades (positive values)
        
    Returns:
        Profit factor (>1 means profitable)
    """
    gross_profit = sum(winning_trades) if winning_trades else 0.0
    gross_loss = sum(abs(l) for l in losing_trades) if losing_trades else 0.0
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def build_confusion_matrix(
    predictions: List[str],
    actual_outcomes: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, Dict[str, int]]:
    """
    Build a confusion matrix for trading decisions.
    
    Args:
        predictions: List of predicted decisions
        actual_outcomes: List of actual optimal decisions
        labels: List of class labels (default: ["BUY", "SELL", "HOLD"])
        
    Returns:
        Nested dict representing confusion matrix [actual][predicted] = count
    """
    if labels is None:
        labels = ["BUY", "SELL", "HOLD"]
    
    # Normalize to uppercase
    predictions = [p.upper() for p in predictions]
    actual_outcomes = [a.upper() for a in actual_outcomes]
    
    # Initialize confusion matrix
    matrix = {actual: {pred: 0 for pred in labels} for actual in labels}
    
    # Fill confusion matrix
    for pred, actual in zip(predictions, actual_outcomes):
        if actual in matrix and pred in matrix[actual]:
            matrix[actual][pred] += 1
    
    return matrix


def calculate_returns_from_decisions(
    decisions: List[str],
    price_changes: List[float]
) -> List[float]:
    """
    Calculate returns based on trading decisions and price changes.
    
    Args:
        decisions: List of decisions (BUY, SELL, HOLD)
        price_changes: List of price changes (as percentages/decimals)
        
    Returns:
        List of returns achieved based on decisions
    """
    if len(decisions) != len(price_changes):
        raise ValueError("Decisions and price changes must have the same length")
    
    returns = []
    for decision, price_change in zip(decisions, price_changes):
        decision_upper = decision.upper()
        
        if decision_upper == "BUY":
            # Long position profits from price increase
            returns.append(price_change)
        elif decision_upper == "SELL":
            # Short position profits from price decrease
            returns.append(-price_change)
        else:  # HOLD
            # No position, no return
            returns.append(0.0)
    
    return returns
