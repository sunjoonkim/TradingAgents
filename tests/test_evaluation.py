"""
Tests for the TradingAgents Evaluation Framework
"""

import pytest
from tradingagents.evaluation import (
    TradingMetrics,
    Backtester,
    Evaluator,
    calculate_accuracy,
    calculate_precision_recall,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    build_confusion_matrix,
    calculate_returns_from_decisions,
)


class TestCalculateAccuracy:
    """Tests for the calculate_accuracy function."""

    def test_perfect_accuracy(self):
        predictions = ["BUY", "SELL", "HOLD"]
        actual = ["BUY", "SELL", "HOLD"]
        accuracy, correct, total = calculate_accuracy(predictions, actual)
        assert accuracy == 1.0
        assert correct == 3
        assert total == 3

    def test_zero_accuracy(self):
        predictions = ["BUY", "BUY", "BUY"]
        actual = ["SELL", "SELL", "SELL"]
        accuracy, correct, total = calculate_accuracy(predictions, actual)
        assert accuracy == 0.0
        assert correct == 0
        assert total == 3

    def test_partial_accuracy(self):
        predictions = ["BUY", "SELL", "HOLD", "BUY"]
        actual = ["BUY", "BUY", "HOLD", "SELL"]
        accuracy, correct, total = calculate_accuracy(predictions, actual)
        assert accuracy == 0.5
        assert correct == 2
        assert total == 4

    def test_empty_lists(self):
        accuracy, correct, total = calculate_accuracy([], [])
        assert accuracy == 0.0
        assert correct == 0
        assert total == 0

    def test_case_insensitive(self):
        predictions = ["buy", "SELL", "Hold"]
        actual = ["BUY", "sell", "HOLD"]
        accuracy, correct, total = calculate_accuracy(predictions, actual)
        assert accuracy == 1.0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            calculate_accuracy(["BUY", "SELL"], ["BUY"])


class TestCalculatePrecisionRecall:
    """Tests for the calculate_precision_recall function."""

    def test_perfect_precision_recall(self):
        predictions = ["BUY", "SELL", "HOLD"]
        actual = ["BUY", "SELL", "HOLD"]
        precision, recall, f1 = calculate_precision_recall(predictions, actual)
        
        assert precision["BUY"] == 1.0
        assert precision["SELL"] == 1.0
        assert precision["HOLD"] == 1.0
        assert recall["BUY"] == 1.0
        assert recall["SELL"] == 1.0
        assert recall["HOLD"] == 1.0

    def test_no_predictions_for_class(self):
        predictions = ["BUY", "BUY", "BUY"]
        actual = ["BUY", "SELL", "HOLD"]
        precision, recall, f1 = calculate_precision_recall(predictions, actual)
        
        # SELL and HOLD have 0 precision (no predictions made for them)
        assert precision["SELL"] == 0.0
        assert precision["HOLD"] == 0.0
        # But they have 0 recall as well (they existed in actual but weren't predicted)
        assert recall["SELL"] == 0.0
        assert recall["HOLD"] == 0.0

    def test_f1_score_calculation(self):
        predictions = ["BUY", "BUY", "SELL"]
        actual = ["BUY", "SELL", "SELL"]
        precision, recall, f1 = calculate_precision_recall(predictions, actual)
        
        # For BUY: TP=1, FP=1, FN=0 -> P=0.5, R=1.0, F1=2*0.5*1/(0.5+1)=0.667
        assert abs(f1["BUY"] - 0.667) < 0.01


class TestCalculateSharpeRatio:
    """Tests for the calculate_sharpe_ratio function."""

    def test_positive_sharpe(self):
        returns = [0.01, 0.02, 0.01, 0.02, 0.01]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe > 0

    def test_negative_sharpe(self):
        returns = [-0.01, -0.02, -0.01, -0.02, -0.01]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe < 0

    def test_zero_volatility(self):
        returns = [0.01, 0.01, 0.01, 0.01]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0  # Zero std means undefined, returns 0

    def test_single_return(self):
        returns = [0.05]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0  # Not enough data

    def test_empty_returns(self):
        sharpe = calculate_sharpe_ratio([])
        assert sharpe == 0.0


class TestCalculateMaxDrawdown:
    """Tests for the calculate_max_drawdown function."""

    def test_no_drawdown(self):
        equity = [100, 110, 120, 130, 140]
        max_dd = calculate_max_drawdown(equity)
        assert max_dd == 0.0

    def test_with_drawdown(self):
        equity = [100, 110, 90, 100]  # Max drawdown is (110-90)/110 = 18.18%
        max_dd = calculate_max_drawdown(equity)
        assert abs(max_dd - 0.1818) < 0.01

    def test_multiple_drawdowns(self):
        equity = [100, 110, 95, 120, 100]  # First DD: (110-95)/110=13.6%, Second: (120-100)/120=16.7%
        max_dd = calculate_max_drawdown(equity)
        assert abs(max_dd - 0.167) < 0.01

    def test_single_value(self):
        max_dd = calculate_max_drawdown([100])
        assert max_dd == 0.0

    def test_empty_curve(self):
        max_dd = calculate_max_drawdown([])
        assert max_dd == 0.0


class TestCalculateProfitFactor:
    """Tests for the calculate_profit_factor function."""

    def test_profitable(self):
        winning = [100, 200, 150]
        losing = [50, 75]
        pf = calculate_profit_factor(winning, losing)
        # 450 / 125 = 3.6
        assert abs(pf - 3.6) < 0.01

    def test_no_losses(self):
        winning = [100, 200]
        losing = []
        pf = calculate_profit_factor(winning, losing)
        assert pf == float('inf')

    def test_no_wins(self):
        winning = []
        losing = [100, 200]
        pf = calculate_profit_factor(winning, losing)
        assert pf == 0.0

    def test_equal_wins_losses(self):
        winning = [100]
        losing = [100]
        pf = calculate_profit_factor(winning, losing)
        assert pf == 1.0


class TestBuildConfusionMatrix:
    """Tests for the build_confusion_matrix function."""

    def test_perfect_predictions(self):
        predictions = ["BUY", "SELL", "HOLD"]
        actual = ["BUY", "SELL", "HOLD"]
        matrix = build_confusion_matrix(predictions, actual)
        
        assert matrix["BUY"]["BUY"] == 1
        assert matrix["SELL"]["SELL"] == 1
        assert matrix["HOLD"]["HOLD"] == 1

    def test_all_wrong(self):
        predictions = ["BUY", "SELL", "HOLD"]
        actual = ["SELL", "HOLD", "BUY"]
        matrix = build_confusion_matrix(predictions, actual)
        
        # Diagonal should be zero
        assert matrix["BUY"]["BUY"] == 0
        assert matrix["SELL"]["SELL"] == 0
        assert matrix["HOLD"]["HOLD"] == 0
        
        # Off-diagonal entries
        assert matrix["SELL"]["BUY"] == 1  # actual=SELL, pred=BUY
        assert matrix["HOLD"]["SELL"] == 1  # actual=HOLD, pred=SELL
        assert matrix["BUY"]["HOLD"] == 1  # actual=BUY, pred=HOLD


class TestCalculateReturnsFromDecisions:
    """Tests for the calculate_returns_from_decisions function."""

    def test_buy_decisions(self):
        decisions = ["BUY", "BUY"]
        price_changes = [0.05, -0.03]
        returns = calculate_returns_from_decisions(decisions, price_changes)
        assert returns == [0.05, -0.03]

    def test_sell_decisions(self):
        decisions = ["SELL", "SELL"]
        price_changes = [0.05, -0.03]
        returns = calculate_returns_from_decisions(decisions, price_changes)
        assert returns == [-0.05, 0.03]  # Inverted for short

    def test_hold_decisions(self):
        decisions = ["HOLD", "HOLD"]
        price_changes = [0.05, -0.03]
        returns = calculate_returns_from_decisions(decisions, price_changes)
        assert returns == [0.0, 0.0]

    def test_mixed_decisions(self):
        decisions = ["BUY", "SELL", "HOLD"]
        price_changes = [0.05, 0.05, 0.05]
        returns = calculate_returns_from_decisions(decisions, price_changes)
        assert returns == [0.05, -0.05, 0.0]


class TestBacktester:
    """Tests for the Backtester class."""

    def test_determine_optimal_decision_buy(self):
        bt = Backtester()
        assert bt.determine_optimal_decision(0.05) == "BUY"

    def test_determine_optimal_decision_sell(self):
        bt = Backtester()
        assert bt.determine_optimal_decision(-0.05) == "SELL"

    def test_determine_optimal_decision_hold(self):
        bt = Backtester()
        assert bt.determine_optimal_decision(0.005) == "HOLD"

    def test_determine_optimal_decision_custom_threshold(self):
        bt = Backtester()
        assert bt.determine_optimal_decision(0.015, threshold=0.02) == "HOLD"
        assert bt.determine_optimal_decision(0.025, threshold=0.02) == "BUY"

    def test_clear_cache(self):
        bt = Backtester()
        bt._price_cache["test"] = "data"
        bt.clear_cache()
        assert len(bt._price_cache) == 0


class TestEvaluator:
    """Tests for the Evaluator class."""

    def test_normalize_decision_buy(self):
        evaluator = Evaluator(results_dir="/tmp/test_eval")
        assert evaluator._normalize_decision("BUY") == "BUY"
        assert evaluator._normalize_decision("buy now") == "BUY"
        assert evaluator._normalize_decision("BUY STRONG") == "BUY"

    def test_normalize_decision_sell(self):
        evaluator = Evaluator(results_dir="/tmp/test_eval")
        assert evaluator._normalize_decision("SELL") == "SELL"
        assert evaluator._normalize_decision("sell short") == "SELL"

    def test_normalize_decision_hold(self):
        evaluator = Evaluator(results_dir="/tmp/test_eval")
        assert evaluator._normalize_decision("HOLD") == "HOLD"
        assert evaluator._normalize_decision("wait and see") == "HOLD"

    def test_generate_trading_dates_weekdays(self):
        evaluator = Evaluator(results_dir="/tmp/test_eval")
        dates = evaluator._generate_trading_dates("2024-01-01", "2024-01-07", skip_weekends=True)
        # Jan 1 (Mon), 2 (Tue), 3 (Wed), 4 (Thu), 5 (Fri) - weekdays only
        assert len(dates) == 5
        assert "2024-01-06" not in dates  # Saturday
        assert "2024-01-07" not in dates  # Sunday

    def test_generate_trading_dates_all(self):
        evaluator = Evaluator(results_dir="/tmp/test_eval")
        dates = evaluator._generate_trading_dates("2024-01-01", "2024-01-07", skip_weekends=False)
        assert len(dates) == 7


class TestTradingMetrics:
    """Tests for the TradingMetrics dataclass."""

    def test_default_values(self):
        metrics = TradingMetrics()
        assert metrics.total_decisions == 0
        assert metrics.accuracy == 0.0
        assert metrics.sharpe_ratio == 0.0

    def test_to_dict(self):
        metrics = TradingMetrics(
            total_decisions=10,
            correct_decisions=7,
            accuracy=0.7
        )
        d = metrics.to_dict()
        assert d["decision_accuracy"]["total_decisions"] == 10
        assert d["decision_accuracy"]["accuracy"] == 0.7
