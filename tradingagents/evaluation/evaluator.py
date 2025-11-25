"""
Evaluator Module

This module provides the main Evaluator class that integrates the TradingAgentsGraph
with backtesting and metrics calculation for comprehensive performance evaluation.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from .backtester import Backtester
from .metrics import (
    TradingMetrics,
    calculate_accuracy,
    calculate_precision_recall,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    build_confusion_matrix,
)


class Evaluator:
    """
    Main evaluator class for the TradingAgents framework.
    
    This class orchestrates the evaluation process by running the trading agent
    on historical data, comparing predictions with actual outcomes, and calculating
    comprehensive performance metrics.
    """
    
    def __init__(
        self,
        backtester: Optional[Backtester] = None,
        results_dir: str = "./evaluation_results",
        horizon_days: int = 1,
        decision_threshold: float = 0.01
    ):
        """
        Initialize the evaluator.
        
        Args:
            backtester: Backtester instance (creates new one if None)
            results_dir: Directory to save evaluation results
            horizon_days: Number of trading days to look forward for evaluation
            decision_threshold: Minimum price change to warrant BUY/SELL
        """
        self.backtester = backtester or Backtester()
        self.results_dir = results_dir
        self.horizon_days = horizon_days
        self.decision_threshold = decision_threshold
        
        # Create results directory
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
    
    def evaluate_single_decision(
        self,
        ticker: str,
        trade_date: str,
        decision: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single trading decision.
        
        Args:
            ticker: Stock ticker symbol
            trade_date: Date of the decision
            decision: The agent's decision (BUY, SELL, HOLD)
            
        Returns:
            Dictionary with evaluation results
        """
        return self.backtester.run_single_evaluation(
            ticker,
            trade_date,
            decision,
            self.horizon_days,
            self.decision_threshold
        )
    
    def evaluate_decision_series(
        self,
        ticker: str,
        decisions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a series of trading decisions.
        
        Args:
            ticker: Stock ticker symbol
            decisions: List of dicts with 'date' and 'decision' keys
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        # Run backtest
        backtest_results = self.backtester.backtest_decisions(
            ticker,
            decisions,
            self.horizon_days,
            self.decision_threshold
        )
        
        # Calculate metrics if we have valid predictions
        if backtest_results["predictions"] and backtest_results["optimal_decisions"]:
            metrics = self._calculate_full_metrics(backtest_results)
        else:
            metrics = TradingMetrics()
        
        return {
            "backtest_results": backtest_results,
            "metrics": metrics.to_dict(),
            "ticker": ticker,
            "evaluation_timestamp": datetime.now().isoformat(),
            "config": {
                "horizon_days": self.horizon_days,
                "decision_threshold": self.decision_threshold
            }
        }
    
    def run_evaluation(
        self,
        trading_agent_graph,
        ticker: str,
        start_date: str,
        end_date: str,
        skip_weekends: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Run a full evaluation by executing the trading agent over a date range.
        
        Args:
            trading_agent_graph: TradingAgentsGraph instance
            ticker: Stock ticker symbol to evaluate
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            skip_weekends: Whether to skip weekend dates
            progress_callback: Optional callback function(message, current, total)
            
        Returns:
            Comprehensive evaluation results
        """
        # Generate trading dates
        dates = self._generate_trading_dates(start_date, end_date, skip_weekends)
        
        decisions = []
        raw_outputs = []
        errors = []
        
        for i, trade_date in enumerate(dates):
            if progress_callback:
                progress_callback(f"Evaluating {ticker} on {trade_date}", i + 1, len(dates))
            
            try:
                # Run the trading agent
                final_state, decision = trading_agent_graph.propagate(ticker, trade_date)
                
                # Normalize decision
                decision_normalized = self._normalize_decision(decision)
                
                decisions.append({
                    "date": trade_date,
                    "decision": decision_normalized,
                    "raw_decision": decision
                })
                
                raw_outputs.append({
                    "date": trade_date,
                    "final_state": self._extract_state_summary(final_state),
                    "decision": decision_normalized
                })
                
            except Exception as e:
                errors.append({
                    "date": trade_date,
                    "error": str(e)
                })
        
        # Evaluate the decisions
        if decisions:
            evaluation_results = self.evaluate_decision_series(ticker, decisions)
        else:
            evaluation_results = {
                "backtest_results": {},
                "metrics": TradingMetrics().to_dict(),
                "ticker": ticker
            }
        
        # Compile full results
        full_results = {
            "evaluation_results": evaluation_results,
            "raw_outputs": raw_outputs,
            "errors": errors,
            "config": {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "dates_evaluated": len(dates),
                "horizon_days": self.horizon_days,
                "decision_threshold": self.decision_threshold
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return full_results
    
    def _calculate_full_metrics(self, backtest_results: Dict[str, Any]) -> TradingMetrics:
        """Calculate comprehensive metrics from backtest results."""
        predictions = backtest_results["predictions"]
        optimal = backtest_results["optimal_decisions"]
        returns = backtest_results["returns"]
        equity_curve = backtest_results["equity_curve"]
        
        metrics = TradingMetrics()
        
        # Decision accuracy
        accuracy, correct, total = calculate_accuracy(predictions, optimal)
        metrics.accuracy = accuracy
        metrics.correct_decisions = correct
        metrics.total_decisions = total
        
        # Precision, recall, F1
        precision, recall, f1 = calculate_precision_recall(predictions, optimal)
        metrics.precision = precision
        metrics.recall = recall
        metrics.f1_score = f1
        
        # Confusion matrix
        metrics.confusion_matrix = build_confusion_matrix(predictions, optimal)
        
        # Financial metrics
        if returns:
            metrics.sharpe_ratio = calculate_sharpe_ratio(returns)
            
            # Calculate total return
            total_return = 1.0
            for r in returns:
                total_return *= (1 + r)
            metrics.total_return = total_return - 1
            
            # Winning and losing trades
            winning = [r for r in returns if r > 0]
            losing = [r for r in returns if r < 0]
            
            metrics.winning_trades = len(winning)
            metrics.losing_trades = len(losing)
            metrics.total_trades = len([r for r in returns if r != 0])
            
            if winning:
                metrics.average_win = sum(winning) / len(winning)
            if losing:
                metrics.average_loss = sum(losing) / len(losing)
            
            if metrics.total_trades > 0:
                metrics.win_rate = metrics.winning_trades / metrics.total_trades
            
            # Profit factor
            metrics.profit_factor = calculate_profit_factor(winning, losing)
        
        # Max drawdown
        if equity_curve:
            metrics.max_drawdown = calculate_max_drawdown(equity_curve)
        
        return metrics
    
    def _generate_trading_dates(
        self, 
        start_date: str, 
        end_date: str,
        skip_weekends: bool
    ) -> List[str]:
        """Generate list of trading dates between start and end."""
        dates = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current <= end:
            if not skip_weekends or current.weekday() < 5:  # Monday=0, Friday=4
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        return dates
    
    def _normalize_decision(self, decision: str) -> str:
        """Normalize a decision string to BUY, SELL, or HOLD."""
        decision_upper = decision.upper().strip()
        
        if "BUY" in decision_upper:
            return "BUY"
        elif "SELL" in decision_upper:
            return "SELL"
        else:
            return "HOLD"
    
    def _extract_state_summary(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a summary from the final state for logging."""
        summary = {}
        
        # Extract key reports (limit size for storage)
        for key in ["market_report", "sentiment_report", "news_report", "fundamentals_report"]:
            if key in final_state:
                value = final_state[key]
                if isinstance(value, str) and len(value) > 500:
                    summary[key] = value[:500] + "..."
                else:
                    summary[key] = value
        
        # Extract decisions
        if "investment_plan" in final_state:
            summary["investment_plan"] = final_state["investment_plan"]
        if "final_trade_decision" in final_state:
            summary["final_trade_decision"] = final_state["final_trade_decision"]
        
        return summary
    
    def save_results(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Evaluation results dictionary
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            ticker = results.get("config", {}).get("ticker", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_{ticker}_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return filepath
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load evaluation results from a JSON file.
        
        Args:
            filepath: Path to the results file
            
        Returns:
            Evaluation results dictionary
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable report from evaluation results.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("TRADING AGENTS EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Configuration
        config = results.get("config", {})
        lines.append("CONFIGURATION")
        lines.append("-" * 30)
        lines.append(f"Ticker: {config.get('ticker', 'N/A')}")
        lines.append(f"Period: {config.get('start_date', 'N/A')} to {config.get('end_date', 'N/A')}")
        lines.append(f"Dates Evaluated: {config.get('dates_evaluated', 0)}")
        lines.append(f"Horizon Days: {config.get('horizon_days', 1)}")
        lines.append(f"Decision Threshold: {config.get('decision_threshold', 0.01):.2%}")
        lines.append("")
        
        # Metrics
        eval_results = results.get("evaluation_results", {})
        metrics = eval_results.get("metrics", {})
        
        # Decision Accuracy
        decision_acc = metrics.get("decision_accuracy", {})
        lines.append("DECISION ACCURACY")
        lines.append("-" * 30)
        lines.append(f"Total Decisions: {decision_acc.get('total_decisions', 0)}")
        lines.append(f"Correct Decisions: {decision_acc.get('correct_decisions', 0)}")
        lines.append(f"Accuracy: {decision_acc.get('accuracy', 0):.2%}")
        lines.append("")
        
        # Per-class metrics
        per_class = metrics.get("per_class_metrics", {})
        lines.append("PER-CLASS METRICS")
        lines.append("-" * 30)
        for label in ["BUY", "SELL", "HOLD"]:
            precision = per_class.get("precision", {}).get(label, 0)
            recall = per_class.get("recall", {}).get(label, 0)
            f1 = per_class.get("f1_score", {}).get(label, 0)
            lines.append(f"{label}: Precision={precision:.2%}, Recall={recall:.2%}, F1={f1:.2%}")
        lines.append("")
        
        # Financial performance
        fin_perf = metrics.get("financial_performance", {})
        lines.append("FINANCIAL PERFORMANCE")
        lines.append("-" * 30)
        lines.append(f"Total Return: {fin_perf.get('total_return', 0):.2%}")
        lines.append(f"Sharpe Ratio: {fin_perf.get('sharpe_ratio', 0):.2f}")
        lines.append(f"Max Drawdown: {fin_perf.get('max_drawdown', 0):.2%}")
        lines.append(f"Profit Factor: {fin_perf.get('profit_factor', 0):.2f}")
        lines.append(f"Win Rate: {fin_perf.get('win_rate', 0):.2%}")
        lines.append("")
        
        # Trade statistics
        trade_stats = metrics.get("trade_statistics", {})
        lines.append("TRADE STATISTICS")
        lines.append("-" * 30)
        lines.append(f"Total Trades: {trade_stats.get('total_trades', 0)}")
        lines.append(f"Winning Trades: {trade_stats.get('winning_trades', 0)}")
        lines.append(f"Losing Trades: {trade_stats.get('losing_trades', 0)}")
        lines.append(f"Average Win: {trade_stats.get('average_win', 0):.2%}")
        lines.append(f"Average Loss: {trade_stats.get('average_loss', 0):.2%}")
        lines.append("")
        
        # Errors
        errors = results.get("errors", [])
        if errors:
            lines.append("ERRORS")
            lines.append("-" * 30)
            for error in errors[:5]:  # Show first 5 errors
                lines.append(f"  {error.get('date', 'N/A')}: {error.get('error', 'Unknown')}")
            if len(errors) > 5:
                lines.append(f"  ... and {len(errors) - 5} more errors")
            lines.append("")
        
        lines.append("=" * 60)
        lines.append(f"Report generated: {results.get('timestamp', 'N/A')}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
