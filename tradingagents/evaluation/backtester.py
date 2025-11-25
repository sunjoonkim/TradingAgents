"""
Backtester Module

This module provides backtesting functionality to evaluate trading agent
performance against historical market data.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import yfinance as yf
import pandas as pd


class Backtester:
    """
    Backtester for evaluating trading agent decisions against actual market outcomes.
    
    This class fetches historical price data and determines what the optimal
    trading decision would have been, allowing comparison with agent predictions.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001  # 0.1% per trade
    ):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital for backtesting
            transaction_cost: Transaction cost as a fraction (e.g., 0.001 for 0.1%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self._price_cache: Dict[str, pd.DataFrame] = {}
    
    def get_price_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{ticker}_{start_date}_{end_date}"
        
        if use_cache and cache_key in self._price_cache:
            return self._price_cache[cache_key]
        
        ticker_obj = yf.Ticker(ticker.upper())
        
        # Add buffer days to get data before start_date for returns calculation
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        buffered_start = (start_dt - timedelta(days=10)).strftime("%Y-%m-%d")
        
        # Add buffer days after end_date for forward-looking returns
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        buffered_end = (end_dt + timedelta(days=10)).strftime("%Y-%m-%d")
        
        data = ticker_obj.history(start=buffered_start, end=buffered_end)
        
        if data.empty:
            raise ValueError(f"No price data found for {ticker} between {start_date} and {end_date}")
        
        # Remove timezone info for cleaner handling
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        if use_cache:
            self._price_cache[cache_key] = data
        
        return data
    
    def get_price_change(
        self,
        ticker: str,
        trade_date: str,
        horizon_days: int = 1
    ) -> Tuple[float, float, float]:
        """
        Get the price change from the trade date to a future date.
        
        Args:
            ticker: Stock ticker symbol
            trade_date: The date the decision was made (YYYY-MM-DD)
            horizon_days: Number of trading days to look forward
            
        Returns:
            Tuple of (start_price, end_price, percent_change)
        """
        # Fetch data with appropriate buffer
        start_dt = datetime.strptime(trade_date, "%Y-%m-%d")
        end_buffer = start_dt + timedelta(days=horizon_days + 15)  # Buffer for weekends/holidays
        
        data = self.get_price_data(
            ticker,
            (start_dt - timedelta(days=5)).strftime("%Y-%m-%d"),
            end_buffer.strftime("%Y-%m-%d")
        )
        
        # Find the trade date or nearest trading day
        trade_date_dt = pd.Timestamp(trade_date)
        
        # Get dates on or after trade_date
        valid_dates = data.index[data.index >= trade_date_dt]
        
        if len(valid_dates) < horizon_days + 1:
            raise ValueError(f"Not enough data after {trade_date} for horizon of {horizon_days} days")
        
        start_price = data.loc[valid_dates[0], 'Close']
        end_price = data.loc[valid_dates[min(horizon_days, len(valid_dates) - 1)], 'Close']
        
        percent_change = (end_price - start_price) / start_price
        
        return float(start_price), float(end_price), float(percent_change)
    
    def determine_optimal_decision(
        self,
        price_change: float,
        threshold: float = 0.01
    ) -> str:
        """
        Determine what the optimal trading decision would have been.
        
        Args:
            price_change: The actual price change (as decimal)
            threshold: Minimum price change to warrant a BUY or SELL
            
        Returns:
            Optimal decision: "BUY", "SELL", or "HOLD"
        """
        if price_change > threshold:
            return "BUY"
        elif price_change < -threshold:
            return "SELL"
        else:
            return "HOLD"
    
    def backtest_decisions(
        self,
        ticker: str,
        decisions: List[Dict[str, Any]],
        horizon_days: int = 1,
        threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Backtest a series of trading decisions against actual market outcomes.
        
        Args:
            ticker: Stock ticker symbol
            decisions: List of dicts with 'date' and 'decision' keys
            horizon_days: Number of trading days to evaluate outcome
            threshold: Minimum price change threshold for optimal decision
            
        Returns:
            Dictionary containing backtest results
        """
        results = {
            "ticker": ticker,
            "decisions_evaluated": [],
            "predictions": [],
            "optimal_decisions": [],
            "price_changes": [],
            "returns": [],
            "equity_curve": [self.initial_capital],
            "summary": {}
        }
        
        current_capital = self.initial_capital
        
        for decision_data in decisions:
            trade_date = decision_data["date"]
            prediction = decision_data["decision"].upper()
            
            try:
                start_price, end_price, price_change = self.get_price_change(
                    ticker, trade_date, horizon_days
                )
                
                optimal = self.determine_optimal_decision(price_change, threshold)
                
                # Calculate return based on decision
                if prediction == "BUY":
                    trade_return = price_change - self.transaction_cost
                elif prediction == "SELL":
                    trade_return = -price_change - self.transaction_cost
                else:  # HOLD
                    trade_return = 0.0
                
                # Update capital
                current_capital *= (1 + trade_return)
                
                results["decisions_evaluated"].append({
                    "date": trade_date,
                    "prediction": prediction,
                    "optimal": optimal,
                    "is_correct": prediction == optimal,
                    "price_change": price_change,
                    "trade_return": trade_return,
                    "start_price": start_price,
                    "end_price": end_price
                })
                
                results["predictions"].append(prediction)
                results["optimal_decisions"].append(optimal)
                results["price_changes"].append(price_change)
                results["returns"].append(trade_return)
                results["equity_curve"].append(current_capital)
                
            except (ValueError, KeyError) as e:
                results["decisions_evaluated"].append({
                    "date": trade_date,
                    "prediction": prediction,
                    "optimal": None,
                    "is_correct": None,
                    "error": str(e)
                })
        
        # Calculate summary statistics
        valid_results = [d for d in results["decisions_evaluated"] if d.get("is_correct") is not None]
        
        if valid_results:
            correct_count = sum(1 for d in valid_results if d["is_correct"])
            results["summary"] = {
                "total_evaluated": len(valid_results),
                "correct_decisions": correct_count,
                "accuracy": correct_count / len(valid_results),
                "total_return": (current_capital - self.initial_capital) / self.initial_capital,
                "final_capital": current_capital,
                "initial_capital": self.initial_capital
            }
        
        return results
    
    def run_single_evaluation(
        self,
        ticker: str,
        trade_date: str,
        decision: str,
        horizon_days: int = 1,
        threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Evaluate a single trading decision.
        
        Args:
            ticker: Stock ticker symbol
            trade_date: Date of the decision (YYYY-MM-DD)
            decision: The agent's decision (BUY, SELL, HOLD)
            horizon_days: Number of trading days to evaluate
            threshold: Minimum price change threshold
            
        Returns:
            Dictionary with evaluation results
        """
        result = self.backtest_decisions(
            ticker,
            [{"date": trade_date, "decision": decision}],
            horizon_days,
            threshold
        )
        
        if result["decisions_evaluated"]:
            return result["decisions_evaluated"][0]
        else:
            return {"error": "No results generated"}
    
    def clear_cache(self):
        """Clear the price data cache."""
        self._price_cache.clear()
