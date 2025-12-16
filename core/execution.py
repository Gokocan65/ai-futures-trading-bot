"""
Execution Layer Module for AI Futures Trading Bot

This module provides comprehensive execution functionality including:
- Fill probability estimation based on market conditions
- Smart order routing for optimal execution
- Transaction cost analysis and optimization
- Adverse selection detection and mitigation
- Order management and execution tracking

Author: Gokocan65
Date: 2025-12-16
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Structures
# ============================================================================

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    POST_ONLY = "POST_ONLY"


class ExecutionVenue(Enum):
    """Available execution venues"""
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    DARK_POOL = "DARK_POOL"
    ALGORITHM = "ALGORITHM"


@dataclass
class OrderBook:
    """Order book snapshot"""
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    asks: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    
    def get_mid_price(self) -> float:
        """Calculate mid price from order book"""
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2.0
        return 0.0
    
    def get_spread(self) -> float:
        """Calculate bid-ask spread"""
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return 0.0
    
    def get_total_bid_volume(self, levels: int = 5) -> float:
        """Get total bid volume at specified levels"""
        return sum(size for _, size in self.bids[:levels])
    
    def get_total_ask_volume(self, levels: int = 5) -> float:
        """Get total ask volume at specified levels"""
        return sum(size for _, size in self.asks[:levels])


@dataclass
class ExecutionReport:
    """Execution report for an order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    executed_quantity: float
    executed_price: float
    execution_time: datetime
    venue: ExecutionVenue
    transaction_cost: float
    slippage: float
    fill_probability: float
    status: str = "FILLED"
    notes: str = ""


@dataclass
class MarketConditions:
    """Market conditions snapshot"""
    symbol: str
    timestamp: datetime
    price: float
    volatility: float
    volume: float
    spread: float
    depth: float
    trend: str  # "UP", "DOWN", "NEUTRAL"
    liquidity_score: float  # 0-1 scale


# ============================================================================
# Fill Probability Estimator
# ============================================================================

class FillProbabilityEstimator:
    """
    Estimates the probability of order fill based on market conditions,
    order characteristics, and historical patterns.
    """
    
    def __init__(self, history_periods: int = 100):
        """
        Initialize the Fill Probability Estimator
        
        Args:
            history_periods: Number of historical periods to consider
        """
        self.history_periods = history_periods
        self.fill_history: Dict[str, List[Dict]] = {}
        self.symbol_stats: Dict[str, Dict] = {}
        
    def estimate_fill_probability(
        self,
        order_book: OrderBook,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.LIMIT
    ) -> float:
        """
        Estimate the probability of order fill
        
        Args:
            order_book: Current order book
            side: Order side (BUY or SELL)
            quantity: Order quantity
            price: Limit price (None for market orders)
            order_type: Type of order
            
        Returns:
            Fill probability between 0 and 1
        """
        
        if order_type == OrderType.MARKET:
            return 0.99  # Market orders have very high fill probability
        
        if order_type == OrderType.FOK:
            return self._estimate_fok_probability(order_book, side, quantity)
        
        if price is None:
            mid_price = order_book.get_mid_price()
            price = mid_price if side == OrderSide.BUY else mid_price
        
        # Calculate fill probability based on multiple factors
        factors = {
            'position_in_book': self._calculate_position_factor(
                order_book, side, quantity, price
            ),
            'volume_availability': self._calculate_volume_factor(
                order_book, side, quantity
            ),
            'price_aggressiveness': self._calculate_price_aggressiveness(
                order_book, side, price
            ),
            'time_factor': 0.95,  # Decays over time
            'volatility_factor': self._calculate_volatility_impact(order_book)
        }
        
        # Weighted combination of factors
        weights = {
            'position_in_book': 0.3,
            'volume_availability': 0.25,
            'price_aggressiveness': 0.25,
            'time_factor': 0.1,
            'volatility_factor': 0.1
        }
        
        fill_probability = sum(
            factors[key] * weights[key] for key in factors
        )
        
        return np.clip(fill_probability, 0.0, 1.0)
    
    def _calculate_position_factor(
        self,
        order_book: OrderBook,
        side: OrderSide,
        quantity: float,
        price: float
    ) -> float:
        """Calculate how deep in the order book the order sits"""
        
        if side == OrderSide.BUY:
            # For buy orders, check ask side
            total_size = sum(size for p, size in order_book.asks if p <= price)
            best_ask = order_book.asks[0][0] if order_book.asks else 0
            
            if best_ask == 0:
                return 0.5
            
            # Higher probability if order is closer to best ask
            price_distance_pct = (price - best_ask) / best_ask
            position_score = 1.0 / (1.0 + price_distance_pct * 10)
            
        else:  # SELL
            # For sell orders, check bid side
            total_size = sum(size for p, size in order_book.bids if p >= price)
            best_bid = order_book.bids[0][0] if order_book.bids else 0
            
            if best_bid == 0:
                return 0.5
            
            # Higher probability if order is closer to best bid
            price_distance_pct = (best_bid - price) / best_bid
            position_score = 1.0 / (1.0 + price_distance_pct * 10)
        
        return position_score
    
    def _calculate_volume_factor(
        self,
        order_book: OrderBook,
        side: OrderSide,
        quantity: float
    ) -> float:
        """Calculate availability of volume at requested price"""
        
        if side == OrderSide.BUY:
            available_volume = order_book.get_total_ask_volume()
        else:
            available_volume = order_book.get_total_bid_volume()
        
        if available_volume == 0:
            return 0.5
        
        volume_ratio = min(quantity / available_volume, 1.0)
        return 1.0 - (volume_ratio * 0.5)
    
    def _calculate_price_aggressiveness(
        self,
        order_book: OrderBook,
        side: OrderSide,
        price: float
    ) -> float:
        """Calculate aggressiveness of the price"""
        
        mid_price = order_book.get_mid_price()
        if mid_price == 0:
            return 0.5
        
        if side == OrderSide.BUY:
            # How far above mid price are we willing to pay?
            aggressiveness = (price - mid_price) / mid_price
        else:
            # How far below mid price are we willing to sell?
            aggressiveness = (mid_price - price) / mid_price
        
        # More aggressive (higher aggressiveness) = higher fill probability
        return 1.0 / (1.0 + np.exp(-aggressiveness * 100))
    
    def _calculate_volatility_impact(self, order_book: OrderBook) -> float:
        """Calculate impact of volatility on fill probability"""
        
        spread = order_book.get_spread()
        mid_price = order_book.get_mid_price()
        
        if mid_price == 0:
            return 0.8
        
        spread_pct = (spread / mid_price) * 100
        
        # Wide spreads indicate high volatility = lower fill probability
        return max(0.5, 1.0 - (spread_pct * 0.01))
    
    def _estimate_fok_probability(
        self,
        order_book: OrderBook,
        side: OrderSide,
        quantity: float
    ) -> float:
        """Estimate probability of Fill or Kill order"""
        
        if side == OrderSide.BUY:
            available = sum(size for _, size in order_book.asks[:3])
        else:
            available = sum(size for _, size in order_book.bids[:3])
        
        return 1.0 if available >= quantity else 0.0
    
    def record_fill(
        self,
        symbol: str,
        order_book: OrderBook,
        estimated_prob: float,
        actual_filled: bool,
        execution_time: float
    ):
        """Record fill event for learning"""
        
        if symbol not in self.fill_history:
            self.fill_history[symbol] = []
        
        self.fill_history[symbol].append({
            'timestamp': datetime.utcnow(),
            'estimated_prob': estimated_prob,
            'actual_filled': actual_filled,
            'execution_time': execution_time,
            'spread': order_book.get_spread()
        })
        
        # Keep only recent history
        if len(self.fill_history[symbol]) > self.history_periods:
            self.fill_history[symbol].pop(0)


# ============================================================================
# Smart Order Router
# ============================================================================

class SmartOrderRouter:
    """
    Routes orders intelligently across multiple venues to minimize
    execution cost and maximize fill probability.
    """
    
    def __init__(self, venues: List[ExecutionVenue] = None):
        """
        Initialize the Smart Order Router
        
        Args:
            venues: List of available execution venues
        """
        self.venues = venues or [
            ExecutionVenue.PRIMARY,
            ExecutionVenue.SECONDARY,
            ExecutionVenue.DARK_POOL
        ]
        self.venue_metrics: Dict[ExecutionVenue, Dict] = {}
        self.routing_history: List[Dict] = []
        
        for venue in self.venues:
            self.venue_metrics[venue] = {
                'total_fills': 0,
                'total_volume': 0,
                'avg_fill_rate': 0.0,
                'avg_slippage': 0.0,
                'last_used': None
            }
    
    def route_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
        order_books: Dict[ExecutionVenue, OrderBook] = None
    ) -> Tuple[ExecutionVenue, Dict]:
        """
        Route an order to the optimal venue
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY or SELL)
            quantity: Order quantity
            price: Limit price
            order_type: Type of order
            order_books: Order books by venue
            
        Returns:
            Tuple of (selected_venue, routing_info)
        """
        
        if not order_books:
            order_books = {}
        
        venue_scores = {}
        
        for venue in self.venues:
            order_book = order_books.get(venue)
            
            if order_book is None:
                logger.warning(f"No order book data for venue {venue}")
                venue_scores[venue] = 0.0
                continue
            
            score = self._calculate_venue_score(
                venue,
                symbol,
                side,
                quantity,
                price,
                order_book,
                order_type
            )
            
            venue_scores[venue] = score
        
        # Select best venue
        best_venue = max(venue_scores, key=venue_scores.get)
        
        routing_info = {
            'symbol': symbol,
            'venue': best_venue,
            'venue_scores': venue_scores,
            'timestamp': datetime.utcnow(),
            'quantity': quantity,
            'side': side.value
        }
        
        self.routing_history.append(routing_info)
        
        logger.info(f"Routed {quantity} {symbol} {side.value} to {best_venue.value}")
        
        return best_venue, routing_info
    
    def _calculate_venue_score(
        self,
        venue: ExecutionVenue,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float],
        order_book: OrderBook,
        order_type: OrderType
    ) -> float:
        """Calculate composite score for a venue"""
        
        components = {
            'liquidity': self._score_liquidity(venue, order_book, quantity),
            'spread': self._score_spread(venue, order_book),
            'venue_quality': self._score_venue_quality(venue),
            'fill_probability': self._score_fill_probability(
                venue, order_book, side, quantity, price, order_type
            )
        }
        
        weights = {
            'liquidity': 0.3,
            'spread': 0.25,
            'venue_quality': 0.25,
            'fill_probability': 0.2
        }
        
        score = sum(components[key] * weights[key] for key in components)
        
        return score
    
    def _score_liquidity(
        self,
        venue: ExecutionVenue,
        order_book: OrderBook,
        quantity: float
    ) -> float:
        """Score venue based on liquidity"""
        
        bid_volume = order_book.get_total_bid_volume()
        ask_volume = order_book.get_total_ask_volume()
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return 0.0
        
        volume_ratio = quantity / total_volume
        
        # Penalty for requesting too much volume
        liquidity_score = 1.0 / (1.0 + volume_ratio * 5)
        
        return min(liquidity_score, 1.0)
    
    def _score_spread(self, venue: ExecutionVenue, order_book: OrderBook) -> float:
        """Score venue based on spread"""
        
        spread = order_book.get_spread()
        mid_price = order_book.get_mid_price()
        
        if mid_price == 0:
            return 0.5
        
        spread_pct = (spread / mid_price) * 100
        
        # Lower spread = higher score
        return max(0.0, 1.0 - (spread_pct * 0.1))
    
    def _score_venue_quality(self, venue: ExecutionVenue) -> float:
        """Score venue based on historical quality"""
        
        metrics = self.venue_metrics.get(venue, {})
        
        if metrics.get('total_fills', 0) == 0:
            return 0.7  # Default score for untested venues
        
        fill_rate = metrics.get('avg_fill_rate', 0.0)
        
        return fill_rate
    
    def _score_fill_probability(
        self,
        venue: ExecutionVenue,
        order_book: OrderBook,
        side: OrderSide,
        quantity: float,
        price: Optional[float],
        order_type: OrderType
    ) -> float:
        """Score venue based on fill probability"""
        
        # Venue-specific logic
        if venue == ExecutionVenue.DARK_POOL:
            # Dark pools typically have lower probability but better prices
            return 0.6
        elif venue == ExecutionVenue.PRIMARY:
            # Primary exchanges have high probability
            return 0.95
        else:
            # Secondary venues
            return 0.85
    
    def update_venue_metrics(
        self,
        venue: ExecutionVenue,
        filled: bool,
        slippage: float,
        volume: float
    ):
        """Update venue metrics based on execution"""
        
        metrics = self.venue_metrics[venue]
        metrics['total_fills'] += 1
        metrics['total_volume'] += volume
        
        # Update average fill rate
        old_avg = metrics['avg_fill_rate']
        metrics['avg_fill_rate'] = (
            old_avg * (metrics['total_fills'] - 1) +
            (1.0 if filled else 0.0)
        ) / metrics['total_fills']
        
        # Update average slippage
        old_slippage = metrics['avg_slippage']
        metrics['avg_slippage'] = (
            old_slippage * (metrics['total_fills'] - 1) + slippage
        ) / metrics['total_fills']
        
        metrics['last_used'] = datetime.utcnow()


# ============================================================================
# Transaction Cost Analyzer
# ============================================================================

class TransactionCostAnalyzer:
    """
    Analyzes and estimates transaction costs including commissions,
    slippage, and opportunity costs.
    """
    
    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_basis_points: float = 1.0
    ):
        """
        Initialize the Transaction Cost Analyzer
        
        Args:
            commission_rate: Commission as percentage (0.001 = 0.1%)
            slippage_basis_points: Expected slippage in basis points
        """
        self.commission_rate = commission_rate
        self.slippage_basis_points = slippage_basis_points
        self.cost_history: List[Dict] = []
    
    def estimate_total_cost(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        market_price: float,
        order_type: OrderType = OrderType.MARKET,
        market_impact: float = 0.0
    ) -> Dict:
        """
        Estimate total transaction costs
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            entry_price: Price at which we're executing
            market_price: Current market price (reference)
            order_type: Type of order
            market_impact: Additional market impact percentage
            
        Returns:
            Dictionary with cost breakdown
        """
        
        notional_value = quantity * market_price
        
        # Commission cost
        commission_cost = notional_value * self.commission_rate
        
        # Slippage cost
        slippage_cost = self._calculate_slippage(
            quantity,
            entry_price,
            market_price,
            order_type
        )
        
        # Market impact cost
        impact_cost = notional_value * market_impact
        
        # Opportunity cost (estimated)
        opportunity_cost = self._estimate_opportunity_cost(
            quantity,
            market_price,
            order_type
        )
        
        # Total cost
        total_cost = commission_cost + slippage_cost + impact_cost + opportunity_cost
        
        cost_breakdown = {
            'symbol': symbol,
            'quantity': quantity,
            'notional_value': notional_value,
            'commission_cost': commission_cost,
            'slippage_cost': slippage_cost,
            'market_impact_cost': impact_cost,
            'opportunity_cost': opportunity_cost,
            'total_cost': total_cost,
            'total_cost_bps': (total_cost / notional_value) * 10000 if notional_value > 0 else 0,
            'timestamp': datetime.utcnow()
        }
        
        self.cost_history.append(cost_breakdown)
        
        return cost_breakdown
    
    def _calculate_slippage(
        self,
        quantity: float,
        entry_price: float,
        market_price: float,
        order_type: OrderType
    ) -> float:
        """Calculate slippage cost"""
        
        # Base slippage
        base_slippage_pct = self.slippage_basis_points / 10000
        
        # Adjust for order type
        if order_type == OrderType.MARKET:
            slippage_multiplier = 1.5  # Market orders have more slippage
        elif order_type == OrderType.LIMIT:
            slippage_multiplier = 0.7  # Limit orders have less slippage
        elif order_type == OrderType.IOC:
            slippage_multiplier = 1.2
        else:
            slippage_multiplier = 1.0
        
        # Size-adjusted slippage
        # Larger orders typically have more slippage
        size_impact = 1.0 + (quantity * 0.001)  # Increase by 0.1% per unit
        
        effective_slippage = base_slippage_pct * slippage_multiplier * size_impact
        
        slippage_amount = quantity * market_price * effective_slippage
        
        return slippage_amount
    
    def _estimate_opportunity_cost(
        self,
        quantity: float,
        market_price: float,
        order_type: OrderType
    ) -> float:
        """Estimate opportunity cost of execution"""
        
        # Opportunity cost increases with time to fill
        if order_type == OrderType.MARKET:
            execution_time = 0.01  # Seconds
        elif order_type == OrderType.LIMIT:
            execution_time = 5.0  # Seconds
        elif order_type == OrderType.IOC:
            execution_time = 0.1
        else:
            execution_time = 1.0
        
        # Assume daily volatility of 1.5%
        annual_volatility = 0.015
        daily_volatility = annual_volatility / np.sqrt(252)
        
        # Price movement during execution
        time_fraction = execution_time / (24 * 3600)
        expected_move = market_price * daily_volatility * np.sqrt(time_fraction)
        
        opportunity_cost = quantity * expected_move
        
        return opportunity_cost
    
    def get_cost_statistics(self, lookback_period: int = 100) -> Dict:
        """Get statistics on transaction costs"""
        
        if not self.cost_history:
            return {}
        
        recent_costs = self.cost_history[-lookback_period:]
        
        costs_bps = [c['total_cost_bps'] for c in recent_costs]
        
        return {
            'avg_cost_bps': np.mean(costs_bps),
            'median_cost_bps': np.median(costs_bps),
            'min_cost_bps': np.min(costs_bps),
            'max_cost_bps': np.max(costs_bps),
            'std_cost_bps': np.std(costs_bps),
            'total_costs': sum(c['total_cost'] for c in recent_costs),
            'num_transactions': len(recent_costs)
        }


# ============================================================================
# Adverse Selection Detector
# ============================================================================

class AdverseSelectionDetector:
    """
    Detects and measures adverse selection events where execution
    prices move against the trader after order placement.
    """
    
    def __init__(self, lookback_period: int = 60):
        """
        Initialize the Adverse Selection Detector
        
        Args:
            lookback_period: Period in seconds to monitor after execution
        """
        self.lookback_period = lookback_period
        self.executions: Dict[str, Dict] = {}
        self.adverse_events: List[Dict] = []
    
    def register_execution(
        self,
        order_id: str,
        symbol: str,
        side: OrderSide,
        executed_price: float,
        quantity: float,
        execution_time: datetime
    ):
        """
        Register an execution to monitor for adverse selection
        
        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            side: Order side
            executed_price: Execution price
            quantity: Executed quantity
            execution_time: Time of execution
        """
        
        self.executions[order_id] = {
            'symbol': symbol,
            'side': side,
            'executed_price': executed_price,
            'quantity': quantity,
            'execution_time': execution_time,
            'prices_observed': [(execution_time, executed_price)],
            'adverse_moves': 0,
            'favorable_moves': 0
        }
    
    def update_price(
        self,
        order_id: str,
        price: float,
        timestamp: datetime
    ) -> Optional[Dict]:
        """
        Update price observation for an execution
        
        Args:
            order_id: Order identifier
            price: Current price
            timestamp: Time of observation
            
        Returns:
            Adverse selection event details if detected, None otherwise
        """
        
        if order_id not in self.executions:
            return None
        
        execution = self.executions[order_id]
        
        # Check if still within lookback period
        time_elapsed = (timestamp - execution['execution_time']).total_seconds()
        
        if time_elapsed > self.lookback_period:
            return None
        
        # Record price observation
        execution['prices_observed'].append((timestamp, price))
        
        # Detect adverse selection
        adverse_event = self._detect_adverse_selection(order_id, price)
        
        return adverse_event
    
    def _detect_adverse_selection(
        self,
        order_id: str,
        current_price: float
    ) -> Optional[Dict]:
        """Detect if adverse selection is occurring"""
        
        execution = self.executions[order_id]
        executed_price = execution['executed_price']
        side = execution['side']
        
        # Calculate price movement
        if side == OrderSide.BUY:
            # For buy orders, adverse if price drops after execution
            price_move = executed_price - current_price
            is_adverse = price_move > 0
        else:  # SELL
            # For sell orders, adverse if price rises after execution
            price_move = current_price - executed_price
            is_adverse = price_move > 0
        
        if is_adverse:
            execution['adverse_moves'] += 1
        else:
            execution['favorable_moves'] += 1
        
        # Check for significant adverse selection
        total_moves = execution['adverse_moves'] + execution['favorable_moves']
        adverse_ratio = execution['adverse_moves'] / total_moves if total_moves > 0 else 0
        
        if adverse_ratio > 0.65:  # Threshold for detection
            event = {
                'order_id': order_id,
                'symbol': execution['symbol'],
                'side': execution['side'],
                'executed_price': executed_price,
                'current_price': current_price,
                'price_move': price_move,
                'adverse_ratio': adverse_ratio,
                'quantity': execution['quantity'],
                'notional_loss': abs(price_move) * execution['quantity'],
                'timestamp': datetime.utcnow(),
                'severity': self._calculate_severity(price_move, executed_price)
            }
            
            self.adverse_events.append(event)
            
            logger.warning(
                f"Adverse selection detected for {order_id}: "
                f"ratio={adverse_ratio:.2%}, move={price_move:.4f}"
            )
            
            return event
        
        return None
    
    def _calculate_severity(self, price_move: float, executed_price: float) -> str:
        """Calculate severity of adverse selection"""
        
        move_pct = abs(price_move) / executed_price * 100
        
        if move_pct > 0.5:
            return "HIGH"
        elif move_pct > 0.2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_adverse_selection_metrics(self) -> Dict:
        """Get aggregate adverse selection metrics"""
        
        if not self.adverse_events:
            return {
                'total_events': 0,
                'high_severity': 0,
                'medium_severity': 0,
                'low_severity': 0,
                'avg_loss': 0.0
            }
        
        high_severity = sum(
            1 for e in self.adverse_events if e['severity'] == 'HIGH'
        )
        medium_severity = sum(
            1 for e in self.adverse_events if e['severity'] == 'MEDIUM'
        )
        low_severity = sum(
            1 for e in self.adverse_events if e['severity'] == 'LOW'
        )
        
        total_loss = sum(e['notional_loss'] for e in self.adverse_events)
        
        return {
            'total_events': len(self.adverse_events),
            'high_severity': high_severity,
            'medium_severity': medium_severity,
            'low_severity': low_severity,
            'avg_loss': total_loss / len(self.adverse_events) if self.adverse_events else 0.0,
            'total_loss': total_loss
        }
    
    def detect_information_leakage(self, symbol: str) -> float:
        """
        Detect if there's evidence of information leakage
        (consistent pattern of adverse selection)
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            Leakage score between 0 and 1
        """
        
        symbol_events = [
            e for e in self.adverse_events if e['symbol'] == symbol
        ]
        
        if len(symbol_events) < 5:
            return 0.0
        
        recent_events = symbol_events[-20:]
        
        # Calculate consistency of adverse moves
        adverse_ratio = sum(
            1 for e in recent_events if e['severity'] in ['HIGH', 'MEDIUM']
        ) / len(recent_events)
        
        # Higher ratio indicates potential information leakage
        leakage_score = adverse_ratio
        
        return leakage_score


# ============================================================================
# Main Execution Engine
# ============================================================================

class ExecutionEngine:
    """
    Main execution engine that coordinates all execution components
    """
    
    def __init__(self):
        """Initialize the Execution Engine"""
        
        self.fill_probability_estimator = FillProbabilityEstimator()
        self.smart_order_router = SmartOrderRouter()
        self.transaction_cost_analyzer = TransactionCostAnalyzer()
        self.adverse_selection_detector = AdverseSelectionDetector()
        
        self.executed_orders: Dict[str, ExecutionReport] = {}
        self.pending_orders: Dict[str, Dict] = {}
    
    def execute_order(
        self,
        order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
        order_books: Dict[ExecutionVenue, OrderBook] = None
    ) -> ExecutionReport:
        """
        Execute an order with full analysis and optimization
        
        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            price: Limit price (optional)
            order_type: Type of order
            order_books: Order books by venue
            
        Returns:
            ExecutionReport with execution details
        """
        
        # Route order to best venue
        venue, routing_info = self.smart_order_router.route_order(
            symbol, side, quantity, price, order_type, order_books
        )
        
        # Get order book for selected venue
        order_book = order_books.get(venue) if order_books else OrderBook(
            symbol=symbol,
            timestamp=datetime.utcnow()
        )
        
        # Estimate fill probability
        fill_prob = self.fill_probability_estimator.estimate_fill_probability(
            order_book, side, quantity, price, order_type
        )
        
        # Simulate execution
        executed_qty, executed_price = self._simulate_execution(
            order_book, side, quantity, price, order_type, fill_prob
        )
        
        # Analyze transaction costs
        cost_analysis = self.transaction_cost_analyzer.estimate_total_cost(
            symbol, executed_qty, executed_price, order_book.get_mid_price(), order_type
        )
        
        # Calculate slippage
        mid_price = order_book.get_mid_price()
        slippage = abs(executed_price - mid_price) if mid_price > 0 else 0
        
        # Create execution report
        report = ExecutionReport(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price or mid_price,
            executed_quantity=executed_qty,
            executed_price=executed_price,
            execution_time=datetime.utcnow(),
            venue=venue,
            transaction_cost=cost_analysis['total_cost'],
            slippage=slippage,
            fill_probability=fill_prob,
            status="FILLED" if executed_qty > 0 else "REJECTED"
        )
        
        # Record execution
        self.executed_orders[order_id] = report
        
        # Register for adverse selection monitoring
        if executed_qty > 0:
            self.adverse_selection_detector.register_execution(
                order_id, symbol, side, executed_price, executed_qty,
                report.execution_time
            )
        
        logger.info(
            f"Order {order_id} executed: {executed_qty} @ {executed_price:.4f}, "
            f"Cost: {cost_analysis['total_cost_bps']:.2f} bps"
        )
        
        return report
    
    def _simulate_execution(
        self,
        order_book: OrderBook,
        side: OrderSide,
        quantity: float,
        price: Optional[float],
        order_type: OrderType,
        fill_prob: float
    ) -> Tuple[float, float]:
        """Simulate order execution"""
        
        if order_type == OrderType.MARKET:
            if side == OrderSide.BUY:
                return quantity, order_book.asks[0][0] if order_book.asks else 0
            else:
                return quantity, order_book.bids[0][0] if order_book.bids else 0
        
        # Limit order execution
        if fill_prob > np.random.random():
            return quantity, price or order_book.get_mid_price()
        else:
            return 0, 0
    
    def update_market_price(
        self,
        order_id: str,
        price: float,
        timestamp: datetime
    ):
        """Update market price for adverse selection detection"""
        
        adverse_event = self.adverse_selection_detector.update_price(
            order_id, price, timestamp
        )
        
        if adverse_event:
            logger.warning(
                f"Adverse selection event: {adverse_event['symbol']}, "
                f"Loss: {adverse_event['notional_loss']:.2f}"
            )
    
    def get_execution_summary(self) -> Dict:
        """Get summary of execution performance"""
        
        if not self.executed_orders:
            return {}
        
        orders = list(self.executed_orders.values())
        
        avg_slippage = np.mean([o.slippage for o in orders])
        avg_fill_prob = np.mean([o.fill_probability for o in orders])
        total_costs = sum(o.transaction_cost for o in orders)
        
        return {
            'total_orders': len(orders),
            'avg_slippage': avg_slippage,
            'avg_fill_probability': avg_fill_prob,
            'total_transaction_costs': total_costs,
            'cost_statistics': self.transaction_cost_analyzer.get_cost_statistics(),
            'adverse_selection_metrics': (
                self.adverse_selection_detector.get_adverse_selection_metrics()
            )
        }


# ============================================================================
# Utility Functions
# ============================================================================

def calculate_vwap(prices: List[float], volumes: List[float]) -> float:
    """
    Calculate Volume Weighted Average Price
    
    Args:
        prices: List of prices
        volumes: List of volumes
        
    Returns:
        VWAP value
    """
    
    if not prices or not volumes or len(prices) != len(volumes):
        return 0.0
    
    total_volume = sum(volumes)
    if total_volume == 0:
        return 0.0
    
    weighted_price = sum(p * v for p, v in zip(prices, volumes))
    
    return weighted_price / total_volume


def calculate_twap(prices: List[float], time_intervals: List[float]) -> float:
    """
    Calculate Time Weighted Average Price
    
    Args:
        prices: List of prices
        time_intervals: Time intervals for each price
        
    Returns:
        TWAP value
    """
    
    if not prices or not time_intervals or len(prices) != len(time_intervals):
        return 0.0
    
    total_time = sum(time_intervals)
    if total_time == 0:
        return 0.0
    
    weighted_price = sum(p * t for p, t in zip(prices, time_intervals))
    
    return weighted_price / total_time


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'OrderSide',
    'OrderType',
    'ExecutionVenue',
    'OrderBook',
    'ExecutionReport',
    'MarketConditions',
    'FillProbabilityEstimator',
    'SmartOrderRouter',
    'TransactionCostAnalyzer',
    'AdverseSelectionDetector',
    'ExecutionEngine',
    'calculate_vwap',
    'calculate_twap'
]
