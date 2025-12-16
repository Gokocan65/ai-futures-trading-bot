"""
Configuration module for AI Futures Trading Bot.

This module defines all configuration dataclasses for exchange, model, risk,
adaptive, backtest, monitoring, and global settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ExchangeType(Enum):
    """Supported exchange types."""
    BINANCE = "binance"
    BYBIT = "bybit"
    KUCOIN = "kucoin"


class OrderType(Enum):
    """Supported order types."""
    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"


@dataclass
class ExchangeConfig:
    """Configuration for exchange connection and trading parameters."""
    
    exchange_type: ExchangeType = ExchangeType.BINANCE
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    sandbox: bool = False
    
    # Connection settings
    timeout: int = 30
    max_retries: int = 3
    rate_limit: int = 1000
    
    # Trading parameters
    default_order_type: OrderType = OrderType.LIMIT
    default_leverage: float = 1.0
    max_leverage: float = 10.0
    min_order_size: float = 10.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    
    # Trading pairs
    trading_pairs: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    
    # Advanced settings
    enable_margin: bool = False
    enable_futures: bool = True
    position_mode: str = "one-way"  # one-way or hedge


@dataclass
class ModelConfig:
    """Configuration for AI/ML model parameters."""
    
    # Model type and path
    model_type: str = "lstm"
    model_path: Optional[str] = None
    pretrained: bool = True
    
    # Input/Output configuration
    input_sequence_length: int = 100
    output_horizon: int = 1
    feature_count: int = 10
    
    # LSTM specific
    lstm_units: int = 128
    lstm_layers: int = 2
    dropout_rate: float = 0.2
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "mse"
    
    # Validation and testing
    train_test_split: float = 0.8
    validation_split: float = 0.1
    shuffle: bool = True
    
    # Model checkpoint settings
    save_best_only: bool = True
    early_stopping_patience: int = 10
    min_delta: float = 0.001
    
    # Prediction confidence
    confidence_threshold: float = 0.65
    use_ensemble: bool = False
    ensemble_models: List[str] = field(default_factory=list)


@dataclass
class RiskConfig:
    """Configuration for risk management parameters."""
    
    # Position sizing
    max_position_size: float = 10000.0
    min_position_size: float = 100.0
    position_sizing_method: str = "kelly"  # kelly, fixed, dynamic
    kelly_fraction: float = 0.25
    
    # Stop loss and take profit
    use_stop_loss: bool = True
    stop_loss_percent: float = 2.0
    trailing_stop: bool = True
    trailing_stop_percent: float = 1.0
    
    use_take_profit: bool = True
    take_profit_percent: float = 5.0
    
    # Drawdown limits
    max_daily_drawdown_percent: float = 5.0
    max_total_drawdown_percent: float = 15.0
    stop_trading_on_drawdown: bool = True
    
    # Loss limits
    max_consecutive_losses: int = 3
    max_loss_per_trade_percent: float = 2.0
    
    # Portfolio risk
    max_portfolio_risk_percent: float = 2.0
    max_correlation_threshold: float = 0.8
    
    # Leverage restrictions
    max_leverage_per_pair: float = 5.0
    max_total_leverage: float = 20.0
    
    # Emergency settings
    emergency_stop_loss: float = 10.0
    panic_sell_enabled: bool = False


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive trading parameters."""
    
    # Adaptive learning
    enable_adaptive: bool = True
    learning_rate_adaptive: bool = True
    
    # Market condition detection
    detect_market_regime: bool = True
    regime_window: int = 50
    regime_threshold: float = 0.3
    
    # Dynamic adjustment
    adjust_position_size: bool = True
    adjust_stop_loss: bool = True
    adjust_leverage: bool = True
    
    # Performance monitoring
    performance_window: int = 100
    win_rate_threshold: float = 0.50
    
    # Volatility adjustment
    use_volatility_scaling: bool = True
    volatility_lookback: int = 30
    volatility_threshold: float = 2.0
    
    # Slippage adjustment
    estimate_slippage: bool = True
    slippage_percent: float = 0.1
    
    # Model retraining
    retrain_frequency: int = 1000  # trades
    retrain_threshold: float = 0.95  # performance ratio
    auto_retrain: bool = True
    
    # Risk parameter optimization
    optimize_kelly_fraction: bool = True
    optimization_window: int = 500  # trades


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    
    # Data settings
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"
    data_frequency: str = "1h"  # 1m, 5m, 1h, 1d
    
    # Initial capital
    initial_capital: float = 10000.0
    
    # Slippage and commissions
    slippage_percent: float = 0.05
    commission_percent: float = 0.1
    
    # Simulation settings
    use_realistic_fills: bool = True
    max_slippage_percent: float = 0.5
    
    # Data validation
    validate_data: bool = True
    fill_missing_data: bool = True
    
    # Performance metrics
    calculate_sharpe: bool = True
    risk_free_rate: float = 0.02
    
    # Output settings
    save_results: bool = True
    save_trades: bool = True
    results_path: str = "./backtest_results/"
    
    # Advanced options
    walk_forward_window: Optional[int] = None
    optimization_metric: str = "sharpe_ratio"
    parallel_processing: bool = True
    num_workers: int = 4


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging parameters."""
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "trading_bot.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_log_size: int = 10485760  # 10MB
    backup_count: int = 5
    
    # Metrics tracking
    track_metrics: bool = True
    metrics_window: int = 100
    metrics_frequency: int = 60  # seconds
    
    # Health checks
    enable_health_checks: bool = True
    health_check_interval: int = 300  # seconds
    
    # Alerts
    enable_alerts: bool = True
    alert_methods: List[str] = field(default_factory=lambda: ["console", "email"])
    
    # Email notifications
    smtp_server: str = ""
    smtp_port: int = 587
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    
    # Database logging
    use_database: bool = False
    database_url: str = "sqlite:///trading_bot.db"
    
    # Performance monitoring
    profile_code: bool = False
    memory_monitoring: bool = True
    memory_threshold_percent: float = 80.0
    
    # Webhook notifications
    enable_webhooks: bool = False
    webhook_urls: List[str] = field(default_factory=list)
    
    # Dashboard settings
    enable_dashboard: bool = True
    dashboard_port: int = 8080
    dashboard_refresh_interval: int = 5  # seconds


@dataclass
class GlobalConfig:
    """Global configuration combining all subsystems."""
    
    # System settings
    bot_name: str = "AI Futures Trading Bot"
    version: str = "1.0.0"
    environment: str = "development"  # development, staging, production
    debug_mode: bool = False
    
    # Component configurations
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Trading control
    trading_enabled: bool = True
    paper_trading: bool = True
    live_trading: bool = False
    
    # Execution settings
    execution_mode: str = "async"  # sync, async
    order_execution_timeout: int = 30
    order_retry_attempts: int = 3
    
    # Data settings
    data_source: str = "exchange"  # exchange, csv, database
    data_cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Performance settings
    use_caching: bool = True
    use_threading: bool = True
    max_threads: int = 4
    
    # Schedule settings
    market_open_time: str = "00:00"
    market_close_time: str = "23:59"
    restart_on_failure: bool = True
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        if self.model.confidence_threshold < 0.0 or self.model.confidence_threshold > 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        if self.risk.stop_loss_percent < 0.0:
            raise ValueError("stop_loss_percent must be positive")
        
        if self.backtest.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        
        if self.exchange.max_leverage < 1.0:
            raise ValueError("max_leverage must be >= 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GlobalConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


# Default configuration instance
DEFAULT_CONFIG = GlobalConfig()
