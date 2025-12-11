"""
YieldGuard Lite Configuration
"""

import os

from pydantic import BaseModel


class APIConfig(BaseModel):
    """External API endpoints configuration"""

    defillama_yields: str = f"{os.getenv('DEFILLAMA_YIELD_BASE')}/pools"
    defillama_chart: str = f"{os.getenv('DEFILLAMA_YIELD_BASE')}/chart"
    defillama_protocol: str = f"{os.getenv('DEFILLAMA_API_BASE')}/protocol"
    coingecko_prices: str = f"{os.getenv('COINGECKO_API_BASE')}/simple/price"
    coingecko_market: str = f"{os.getenv('COINGECKO_API_BASE')}/coins/markets"
    etherscan_gas: str = f"{os.getenv('ETHERSCAN_API_URL')}"
    owlracle_gas: str = f"{os.getenv('OWLRACLE_API_BASE')}/gas"


class CacheConfig(BaseModel):
    """Caching configuration"""

    yield_ttl_seconds: int = 300
    gas_ttl_seconds: int = 60  # 1 minute
    market_ttl_seconds: int = 300  # 5 minutes
    historical_ttl_seconds: int = 3600  # 1 hour


class FilterConfig(BaseModel):
    """Data filtering configuration"""

    min_tvl_usd: float = 1_000_000  # Minimum $1M TVL for pools
    max_apy_percent: float = 100  # Filter out unrealistic APYs
    min_apy_percent: float = 0.1  # Minimum yield to consider
    supported_chains: list[str] = ["Ethereum"]
    max_pools_to_fetch: int = 50  # Limit API response processing


class RiskConfig(BaseModel):
    """Risk assessment configuration"""

    # Volatility thresholds (standard deviation of yields)
    low_volatility_threshold: float = 5.0
    high_volatility_threshold: float = 15.0

    # TVL-based risk adjustment
    low_tvl_risk_penalty: float = 0.2  # Added risk for < $10M TVL
    medium_tvl_risk_penalty: float = 0.05  # Added risk for < $100M TVL

    # Risk profile allocation constraints
    risk_profiles: dict[str, dict] = {
        "low": {
            "max_single_allocation": 50.0,
            "min_diversification": 3,
            "max_risk_score": 0.4,
            "stable_preference": 0.7,
        },
        "medium": {
            "max_single_allocation": 60.0,
            "min_diversification": 2,
            "max_risk_score": 0.6,
            "stable_preference": 0.5,
        },
        "high": {
            "max_single_allocation": 80.0,
            "min_diversification": 1,
            "max_risk_score": 0.9,
            "stable_preference": 0.2,
        },
    }


class ModelConfig(BaseModel):
    """AI model configuration"""

    # Groq models in order of preference
    available_models: list[str] = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]

    # Model parameters
    temperature: float = 0.1
    max_tokens: int = 1500
    timeout_seconds: int = 30
    max_retries: int = 3
    rate_limit_cooldown_seconds: int = 3600  # 1 hour


class AnalysisConfig(BaseModel):
    """Yield analysis configuration"""

    # Trend detection
    trend_window_days: int = 7
    trend_rising_threshold: float = 0.02  # 2% increase = rising
    trend_falling_threshold: float = -0.02  # 2% decrease = falling

    # Confidence scoring
    min_data_points_for_confidence: int = 7
    confidence_boost_stable: float = 0.1
    confidence_penalty_volatile: float = 0.15

    # Recommendation thresholds
    caution_volatility_threshold: float = 0.15
    favorable_trend_threshold: float = 0.05


class GasConfig(BaseModel):
    """Gas estimation configuration"""

    # Standard gas limits for DeFi operations
    swap_gas_limit: int = 150_000
    deposit_gas_limit: int = 200_000
    withdraw_gas_limit: int = 180_000

    # Transaction count estimates per strategy
    estimated_transactions: int = 3


class Config:
    """Main configuration class"""

    api: APIConfig = APIConfig()
    cache: CacheConfig = CacheConfig()
    filters: FilterConfig = FilterConfig()
    risk: RiskConfig = RiskConfig()
    model: ModelConfig = ModelConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    gas: GasConfig = GasConfig()

    # Environment variables
    @property
    def groq_api_key(self) -> str:
        return os.getenv("GROQ_API_KEY", "")

    @property
    def etherscan_api_key(self) -> str:
        return os.getenv("ETHERSCAN_API_KEY", "")

    @property
    def owlracle_api_key(self) -> str:
        return os.getenv("OWLRACLE_API_KEY", "")


# Global config instance
config = Config()
