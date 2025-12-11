import pytest

from src.services.analyzer import (
    MarketStance,
    Stance,
    TrendDirection,
    VolatilityAnalysis,
    VolatilityBucket,
    YieldAnalyzer,
)

# Mock Pools
MOCK_POOLS = [
    # Stable Pools
    {"symbol": "USDC-USDT", "apy": 5.0, "tvl": 50_000_000, "il_risk": "low", "pool_id": "p1", "stable_coin": True},
    {"symbol": "DAI-USDC", "apy": 4.5, "tvl": 40_000_000, "il_risk": "low", "pool_id": "p2", "stable_coin": True},
    {"symbol": "FRAX-USDC", "apy": 4.0, "tvl": 30_000_000, "il_risk": "low", "pool_id": "p3", "stable_coin": True},
    {"symbol": "LUSD-DAI", "apy": 3.5, "tvl": 20_000_000, "il_risk": "low", "pool_id": "p4", "stable_coin": True},
    # Volatile Pools
    {"symbol": "ETH-USDC", "apy": 15.0, "tvl": 100_000_000, "il_risk": "high", "pool_id": "p5", "stable_coin": False},
    {"symbol": "WBTC-ETH", "apy": 12.0, "tvl": 80_000_000, "il_risk": "medium", "pool_id": "p6", "stable_coin": False},
    {"symbol": "SOL-USDC", "apy": 25.0, "tvl": 10_000_000, "il_risk": "high", "pool_id": "p7", "stable_coin": False},
    {"symbol": "ARB-ETH", "apy": 20.0, "tvl": 5_000_000, "il_risk": "high", "pool_id": "p8", "stable_coin": False},
]


@pytest.fixture
def analyzer():
    return YieldAnalyzer()


@pytest.fixture
def neutral_stance():
    return MarketStance(
        stance=Stance.NEUTRAL, confidence=0.8, reasoning=["Neutral market"], recommended_stable_pct=50.0
    )


def test_suggest_allocation_diversification_medium_risk(analyzer, neutral_stance):
    """Test that medium risk profile produces diversified allocations."""
    # Medium risk usually has min_diversification=2
    allocations = analyzer.suggest_allocation("medium", neutral_stance, MOCK_POOLS)

    assert len(allocations) >= 2

    # Check if we have both stable and volatile
    stables = [a for a in allocations if a["asset"] in ["USDC-USDT", "DAI-USDC", "FRAX-USDC"]]
    volatiles = [a for a in allocations if a["asset"] in ["ETH-USDC", "WBTC-ETH", "SOL-USDC"]]

    assert len(stables) >= 1
    assert len(volatiles) >= 1

    total_pct = sum(a["percentage"] for a in allocations)
    assert 99.0 <= total_pct <= 100.1  # Floating point tolerance


def test_suggest_allocation_diversification_low_risk(analyzer, neutral_stance):
    """Test that low risk profile produces more stable allocations."""
    # Low risk has higher stable preference
    allocations = analyzer.suggest_allocation("low", neutral_stance, MOCK_POOLS)

    stables = [a for a in allocations if "stable" in a["reasoning"].lower()]
    volatiles = [a for a in allocations if "stable" not in a["reasoning"].lower()]

    stable_pct = sum(a["percentage"] for a in stables)
    volatile_pct = sum(a["percentage"] for a in volatiles)

    assert stable_pct > volatile_pct


def test_analyze_pool_risk_thresholds(analyzer):
    """Test that risk analysis uses correct thresholds."""

    # Very High APY
    score, factors = analyzer.analyze_pool_risk(apy=60.0, tvl=200_000_000, il_risk="low")
    assert "Very high APY" in factors[0]
    assert score >= 0.3

    # Low TVL
    score, factors = analyzer.analyze_pool_risk(apy=5.0, tvl=1_000_000, il_risk="low")
    assert "Low TVL" in str(factors)

    # High Volatility
    vol = VolatilityAnalysis(
        bucket=VolatilityBucket.EXTREME, std_dev=20, coefficient_of_variation=1, max_drawdown=10, range_pct=20
    )
    score, factors = analyzer.analyze_pool_risk(apy=5.0, tvl=200_000_000, il_risk="low", volatility=vol)
    assert "Extreme yield volatility" in str(factors)


def test_allocations_respect_max_single(analyzer, neutral_stance):
    """Test that no single allocation exceeds the limit."""
    allocations = analyzer.suggest_allocation("high", neutral_stance, MOCK_POOLS)

    # High risk might have max_single of 80% (from config default in code)
    # Check against whatever the analyzer used.
    # We can infer it from the result or just check for sanity.

    for alloc in allocations:
        assert alloc["percentage"] <= 80.0  # From config for "high"


def test_trend_analysis_basic(analyzer):
    """Test basic linear trend analysis."""
    # Perfect rising trend: 1, 2, 3, 4, 5
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    trend = analyzer.analyze_trend(values)

    assert trend.slope > 0
    assert (
        trend.direction == TrendDirection.RISING or trend.direction == TrendDirection.STABLE
    )  # Depends on threshold vs % change
    # Mean is 3. Slope is 1. Norm slope = 1/3 = 0.33.
    # Threshold is 0.02. So it should be RISING.
    assert trend.direction == TrendDirection.RISING
    assert trend.r_squared > 0.9
