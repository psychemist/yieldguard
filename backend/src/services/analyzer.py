"""
YieldGuard Lite - Yield Analysis Engine
Lightweight model for trend detection, volatility analysis, and stance computation.
"""

from dataclasses import dataclass
from enum import Enum

from ..utils.config import config


class TrendDirection(Enum):
    RISING = "rising"
    STABLE = "stable"
    FALLING = "falling"
    UNKNOWN = "unknown"


class VolatilityBucket(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class Stance(Enum):
    FAVORABLE = "favorable"  # Good conditions, deploy capital
    NEUTRAL = "neutral"  # Average conditions, balanced approach
    CAUTION = "caution"  # Elevated risk, reduce exposure
    WAIT = "wait"  # Poor conditions, hold off


@dataclass
class TrendAnalysis:
    """Result of trend analysis on a time series."""

    direction: TrendDirection
    slope: float  # Raw slope (units per day)
    normalized_slope: float  # Slope as % of mean
    strength: float  # 0-1, how strong the trend is
    r_squared: float  # 0-1, how well data fits trend

    def to_dict(self) -> dict:
        return {
            "direction": self.direction.value,
            "slope": round(self.slope, 4),
            "normalized_slope": round(self.normalized_slope, 4),
            "strength": round(self.strength, 3),
            "r_squared": round(self.r_squared, 3),
        }


@dataclass
class VolatilityAnalysis:
    """Result of volatility analysis."""

    bucket: VolatilityBucket
    std_dev: float
    coefficient_of_variation: float  # std_dev / mean
    max_drawdown: float  # Largest peak-to-trough drop
    range_pct: float  # (max - min) / mean

    def to_dict(self) -> dict:
        return {
            "bucket": self.bucket.value,
            "std_dev": round(self.std_dev, 2),
            "cv": round(self.coefficient_of_variation, 4),
            "max_drawdown": round(self.max_drawdown, 2),
            "range_pct": round(self.range_pct, 4),
        }


@dataclass
class MarketStance:
    """Computed market stance with reasoning."""

    stance: Stance
    confidence: float
    reasoning: list[str]
    recommended_stable_pct: float

    def to_dict(self) -> dict:
        return {
            "stance": self.stance.value,
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning,
            "recommended_stable_pct": round(self.recommended_stable_pct, 1),
        }


class YieldAnalyzer:
    """
    Lightweight yield analysis engine.
    Uses statistical methods for:
    - Trend detection (linear regression)
    - Volatility bucketing (coefficient of variation)
    - Stance computation (rule-based scoring)
    """

    def __init__(self):
        self.cfg = config.analysis
        self.risk_cfg = config.risk

    def analyze_trend(self, values: list[float]) -> TrendAnalysis:
        """
        Detect yield trend using linear regression.
        Returns direction, slope, and fit quality.
        """
        if len(values) < 3:
            return TrendAnalysis(direction=TrendDirection.UNKNOWN, slope=0, normalized_slope=0, strength=0, r_squared=0)

        n = len(values)

        # Mean calculations
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        if y_mean == 0:
            return TrendAnalysis(direction=TrendDirection.UNKNOWN, slope=0, normalized_slope=0, strength=0, r_squared=0)

        # Linear regression: y = mx + b
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Normalized slope (% change per day)
        normalized_slope = slope / y_mean

        # R-squared (coefficient of determination)
        intercept = y_mean - slope * x_mean
        ss_res = sum((values[i] - (slope * i + intercept)) ** 2 for i in range(n))
        ss_tot = sum((v - y_mean) ** 2 for v in values)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        r_squared = max(0, r_squared)  # Clamp to valid range

        # Determine direction
        if normalized_slope > self.cfg.trend_rising_threshold:
            direction = TrendDirection.RISING
        elif normalized_slope < self.cfg.trend_falling_threshold:
            direction = TrendDirection.FALLING
        else:
            direction = TrendDirection.STABLE

        # Trend strength (combine slope magnitude with fit quality)
        strength = min(1.0, abs(normalized_slope) * 10) * r_squared

        return TrendAnalysis(
            direction=direction, slope=slope, normalized_slope=normalized_slope, strength=strength, r_squared=r_squared
        )

    def analyze_volatility(self, values: list[float]) -> VolatilityAnalysis:
        """
        Analyze yield volatility and bucket it.
        Uses coefficient of variation and max drawdown.
        """
        if len(values) < 2:
            return VolatilityAnalysis(
                bucket=VolatilityBucket.LOW, std_dev=0, coefficient_of_variation=0, max_drawdown=0, range_pct=0
            )

        n = len(values)
        mean = sum(values) / n

        if mean == 0:
            return VolatilityAnalysis(
                bucket=VolatilityBucket.LOW, std_dev=0, coefficient_of_variation=0, max_drawdown=0, range_pct=0
            )

        # Standard deviation
        variance = sum((v - mean) ** 2 for v in values) / n
        std_dev = variance**0.5

        # Coefficient of variation (normalized volatility)
        cv = std_dev / mean

        # Max drawdown
        peak = values[0]
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Range as percentage of mean
        range_pct = (max(values) - min(values)) / mean if mean > 0 else 0

        # Bucket classification
        if std_dev < self.risk_cfg.low_volatility_threshold:
            bucket = VolatilityBucket.LOW
        elif std_dev < self.risk_cfg.high_volatility_threshold:
            bucket = VolatilityBucket.MODERATE
        elif cv < 0.5:
            bucket = VolatilityBucket.HIGH
        else:
            bucket = VolatilityBucket.EXTREME

        return VolatilityAnalysis(
            bucket=bucket,
            std_dev=std_dev,
            coefficient_of_variation=cv,
            max_drawdown=max_dd * 100,  # As percentage
            range_pct=range_pct,
        )

    def compute_stance(
        self,
        trend: TrendAnalysis,
        volatility: VolatilityAnalysis,
        avg_apy: float,
        gas_cost_pct: float = 0,  # Gas cost as % of capital
    ) -> MarketStance:
        """
        Compute recommended market stance based on analysis.
        Returns stance, confidence, and allocation guidance.
        """
        score = 0.5  # Start neutral
        reasoning = []

        # Trend scoring (-0.2 to +0.2)
        if trend.direction == TrendDirection.RISING:
            score += 0.15 * trend.strength
            reasoning.append(f"Rising yields (+{trend.normalized_slope * 100:.1f}%/day)")
        elif trend.direction == TrendDirection.FALLING:
            score -= 0.15 * trend.strength
            reasoning.append(f"Falling yields ({trend.normalized_slope * 100:.1f}%/day)")
        else:
            reasoning.append("Stable yield environment")

        # Volatility scoring (-0.3 to +0.1)
        if volatility.bucket == VolatilityBucket.LOW:
            score += 0.1
            reasoning.append("Low volatility - stable conditions")
        elif volatility.bucket == VolatilityBucket.MODERATE:
            pass  # No adjustment
            reasoning.append("Moderate volatility")
        elif volatility.bucket == VolatilityBucket.HIGH:
            score -= 0.15
            reasoning.append(f"High volatility (σ={volatility.std_dev:.1f}%)")
        else:
            score -= 0.25
            reasoning.append("Extreme volatility - elevated risk")

        # APY attractiveness (+/- 0.15)
        if avg_apy > 10:
            score += 0.1
            reasoning.append(f"Attractive yields ({avg_apy:.1f}% avg)")
        elif avg_apy < 3:
            score -= 0.1
            reasoning.append(f"Low yields ({avg_apy:.1f}% avg)")

        # Gas cost impact (-0.1 to 0)
        if gas_cost_pct > 2:
            score -= 0.1
            reasoning.append(f"High gas costs ({gas_cost_pct:.1f}% of capital)")
        elif gas_cost_pct > 1:
            score -= 0.05
            reasoning.append(f"Moderate gas costs ({gas_cost_pct:.1f}%)")

        # Determine stance from score
        if score >= 0.65:
            stance = Stance.FAVORABLE
            stable_pct = 30
        elif score >= 0.45:
            stance = Stance.NEUTRAL
            stable_pct = 50
        elif score >= 0.3:
            stance = Stance.CAUTION
            stable_pct = 70
        else:
            stance = Stance.WAIT
            stable_pct = 90

        # Confidence based on data quality
        confidence = min(1.0, trend.r_squared * 0.4 + 0.6)
        if len(reasoning) > 3:
            confidence *= 0.95  # Slight penalty for complexity

        return MarketStance(
            stance=stance, confidence=confidence, reasoning=reasoning, recommended_stable_pct=stable_pct
        )

    def analyze_pool_risk(
        self, apy: float, tvl: float, il_risk: str, volatility: VolatilityAnalysis | None = None
    ) -> tuple[float, list[str]]:
        """
        Compute risk score (0-1) for a single pool.
        Returns (score, risk_factors).
        """
        risk_score = 0.3  # Base risk
        factors = []

        # TVL-based risk
        if tvl < 10_000_000:  # < $10M
            risk_score += self.risk_cfg.low_tvl_risk_penalty
            factors.append("Low TVL (< $10M)")
        elif tvl < 100_000_000:  # < $100M
            risk_score += self.risk_cfg.medium_tvl_risk_penalty
            factors.append("Medium TVL")

        # APY-based risk (unusually high APY = suspicious)
        if apy > config.filters.max_apy_percent / 2:  # e.g. > 50%
            risk_score += 0.3
            factors.append(f"Very high APY ({apy:.0f}%) - verify legitimacy")
        elif apy > 20:
            risk_score += 0.1
            factors.append("Above-average APY")

        # IL risk
        if il_risk == "high":
            risk_score += 0.15
            factors.append("High impermanent loss risk")
        elif il_risk == "medium":
            risk_score += 0.05
            factors.append("Moderate IL risk")

        # Volatility-based risk
        if volatility:
            if volatility.bucket == VolatilityBucket.HIGH:
                risk_score += 0.1
                factors.append("High yield volatility")
            elif volatility.bucket == VolatilityBucket.EXTREME:
                risk_score += 0.2
                factors.append("Extreme yield volatility")

        return min(1.0, risk_score), factors

    def _select_top_pools(self, pools: list[dict], count: int, sort_key: str = "apy") -> list[dict]:
        """Select top pools based on criteria."""
        # Sort by key (descending)
        sorted_pools = sorted(pools, key=lambda p: p.get(sort_key, 0), reverse=True)
        return sorted_pools[:count]

    def suggest_allocation(self, risk_profile: str, stance: MarketStance, pools: list[dict]) -> list[dict]:
        """
        Suggest allocation based on risk profile and market stance.
        Returns list of {asset, percentage, reasoning}.
        """
        constraints = self.risk_cfg.risk_profiles.get(risk_profile, self.risk_cfg.risk_profiles["medium"])
        min_diversification = constraints.get("min_diversification", 2)
        max_single = constraints["max_single_allocation"]

        # Adjust stable preference based on stance
        base_stable = constraints["stable_preference"] * 100
        adjusted_stable = (base_stable + stance.recommended_stable_pct) / 2

        # Ensure we have enough percentage for at least one asset if very low
        if adjusted_stable < 5 and adjusted_stable > 0:
            adjusted_stable = 5
        if adjusted_stable > 95:
            adjusted_stable = 95

        # Separate stable and volatile pools
        stable_pools = [p for p in pools if p.get("il_risk") == "low" or "stable" in p.get("symbol", "").lower()]
        volatile_pools = [p for p in pools if p not in stable_pools]

        allocations = []
        remaining_portfolio = 100.0

        # Calculate target counts
        # We try to split the diversification requirement between stable and volatile
        # proportionally to their allocation size, but at least 1 each if available.
        target_stable_count = max(1, round(min_diversification * (adjusted_stable / 100)))
        target_volatile_count = max(1, min_diversification - target_stable_count)

        # --- Stable Allocation ---
        if stable_pools and adjusted_stable > 0:
            top_stables = self._select_top_pools(stable_pools, target_stable_count, "apy")

            # Divide allocation among selected pools
            per_pool_alloc = adjusted_stable / len(top_stables)

            # Cap at max_single
            if per_pool_alloc > max_single:
                # If we exceed max_single per pool, we might need more pools.
                # For this lite version, we just cap it and leave the rest unallocated
                # (or it flows to volatile, but we want to respect the stable preference).
                # Better strategy: Add more pools if available.
                needed_pools = int(adjusted_stable // max_single) + 1
                if needed_pools > len(top_stables):
                    top_stables = self._select_top_pools(stable_pools, needed_pools, "apy")
                    per_pool_alloc = adjusted_stable / len(top_stables)  # Recalculate

                per_pool_alloc = min(per_pool_alloc, max_single)

            for pool in top_stables:
                allocations.append(
                    {
                        "asset": pool.get("symbol", "STABLE"),
                        "pool_id": pool.get("pool_id"),
                        "percentage": round(per_pool_alloc, 2),
                        "expected_yield": pool.get("apy", 0),
                        "risk_score": 0.2,  # Simplified
                        "reasoning": f"Stable allocation ({stance.stance.value} stance)",
                    }
                )
                remaining_portfolio -= per_pool_alloc

        # --- Volatile Allocation ---
        if volatile_pools and remaining_portfolio > 1.0:  # Ignore tiny dust
            top_volatile = self._select_top_pools(
                volatile_pools, target_volatile_count, "tvl"
            )  # Prefer TVL for volatile

            per_pool_alloc = remaining_portfolio / len(top_volatile)

            # Cap at max_single
            if per_pool_alloc > max_single:
                needed_pools = int(remaining_portfolio // max_single) + 1
                if needed_pools > len(top_volatile):
                    top_volatile = self._select_top_pools(volatile_pools, needed_pools, "tvl")
                    per_pool_alloc = remaining_portfolio / len(top_volatile)
                per_pool_alloc = min(per_pool_alloc, max_single)

            for pool in top_volatile:
                allocations.append(
                    {
                        "asset": pool.get("symbol", "VOLATILE"),
                        "pool_id": pool.get("pool_id"),
                        "percentage": round(per_pool_alloc, 2),
                        "expected_yield": pool.get("apy", 0),
                        "risk_score": 0.5,  # Simplified
                        "reasoning": "Higher yield opportunity",
                    }
                )

        return allocations


# Global instance
yield_analyzer = YieldAnalyzer()
