"""
YieldGuard Lite - Unified Data Service
Single entry point for all data fetching with caching, rate limiting, and normalization.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import httpx

from ..utils.config import config


class DataSource(Enum):
    """Data source identifiers"""
    DEFILLAMA = "defillama"
    COINGECKO = "coingecko"
    ETHERSCAN = "etherscan"
    OWLRACLE = "owlracle"

@dataclass
class CacheEntry:
    """Cache entry with TTL tracking"""
    data: Any
    expires_at: float
    source: DataSource

    @property
    def is_valid(self) -> bool:
        return datetime.now().timestamp() < self.expires_at

@dataclass
class TimeSeries:
    """Normalized time-series data structure"""
    timestamps: list[str] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    source: str = ""
    asset: str = ""
    metric: str = ""  # "apy", "tvl", "price", "gas"

    def to_dict(self) -> dict:
        return {
            "timestamps": self.timestamps,
            "values": self.values,
            "source": self.source,
            "asset": self.asset,
            "metric": self.metric
        }

@dataclass
class YieldPool:
    """Normalized yield pool data"""
    protocol: str
    symbol: str
    chain: str
    apy: float
    tvl_usd: float
    pool_id: str
    il_risk: str = "unknown"
    stable_coin: bool = False
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "symbol": self.symbol,
            "chain": self.chain,
            "apy": self.apy,
            "tvl_usd": self.tvl_usd,
            "pool_id": self.pool_id,
            "il_risk": self.il_risk,
            "stable_coin": self.stable_coin,
            "timestamp": self.timestamp
        }


@dataclass
class GasData:
    """Normalized gas data"""
    slow_gwei: float
    standard_gwei: float
    fast_gwei: float
    eth_price_usd: float
    timestamp: str

    def estimate_cost_usd(self, gas_limit: int, speed: str = "standard") -> float:
        """Calculate transaction cost in USD"""
        gwei = getattr(self, f"{speed}_gwei", self.standard_gwei)
        return (gwei * gas_limit * self.eth_price_usd) / 1e9

    def to_dict(self) -> dict:
        return {
            "slow_gwei": self.slow_gwei,
            "standard_gwei": self.standard_gwei,
            "fast_gwei": self.fast_gwei,
            "eth_price_usd": self.eth_price_usd,
            "timestamp": self.timestamp
        }


class RateLimiter:
    """Simple rate limiter with exponential backoff"""

    def __init__(self):
        self._last_request: dict[str, float] = {}
        self._backoff: dict[str, int] = {}
        self._min_interval = 0.5  # 500ms between requests

    async def wait(self, source: str):
        """Wait if needed to respect rate limits"""
        now = datetime.now().timestamp()
        last = self._last_request.get(source, 0)
        backoff = self._backoff.get(source, 0)

        wait_time = max(0, (last + self._min_interval + backoff) - now)
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        self._last_request[source] = datetime.now().timestamp()

    def record_success(self, source: str):
        """Reset backoff on success"""
        self._backoff[source] = 0

    def record_failure(self, source: str):
        """Increase backoff on failure"""
        current = self._backoff.get(source, 0)
        self._backoff[source] = min(current + 2, 30)  # Max 30s backoff


class DataService:
    """
    Unified data service for all YieldGuard data needs.
    Features:
    - TTL-based caching
    - Rate limiting with backoff
    - Normalized data structures
    - Multiple source fallbacks
    """

    def __init__(self):
        self._cache: dict[str, CacheEntry] = {}
        self._rate_limiter = RateLimiter()
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-load HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"Accept": "application/json"}
            )
        return self._client

    def _get_cached(self, key: str) -> Any | None:
        """Get cached data if valid"""
        entry = self._cache.get(key)
        if entry and entry.is_valid:
            return entry.data
        return None

    def _set_cached(self, key: str, data: Any, ttl_seconds: int, source: DataSource):
        """Cache data with TTL"""
        self._cache[key] = CacheEntry(
            data=data,
            expires_at=datetime.now().timestamp() + ttl_seconds,
            source=source
        )

    async def _request(self, url: str, source: DataSource, raise_on_403: bool = False) -> dict | None:
        """Make rate-limited HTTP request with error handling"""
        await self._rate_limiter.wait(source.value)

        try:
            response = await self.client.get(url)
            if response.status_code == 200:
                self._rate_limiter.record_success(source.value)
                return response.json()
            elif response.status_code == 429:
                print(f"[DataService] Rate limited by {source.value}")
                self._rate_limiter.record_failure(source.value)
            elif response.status_code == 403:
                print(f"[DataService] {source.value} returned 403 Forbidden")
                self._rate_limiter.record_failure(source.value)
                if raise_on_403:
                    raise PermissionError(f"{source.value} returned 403 Forbidden")
            else:
                print(f"[DataService] {source.value} returned {response.status_code}")
        except PermissionError:
            raise  # Re-raise 403 errors when flagged
        except Exception as e:
            print(f"[DataService] Request error for {source.value}: {e}")
            self._rate_limiter.record_failure(source.value)

    # ==================== YIELD DATA ====================

    async def get_yield_pools(self) -> list[YieldPool]:
        """
        Fetch current yield pools from DeFiLlama.
        Returns normalized YieldPool objects.
        """
        cache_key = "yield_pools"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        data = await self._request(config.api.defillama_yields, DataSource.DEFILLAMA)
        if not data:
            return []

        pools = []
        now = datetime.now().isoformat()

        for raw_pool in data.get("data", []):
            chain = raw_pool.get("chain", "")
            apy = raw_pool.get("apy", 0) or 0
            tvl = raw_pool.get("tvlUsd", 0) or 0

            # Apply filters from config
            if chain not in config.filters.supported_chains:
                continue
            if not (config.filters.min_apy_percent < apy < config.filters.max_apy_percent):
                continue
            if tvl < config.filters.min_tvl_usd:
                continue

            # Determine IL risk based on pool composition
            symbol = raw_pool.get("symbol", "")
            exposure = raw_pool.get("exposure", "")
            il_risk = self._assess_il_risk(symbol, exposure)

            # Determine if stable pool
            symbol_upper = symbol.upper()
            stables = ["USDC", "USDT", "DAI", "FRAX", "LUSD", "GUSD", "USDS", "PYUSD", "USDE"]
            stable_count = sum(1 for s in stables if s in symbol_upper)
            is_stable = stable_count >= 2 or (stable_count == 1 and "-" not in symbol)

            pools.append(YieldPool(
                protocol=raw_pool.get("project", "unknown"),
                symbol=symbol,
                chain=chain,
                apy=round(apy, 2),
                tvl_usd=round(tvl, 0),
                pool_id=raw_pool.get("pool", ""),
                il_risk=il_risk,
                stable_coin=is_stable,
                timestamp=now
            ))

        # Sort by TVL (most liquid first) and limit
        pools.sort(key=lambda p: p.tvl_usd, reverse=True)
        pools = pools[:config.filters.max_pools_to_fetch]

        self._set_cached(cache_key, pools, config.cache.yield_ttl_seconds, DataSource.DEFILLAMA)
        print(f"[DataService] Fetched {len(pools)} yield pools")
        return pools

    def _assess_il_risk(self, symbol: str, exposure: str) -> str:
        """Assess impermanent loss risk based on pool composition"""
        symbol_upper = symbol.upper()

        # Stablecoin pairs have minimal IL risk
        stables = ["USDC", "USDT", "DAI", "FRAX", "LUSD", "GUSD"]
        stable_count = sum(1 for s in stables if s in symbol_upper)

        if stable_count >= 2:
            return "low"
        elif stable_count == 1:
            return "medium"
        elif "ETH" in symbol_upper and "WETH" in symbol_upper:
            return "low"
        else:
            return "high"

    async def get_yield_history(self, pool_id: str, days: int = 30) -> TimeSeries:
        """
        Fetch historical yield data for a specific pool.
        Returns normalized TimeSeries.
        """
        cache_key = f"yield_history_{pool_id}_{days}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        url = f"{config.api.defillama_chart}/{pool_id}"
        data = await self._request(url, DataSource.DEFILLAMA)

        if not data:
            return TimeSeries(source="defillama", asset=pool_id, metric="apy")

        # Handle both list and dict response formats
        points = data if isinstance(data, list) else data.get("data", [])

        # Get last N days (use UTC for comparison)
        cutoff = datetime.now(UTC) - timedelta(days=days)

        timestamps = []
        values = []

        for point in points:
            if isinstance(point, dict):
                ts = point.get("timestamp", point.get("date", ""))
                val = point.get("apy", point.get("value", 0))
            else:
                continue

            # Parse timestamp
            try:
                if isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts, tz=UTC)
                else:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))

                if dt >= cutoff:
                    timestamps.append(dt.strftime("%Y-%m-%d"))
                    values.append(round(float(val), 2) if val else 0)
            except Exception as e:
                print(f"[DataService] Failed to parse timestamp {ts}: {e}")
                continue

        result = TimeSeries(
            timestamps=timestamps,
            values=values,
            source="defillama",
            asset=pool_id,
            metric="apy"
        )

        self._set_cached(cache_key, result, config.cache.historical_ttl_seconds, DataSource.DEFILLAMA)
        return result

    # ==================== PRICE DATA ====================

    async def get_eth_price(self) -> float:
        """Get current ETH price in USD from CoinGecko"""
        cache_key = "eth_price"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        url = f"{config.api.coingecko_prices}?ids=ethereum&vs_currencies=usd"
        data = await self._request(url, DataSource.COINGECKO)

        if data:
            price = data.get("ethereum", {}).get("usd", 0)
            if price:
                self._set_cached(cache_key, price, config.cache.market_ttl_seconds, DataSource.COINGECKO)
                return price

        return 0

    async def get_token_prices(self, tokens: list[str]) -> dict[str, float]:
        """
        Get current prices for multiple tokens.
        Tokens should be CoinGecko IDs (e.g., "ethereum", "bitcoin").
        """
        if not tokens:
            return {}

        cache_key = f"token_prices_{'_'.join(sorted(tokens))}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        ids = ",".join(tokens)
        url = f"{config.api.coingecko_prices}?ids={ids}&vs_currencies=usd"
        data = await self._request(url, DataSource.COINGECKO)

        if not data:
            return {}

        prices = {token: data.get(token, {}).get("usd", 0) for token in tokens}
        self._set_cached(cache_key, prices, config.cache.market_ttl_seconds, DataSource.COINGECKO)
        return prices

    # ==================== GAS DATA ====================

    async def get_gas_data(self) -> GasData | None:
        """
        Fetch current gas prices with ETH price for USD estimation.
        Tries Etherscan first, then Owlracle as fallback.
        """
        cache_key = "gas_data"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Get ETH price first (needed for USD calculations)
        eth_price = await self.get_eth_price()

        # Try Etherscan first
        gas_data = await self._fetch_gas_etherscan(eth_price)

        if gas_data:
            self._set_cached(cache_key, gas_data, config.cache.gas_ttl_seconds, DataSource.ETHERSCAN)
            print(f"[DataService] Gas (Etherscan): {gas_data.standard_gwei} gwei, ETH: ${eth_price}")
            return gas_data

        # Fallback to Owlracle
        try:
            gas_data = await self._fetch_gas_owlracle(eth_price)
        except PermissionError:
            print("[DataService] Owlracle 403 - no fallback available")
            gas_data = None

        if gas_data:
            self._set_cached(cache_key, gas_data, config.cache.gas_ttl_seconds, DataSource.OWLRACLE)
            print(f"[DataService] Gas (Owlracle): {gas_data.standard_gwei} gwei, ETH: ${eth_price}")

        return gas_data

    async def _fetch_gas_owlracle(self, eth_price: float) -> GasData | None:
        """Fetch gas from Owlracle. Raises PermissionError on 403 for fallback."""
        url = config.api.owlracle_gas
        if config.owlracle_api_key:
            url += f"?apikey={config.owlracle_api_key}"

        # Raise on 403 so we can fall back to Etherscan
        data = await self._request(url, DataSource.OWLRACLE, raise_on_403=True)

        if not data:
            return None

        speeds = data.get("speeds", [])
        if len(speeds) < 3:
            return None

        return GasData(
            slow_gwei=float(speeds[0].get("gasPrice", 20)),
            standard_gwei=float(speeds[1].get("gasPrice", 25)),
            fast_gwei=float(speeds[2].get("gasPrice", 30)),
            eth_price_usd=eth_price,
            timestamp=datetime.now().isoformat()
        )

    async def _fetch_gas_etherscan(self, eth_price: float) -> GasData | None:
        """Fetch gas from Etherscan"""
        url = f"{config.api.etherscan_gas}?module=gastracker&action=gasoracle&chainid=1"
        if config.etherscan_api_key:
            url += f"&apikey={config.etherscan_api_key}"

        data = await self._request(url, DataSource.ETHERSCAN)

        if not data or data.get("status") != "1":
            return None

        result = data.get("result", {})

        return GasData(
            slow_gwei=float(result.get("SafeGasPrice", 20)),
            standard_gwei=float(result.get("ProposeGasPrice", 25)),
            fast_gwei=float(result.get("FastGasPrice", 30)),
            eth_price_usd=eth_price,
            timestamp=datetime.now().isoformat()
        )

    # ==================== ANALYSIS HELPERS ====================

    async def get_market_snapshot(self) -> dict:
        """
        Get a complete market snapshot for analysis.
        Combines yield, gas, and price data.
        """
        # Fetch all data in parallel
        pools_task = self.get_yield_pools()
        gas_task = self.get_gas_data()
        eth_task = self.get_eth_price()

        pools, gas, eth_price = await asyncio.gather(
            pools_task, gas_task, eth_task
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "eth_price_usd": eth_price,
            "gas": gas.to_dict() if gas else None,
            "pools": [p.to_dict() for p in pools],
            "pool_count": len(pools),
            "top_apy": max((p.apy for p in pools), default=0),
            "avg_apy": sum(p.apy for p in pools) / len(pools) if pools else 0
        }

    def compute_trend(self, time_series: TimeSeries) -> dict:
        """
        Compute trend metrics from a time series.
        Returns slope, direction, and volatility.
        """
        values = time_series.values
        if len(values) < 2:
            return {"direction": "unknown", "slope": 0, "volatility": 0}

        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Normalized slope (% change per day relative to mean)
        norm_slope = (slope / y_mean) if y_mean != 0 else 0

        # Direction based on config thresholds
        if norm_slope > config.analysis.trend_rising_threshold:
            direction = "rising"
        elif norm_slope < config.analysis.trend_falling_threshold:
            direction = "falling"
        else:
            direction = "stable"

        # Volatility (coefficient of variation)
        variance = sum((v - y_mean) ** 2 for v in values) / n
        std_dev = variance ** 0.5
        volatility = std_dev / y_mean if y_mean != 0 else 0

        return {
            "direction": direction,
            "slope": round(slope, 4),
            "normalized_slope": round(norm_slope, 4),
            "volatility": round(volatility, 4),
            "mean": round(y_mean, 2),
            "latest": values[-1] if values else 0
        }

    async def close(self):
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Global instance
data_service = DataService()
