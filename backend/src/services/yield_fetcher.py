import httpx
import asyncio
from typing import Dict, List
from datetime import datetime, timedelta
from ..models.recommendation import YieldData

class YieldFetcher:
    """
    Service to fetch yield data from DeFi protocols
    For MVP, we'll focus on Uniswap V3 and use DefiLlama API
    """
    
    def __init__(self):
        self.base_url = "https://yields.llama.fi"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_current_yields(self) -> List[YieldData]:
        """
        Fetch current yield data from supported protocols
        """
        try:
            # Get pools from DefiLlama
            response = await self.client.get(f"{self.base_url}/pools")
            data = response.json()
            
            # Filter for Uniswap V3 pools for MVP
            uniswap_pools = [
                pool for pool in data.get("data", [])
                if pool.get("project") == "uniswap-v3" and 
                pool.get("chain") == "Ethereum" and
                pool.get("apy", 0) > 0
            ]
            
            # Convert to our YieldData format
            yield_data = []
            for pool in uniswap_pools[:10]:  # Limit to top 10 for MVP
                yield_data.append(YieldData(
                    protocol="uniswap-v3",
                    asset=pool.get("symbol", ""),
                    apy=pool.get("apy", 0),
                    tvl=pool.get("tvlUsd", 0),
                    timestamp=datetime.now()
                ))
            
            return yield_data
            
        except Exception as e:
            print(f"Error fetching yields: {e}")
            # Return mock data for development
            return self._get_mock_yields()
    
    async def get_historical_yields(self, days: int = 30) -> Dict:
        """
        Get historical yield data for charting
        """
        try:
            # For MVP, we'll use mock historical data
            # In production, this would fetch from DefiLlama historical endpoint
            return self._get_mock_historical_data(days)
            
        except Exception as e:
            print(f"Error fetching historical yields: {e}")
            return self._get_mock_historical_data(days)
    
    def _get_mock_yields(self) -> List[YieldData]:
        """Mock data for development"""
        return [
            YieldData(
                protocol="uniswap-v3",
                asset="USDC/ETH",
                apy=12.5,
                tvl=50000000,
                timestamp=datetime.now()
            ),
            YieldData(
                protocol="uniswap-v3",
                asset="USDT/ETH",
                apy=8.3,
                tvl=30000000,
                timestamp=datetime.now()
            ),
            YieldData(
                protocol="uniswap-v3",
                asset="WBTC/ETH",
                apy=15.2,
                tvl=20000000,
                timestamp=datetime.now()
            )
        ]
    
    def _get_mock_historical_data(self, days: int) -> Dict:
        """Mock historical data for development"""
        dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
        return {
            "dates": dates,
            "yields": {
                "USDC/ETH": [12.5 + (i * 0.1) for i in range(days)],
                "USDT/ETH": [8.3 + (i * 0.05) for i in range(days)],
                "WBTC/ETH": [15.2 + (i * 0.2) for i in range(days)]
            }
        }
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()