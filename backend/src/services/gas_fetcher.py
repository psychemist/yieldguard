import httpx
import asyncio
from typing import Dict
from datetime import datetime
from ..models.recommendation import GasData

class GasFetcher:
    """
    Service to fetch current gas prices from Ethereum network
    """
    
    def __init__(self):
        # Using Etherscan API for gas prices
        self.etherscan_url = "https://api.etherscan.io/api"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_current_gas_price(self) -> GasData:
        """
        Fetch current gas prices from Ethereum network
        """
        try:
            # Try to get gas prices from Etherscan
            response = await self.client.get(
                f"{self.etherscan_url}?module=gastracker&action=gasoracle"
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "1":
                    result = data.get("result", {})
                    return GasData(
                        slow=float(result.get("SafeGasPrice", 20)),
                        standard=float(result.get("ProposeGasPrice", 25)),
                        fast=float(result.get("FastGasPrice", 30)),
                        timestamp=datetime.now()
                    )
            
            # Fallback to mock data
            return self._get_mock_gas_data()
            
        except Exception as e:
            print(f"Error fetching gas prices: {e}")
            return self._get_mock_gas_data()
    
    def _get_mock_gas_data(self) -> GasData:
        """Mock gas data for development"""
        return GasData(
            slow=20.0,
            standard=25.0,
            fast=30.0,
            timestamp=datetime.now()
        )
    
    async def estimate_transaction_cost(self, gas_price: float, gas_limit: int = 21000) -> float:
        """
        Estimate transaction cost in USD
        """
        try:
            # Get ETH price from CoinGecko
            response = await self.client.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
            )
            
            if response.status_code == 200:
                data = response.json()
                eth_price = data.get("ethereum", {}).get("usd", 2000)
                
                # Calculate cost: gas_price (gwei) * gas_limit * ETH_price / 1e9
                cost_usd = (gas_price * gas_limit * eth_price) / 1e9
                return cost_usd
            
            # Fallback calculation with assumed ETH price
            return (gas_price * gas_limit * 2000) / 1e9
            
        except Exception as e:
            print(f"Error estimating transaction cost: {e}")
            return (gas_price * gas_limit * 2000) / 1e9
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()