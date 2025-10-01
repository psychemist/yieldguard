import httpx
import asyncio
from typing import Dict
from datetime import datetime

class GasFetcher:
    """
    Service to fetch REAL current gas prices from Ethereum network
    """
    
    def __init__(self):
        # Using multiple APIs for gas prices
        self.etherscan_url = "https://api.etherscan.io/api"
        self.owlracle_url = "https://api.owlracle.info/v4/eth"
        self.blocknative_url = "https://api.blocknative.com/gasprices/blockprices"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_current_gas_price(self) -> Dict:
        """
        Fetch REAL current gas prices from Ethereum network
        """
        try:
            print("Fetching real gas prices from Ethereum network...")
            
            # Try Owlracle first (free, no API key needed)
            try:
                response = await self.client.get(f"{self.owlracle_url}/gas")
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract gas prices from response
                    slow = data.get("speeds", [{}])[0].get("gasPrice", 20)
                    standard = data.get("speeds", [{}])[1].get("gasPrice", 25) if len(data.get("speeds", [])) > 1 else slow + 5
                    fast = data.get("speeds", [{}])[2].get("gasPrice", 30) if len(data.get("speeds", [])) > 2 else standard + 5
                    
                    print(f"Real gas prices: Slow={slow}, Standard={standard}, Fast={fast}")
                    
                    return {
                        "slow": float(slow),
                        "standard": float(standard),
                        "fast": float(fast),
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception as e:
                print(f"Owlracle API failed: {e}")
            
            # Fallback to Etherscan
            try:
                response = await self.client.get(
                    f"{self.etherscan_url}?module=gastracker&action=gasoracle"
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "1":
                        result = data.get("result", {})
                        
                        slow = float(result.get("SafeGasPrice", 20))
                        standard = float(result.get("ProposeGasPrice", 25))
                        fast = float(result.get("FastGasPrice", 30))
                        
                        print(f"Real gas prices from Etherscan: Slow={slow}, Standard={standard}, Fast={fast}")
                        
                        return {
                            "slow": slow,
                            "standard": standard,
                            "fast": fast,
                            "timestamp": datetime.now().isoformat()
                        }
            except Exception as e:
                print(f"Etherscan API failed: {e}")
            
            # If all APIs fail, raise error
            raise Exception("All gas price APIs failed")
            
        except Exception as e:
            print(f"Error fetching real gas prices: {e}")
            raise Exception(f"Failed to fetch real gas data: {e}")
    
    async def estimate_transaction_cost(self, gas_price: float, gas_limit: int = 21000) -> float:
        """
        Estimate REAL transaction cost in USD using current ETH price
        """
        try:
            print("Fetching real ETH price for gas cost calculation...")
            
            # Get current ETH price from CoinGecko
            response = await self.client.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
            )
            
            if response.status_code == 200:
                data = response.json()
                eth_price = data.get("ethereum", {}).get("usd")
                
                if eth_price:
                    # Calculate real cost: gas_price (gwei) * gas_limit * ETH_price / 1e9
                    cost_usd = (gas_price * gas_limit * eth_price) / 1e9
                    print(f"Real transaction cost: ${cost_usd:.2f} (ETH price: ${eth_price})")
                    return cost_usd
            
            raise Exception("Failed to get ETH price")
            
        except Exception as e:
            print(f"Error calculating real transaction cost: {e}")
            raise Exception(f"Failed to calculate real gas cost: {e}")
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()