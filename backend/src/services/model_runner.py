import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime
from ..models.recommendation import (
    RecommendationResponse, 
    AllocationItem, 
    YieldData, 
    GasData, 
    RiskProfile
)

class ModelRunner:
    """
    AI Model Runner for yield optimization recommendations
    For MVP, we'll use a simplified rule-based approach with basic ML concepts
    """
    
    def __init__(self):
        self.risk_multipliers = {
            RiskProfile.LOW: 0.5,
            RiskProfile.MEDIUM: 1.0,
            RiskProfile.HIGH: 1.5
        }
        
        # Asset risk scores (0-1 scale)
        self.asset_risk_scores = {
            "USDC/ETH": 0.3,
            "USDT/ETH": 0.4,
            "WBTC/ETH": 0.6,
            "ETH/USDC": 0.3,
            "ETH/USDT": 0.4,
            "ETH/WBTC": 0.6
        }
    
    async def generate_recommendation(
        self,
        capital: float,
        risk_profile: RiskProfile,
        yield_data: List[YieldData],
        gas_data: GasData
    ) -> RecommendationResponse:
        """
        Generate AI-powered yield optimization recommendation
        """
        try:
            # Calculate risk-adjusted scores for each asset
            scored_assets = self._calculate_risk_adjusted_scores(yield_data, risk_profile)
            
            # Generate optimal allocation
            allocations = self._generate_allocations(scored_assets, capital, risk_profile)
            
            # Calculate gas cost estimate
            gas_cost = await self._estimate_gas_costs(gas_data, len(allocations))
            
            # Calculate overall metrics
            total_yield = sum(alloc.expected_yield * alloc.percentage / 100 for alloc in allocations)
            total_risk = sum(alloc.risk_score * alloc.percentage / 100 for alloc in allocations)
            
            # Generate confidence score based on data quality and market conditions
            confidence = self._calculate_confidence_score(yield_data, gas_data)
            
            return RecommendationResponse(
                timestamp=datetime.now(),
                capital=capital,
                risk_profile=risk_profile,
                allocations=allocations,
                total_expected_yield=total_yield,
                total_risk_score=total_risk,
                gas_cost_estimate=gas_cost,
                confidence_score=confidence
            )
            
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return self._generate_fallback_recommendation(capital, risk_profile)
    
    def _calculate_risk_adjusted_scores(
        self, 
        yield_data: List[YieldData], 
        risk_profile: RiskProfile
    ) -> List[Dict]:
        """
        Calculate risk-adjusted scores for each asset
        """
        scored_assets = []
        risk_multiplier = self.risk_multipliers[risk_profile]
        
        for asset_data in yield_data:
            asset_name = asset_data.asset
            base_risk = self.asset_risk_scores.get(asset_name, 0.5)
            
            # Risk-adjusted yield calculation
            # Higher risk tolerance = higher weight on yield
            # Lower risk tolerance = higher penalty for risky assets
            if risk_profile == RiskProfile.LOW:
                score = asset_data.apy * (1 - base_risk * 0.8)
            elif risk_profile == RiskProfile.MEDIUM:
                score = asset_data.apy * (1 - base_risk * 0.4)
            else:  # HIGH
                score = asset_data.apy * (1 + base_risk * 0.2)
            
            # TVL factor (higher TVL = more stable, slight bonus)
            tvl_factor = min(1.1, 1 + (asset_data.tvl / 100000000) * 0.1)
            score *= tvl_factor
            
            scored_assets.append({
                'asset': asset_name,
                'score': score,
                'apy': asset_data.apy,
                'risk': base_risk,
                'tvl': asset_data.tvl
            })
        
        return sorted(scored_assets, key=lambda x: x['score'], reverse=True)
    
    def _generate_allocations(
        self, 
        scored_assets: List[Dict], 
        capital: float, 
        risk_profile: RiskProfile
    ) -> List[AllocationItem]:
        """
        Generate allocation percentages based on scores and risk profile
        """
        allocations = []
        
        # Determine number of assets based on risk profile and capital
        if risk_profile == RiskProfile.LOW:
            max_assets = min(3, len(scored_assets))
            concentration_limit = 60  # Max 60% in any single asset
        elif risk_profile == RiskProfile.MEDIUM:
            max_assets = min(4, len(scored_assets))
            concentration_limit = 70
        else:  # HIGH
            max_assets = min(5, len(scored_assets))
            concentration_limit = 80
        
        # Take top assets
        top_assets = scored_assets[:max_assets]
        
        # Calculate weights based on scores
        total_score = sum(asset['score'] for asset in top_assets)
        
        for asset in top_assets:
            # Base allocation based on score
            base_percentage = (asset['score'] / total_score) * 100
            
            # Apply concentration limit
            percentage = min(base_percentage, concentration_limit)
            
            allocations.append(AllocationItem(
                asset=asset['asset'],
                percentage=round(percentage, 2),
                expected_yield=asset['apy'],
                risk_score=asset['risk']
            ))
        
        # Normalize to 100%
        total_percentage = sum(alloc.percentage for alloc in allocations)
        if total_percentage != 100:
            factor = 100 / total_percentage
            for alloc in allocations:
                alloc.percentage = round(alloc.percentage * factor, 2)
        
        return allocations
    
    async def _estimate_gas_costs(self, gas_data: GasData, num_transactions: int) -> float:
        """
        Estimate total gas costs for implementing the strategy
        """
        # Estimate gas usage:
        # - Token approvals: ~50k gas each
        # - Liquidity provision: ~200k gas each
        # - Asset swaps: ~150k gas each
        
        approval_gas = 50000 * num_transactions
        liquidity_gas = 200000 * num_transactions
        swap_gas = 150000 * (num_transactions - 1)  # One less swap than assets
        
        total_gas = approval_gas + liquidity_gas + swap_gas
        
        # Use standard gas price for estimation
        gas_price_wei = gas_data.standard * 1e9  # Convert gwei to wei
        gas_cost_eth = (total_gas * gas_price_wei) / 1e18
        
        # Convert to USD (assuming ~$2000 ETH for MVP)
        gas_cost_usd = gas_cost_eth * 2000
        
        return round(gas_cost_usd, 2)
    
    def _calculate_confidence_score(self, yield_data: List[YieldData], gas_data: GasData) -> float:
        """
        Calculate confidence score based on data quality and market conditions
        """
        # Base confidence
        confidence = 0.8
        
        # Data quality factors
        if len(yield_data) >= 5:
            confidence += 0.1
        elif len(yield_data) < 3:
            confidence -= 0.2
        
        # Gas price stability (lower gas = higher confidence)
        if gas_data.standard < 20:
            confidence += 0.05
        elif gas_data.standard > 50:
            confidence -= 0.1
        
        # TVL quality check
        avg_tvl = sum(asset.tvl for asset in yield_data) / len(yield_data) if yield_data else 0
        if avg_tvl > 50000000:  # High TVL = more stable
            confidence += 0.05
        
        return round(max(0.3, min(0.95, confidence)), 2)
    
    def _generate_fallback_recommendation(
        self, 
        capital: float, 
        risk_profile: RiskProfile
    ) -> RecommendationResponse:
        """
        Generate a safe fallback recommendation when the main algorithm fails
        """
        if risk_profile == RiskProfile.LOW:
            allocations = [
                AllocationItem(asset="USDC/ETH", percentage=60.0, expected_yield=8.0, risk_score=0.3),
                AllocationItem(asset="USDT/ETH", percentage=40.0, expected_yield=6.5, risk_score=0.4)
            ]
        elif risk_profile == RiskProfile.MEDIUM:
            allocations = [
                AllocationItem(asset="USDC/ETH", percentage=40.0, expected_yield=10.0, risk_score=0.3),
                AllocationItem(asset="USDT/ETH", percentage=35.0, expected_yield=8.5, risk_score=0.4),
                AllocationItem(asset="WBTC/ETH", percentage=25.0, expected_yield=12.0, risk_score=0.6)
            ]
        else:  # HIGH
            allocations = [
                AllocationItem(asset="WBTC/ETH", percentage=50.0, expected_yield=15.0, risk_score=0.6),
                AllocationItem(asset="USDC/ETH", percentage=30.0, expected_yield=10.0, risk_score=0.3),
                AllocationItem(asset="USDT/ETH", percentage=20.0, expected_yield=8.5, risk_score=0.4)
            ]
        
        total_yield = sum(alloc.expected_yield * alloc.percentage / 100 for alloc in allocations)
        total_risk = sum(alloc.risk_score * alloc.percentage / 100 for alloc in allocations)
        
        return RecommendationResponse(
            timestamp=datetime.now(),
            capital=capital,
            risk_profile=risk_profile,
            allocations=allocations,
            total_expected_yield=total_yield,
            total_risk_score=total_risk,
            gas_cost_estimate=50.0,  # Fallback gas estimate
            confidence_score=0.6
        )