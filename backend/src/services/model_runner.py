import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime
from .ai_agent import YieldOptimizationAgent
from ..models.recommendation import (
    RecommendationResponse,
    AllocationItem,
    YieldData,
    RiskProfile
)

class ModelRunner:
    """
    AI Agent-Powered Model Runner using LangChain and GPT-4
    - Uses sophisticated AI Agent with reasoning capabilities
    - Agent can use tools to gather market intelligence
    - Makes adaptive decisions based on real-time analysis
    """
    
    def __init__(self):
        self.ai_agent = YieldOptimizationAgent()
    
    async def generate_recommendation(
        self,
        capital: float,
        risk_profile: RiskProfile,
        yield_data: List[YieldData],
        gas_data: Dict
    ) -> RecommendationResponse:
        """
        Generate AI Agent-powered yield optimization using advanced reasoning
        """
        try:
            print(f"🤖 Activating AI Agent with reasoning capabilities for ${capital:,}...")
            
            if not yield_data:
                raise Exception("No real yield data available for AI Agent analysis")
            
            # Use AI Agent with tools and reasoning
            recommendation = await self.ai_agent.analyze_and_recommend(
                capital=capital,
                risk_profile=risk_profile,
                yield_data=yield_data,
                gas_data=gas_data
            )
            
            print(f"🧠 AI Agent reasoning complete: {recommendation.total_expected_yield:.2f}% yield with {recommendation.confidence_score:.0%} confidence")
            
            return recommendation
            
        except Exception as e:
            print(f"❌ AI Agent failed: {e}")
            # Fallback to simple allocation if AI fails
            return self._generate_simple_fallback(capital, risk_profile, yield_data, gas_data)
    
    def _generate_simple_fallback(
        self, 
        capital: float, 
        risk_profile: RiskProfile,
        yield_data: List[YieldData],
        gas_data: Dict
    ) -> RecommendationResponse:
        """
        Simple fallback if AI agent fails
        """
        print("🔄 Using fallback recommendation logic...")
        
        # Take top 3 safest assets by TVL
        safe_assets = sorted(yield_data, key=lambda x: x.tvl, reverse=True)[:3]
        
        allocations = []
        percentages = [50.0, 30.0, 20.0]  # Simple allocation
        
        for i, asset in enumerate(safe_assets):
            allocations.append(AllocationItem(
                asset=asset.asset,
                percentage=percentages[i],
                expected_yield=asset.apy,
                risk_score=0.4  # Conservative risk estimate
            ))
        
        total_yield = sum(alloc.expected_yield * alloc.percentage / 100 for alloc in allocations)
        
        return RecommendationResponse(
            timestamp=datetime.now(),
            capital=capital,
            risk_profile=risk_profile,
            allocations=allocations,
            total_expected_yield=total_yield,
            total_risk_score=0.4,
            gas_cost_estimate=gas_data.get('standard', 25) * 10,  # Simple gas estimate
            confidence_score=0.6  # Lower confidence for fallback
        )