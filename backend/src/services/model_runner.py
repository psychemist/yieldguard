"""
YieldGuard Lite - Model Runner
Bridges the agentic system with the API layer.
"""

from datetime import datetime

from ..models.recommendation import AllocationItem, RecommendationResponse, RiskProfile, YieldData
from .agent import get_agent


class ModelRunner:
    """
    AI Agent-Powered Model Runner
    - Uses true agentic system with tools, planning, ReAct, and memory
    - Provides backward-compatible interface for the API layer
    """

    def __init__(self):
        self.agent = get_agent()

    async def generate_recommendation(
        self, capital: float, risk_profile: RiskProfile, yield_data: list[YieldData], gas_data: dict
    ) -> RecommendationResponse:
        """
        Generate AI Agent-powered yield optimization.
        Uses the agentic system's quick_strategy for direct tool calls,
        or process_request for full ReAct reasoning.
        """
        try:
            print(f"🤖 Activating Agentic System for ${capital:,} ({risk_profile.value})...")

            # Store user preferences in agent memory
            self.agent.memory.set_preference("investment_amount", capital)
            self.agent.memory.set_preference("risk_tolerance", risk_profile.value)

            # Use quick_strategy for direct tool invocation (faster)
            result = await self.agent.get_quick_strategy(investment_amount=capital, risk_tolerance=risk_profile.value)

            if not result.get("success"):
                print(f"⚠️ Agent strategy failed: {result.get('error')}")
                return self._generate_simple_fallback(capital, risk_profile, yield_data, gas_data)

            strategy = result.get("strategy", {})
            allocations_data = strategy.get("allocations", [])
            summary = strategy.get("summary", {})

            # Convert to API response format
            allocations = []
            for alloc in allocations_data:
                allocations.append(
                    AllocationItem(
                        asset=alloc.get("symbol", "Unknown"),
                        percentage=alloc.get("allocation_pct", 0),
                        expected_yield=alloc.get("apy", 0),
                        risk_score=self._estimate_risk_score(alloc, risk_profile),
                    )
                )

            if not allocations:
                return self._generate_simple_fallback(capital, risk_profile, yield_data, gas_data)

            total_yield = summary.get("weighted_avg_apy", 0)
            gas_cost = summary.get("total_gas_cost", 0)

            # Calculate confidence based on pool quality
            avg_tvl = (
                sum(a.get("tvl_usd", 0) for a in allocations_data) / len(allocations_data) if allocations_data else 0
            )
            confidence = min(0.95, 0.7 + (avg_tvl / 1e10))  # Higher TVL = higher confidence

            print(f"🧠 Agent complete: {total_yield:.2f}% APY, ${gas_cost:.2f} gas, {confidence:.0%} confidence")

            return RecommendationResponse(
                timestamp=datetime.now(),
                capital=capital,
                risk_profile=risk_profile,
                allocations=allocations,
                total_expected_yield=total_yield,
                total_risk_score=self._calculate_total_risk(allocations),
                gas_cost_estimate=gas_cost,
                confidence_score=confidence,
            )

        except Exception as e:
            print(f"❌ Agent error: {e}")
            return self._generate_simple_fallback(capital, risk_profile, yield_data, gas_data)

    def _estimate_risk_score(self, alloc: dict, risk_profile: RiskProfile) -> float:
        """Estimate risk score for an allocation."""
        tvl = alloc.get("tvl_usd", 0)
        apy = alloc.get("apy", 0)

        # Higher TVL = lower risk
        tvl_risk = max(0, 0.5 - (tvl / 2e9))

        # Higher APY = higher risk
        apy_risk = min(0.5, apy / 100)

        base_risk = tvl_risk + apy_risk

        # Adjust by risk profile
        if risk_profile == RiskProfile.LOW:
            return min(0.3, base_risk * 0.7)
        elif risk_profile == RiskProfile.MEDIUM:
            return min(0.5, base_risk)
        else:  # HIGH
            return min(0.7, base_risk * 1.2)

    def _calculate_total_risk(self, allocations: list[AllocationItem]) -> float:
        """Calculate weighted total risk score."""
        if not allocations:
            return 0.5

        weighted_risk = sum(a.risk_score * a.percentage / 100 for a in allocations)
        return round(weighted_risk, 2)

    def _generate_simple_fallback(
        self, capital: float, risk_profile: RiskProfile, yield_data: list[YieldData], gas_data: dict
    ) -> RecommendationResponse:
        """Simple fallback if agent fails."""
        print("🔄 Using fallback recommendation logic...")

        if not yield_data:
            yield_data = [
                YieldData(protocol="Aave", asset="USDC", apy=3.5, tvl=1e9, timestamp=datetime.now()),
                YieldData(protocol="Compound", asset="USDT", apy=2.8, tvl=8e8, timestamp=datetime.now()),
            ]

        # Take top 3 by TVL
        safe_assets = sorted(yield_data, key=lambda x: x.tvl, reverse=True)[:3]

        allocations = []
        percentages = [50.0, 30.0, 20.0][: len(safe_assets)]

        for i, asset in enumerate(safe_assets):
            allocations.append(
                AllocationItem(asset=asset.asset, percentage=percentages[i], expected_yield=asset.apy, risk_score=0.4)
            )

        total_yield = sum(alloc.expected_yield * alloc.percentage / 100 for alloc in allocations)

        return RecommendationResponse(
            timestamp=datetime.now(),
            capital=capital,
            risk_profile=risk_profile,
            allocations=allocations,
            total_expected_yield=total_yield,
            total_risk_score=0.4,
            gas_cost_estimate=gas_data.get("standard", 25) * 10 if gas_data else 250,
            confidence_score=0.6,
        )

    async def chat(self, message: str) -> dict:
        """
        Process a natural language request using full agentic capabilities.
        Returns the full response including planning and ReAct trace.
        """
        return await self.agent.process_request(message)

    def get_agent_status(self) -> dict:
        """Get current agent state."""
        return {
            "tools_available": self.agent.tools.list_tools(),
            "memory_summary": self.agent.get_memory_summary(),
            "current_plan": {
                "goal": self.agent.current_plan.goal if self.agent.current_plan else None,
                "steps": len(self.agent.current_plan.steps) if self.agent.current_plan else 0,
            },
        }
