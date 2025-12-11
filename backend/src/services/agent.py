"""
YieldGuard Lite - AI Yield Optimization Agent
Simplified, production-ready agent focused on Uniswap/DeFi yield optimization on Ethereum.
Implements: Tool System, Planning Loop, ReAct Pattern, Memory System
"""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from groq import AsyncGroq

from ..utils.config import config
from .data_service import DataService

logger = logging.getLogger(__name__)


# =============================================================================
# 1. TOOL SYSTEM - Callable tools the agent can invoke dynamically
# =============================================================================


class ToolRegistry:
    """Registry of tools the agent can invoke."""

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}

    def register(self, name: str, func: Callable, schema: dict):
        """Register a tool with its schema."""
        self._tools[name] = func
        self._schemas[name] = schema

    def get_tool(self, name: str) -> Callable | None:
        return self._tools.get(name)

    def get_all_schemas(self) -> list[dict]:
        """Get all tool schemas for LLM function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema.get("description", ""),
                    "parameters": schema.get("parameters", {}),
                },
            }
            for name, schema in self._schemas.items()
        ]

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


# =============================================================================
# 2. MEMORY SYSTEM - Conversation history, preferences, past recommendations
# =============================================================================


class MemoryType(Enum):
    CONVERSATION = "conversation"
    USER_PREFERENCE = "user_preference"
    RECOMMENDATION = "recommendation"
    OBSERVATION = "observation"


@dataclass
class MemoryEntry:
    """Single memory entry."""

    type: MemoryType
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class AgentMemory:
    """Memory system for the agent."""

    def __init__(self, max_conversation_history: int = 20):
        self.max_history = max_conversation_history
        self._conversation: list[dict] = []
        self._preferences: dict[str, Any] = {}
        self._recommendations: list[MemoryEntry] = []
        self._observations: list[MemoryEntry] = []

    def add_message(self, role: str, content: str):
        """Add to conversation history."""
        self._conversation.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
        # Trim old messages
        if len(self._conversation) > self.max_history:
            self._conversation = self._conversation[-self.max_history :]

    def add_tool_result(self, tool_name: str, result: Any):
        """Store tool observation."""
        self._observations.append(
            MemoryEntry(
                type=MemoryType.OBSERVATION,
                content={"tool": tool_name, "result": result},
                metadata={"tool_name": tool_name},
            )
        )

    def set_preference(self, key: str, value: Any):
        """Store user preference."""
        self._preferences[key] = value

    def get_preference(self, key: str, default: Any = None) -> Any:
        return self._preferences.get(key, default)

    def add_recommendation(self, recommendation: dict):
        """Store past recommendation."""
        self._recommendations.append(MemoryEntry(type=MemoryType.RECOMMENDATION, content=recommendation))
        # Keep last 10 recommendations
        if len(self._recommendations) > 10:
            self._recommendations = self._recommendations[-10:]

    def get_conversation_for_llm(self) -> list[dict]:
        """Get conversation history formatted for LLM."""
        return [{"role": m["role"], "content": m["content"]} for m in self._conversation]

    def get_context_summary(self) -> str:
        """Get summary of memory for context."""
        parts = []
        if self._preferences:
            parts.append(f"User Preferences: {json.dumps(self._preferences)}")
        if self._recommendations:
            recent = self._recommendations[-3:]
            parts.append(f"Recent Recommendations: {len(recent)} strategies suggested")
        return "\n".join(parts) if parts else "No prior context"

    def clear(self):
        """Clear all memory."""
        self._conversation.clear()
        self._preferences.clear()
        self._recommendations.clear()
        self._observations.clear()


# =============================================================================
# 3. REACT PATTERN - Thought → Action → Observation cycle
# =============================================================================


class ReActStep(Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


@dataclass
class ReActTrace:
    """Single step in ReAct trace."""

    step_type: ReActStep
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# 4. PLANNING LOOP - Multi-step plan creation and execution
# =============================================================================


@dataclass
class PlanStep:
    """Single step in a plan."""

    step_number: int
    description: str
    tool_name: str | None = None
    tool_args: dict | None = None
    status: str = "pending"  # pending, running, completed, failed
    result: Any | None = None


@dataclass
class ExecutionPlan:
    """Multi-step execution plan."""

    goal: str
    steps: list[PlanStep]
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "created"  # created, running, completed, failed


# =============================================================================
# MAIN AGENT CLASS
# =============================================================================


class YieldOptimizationAgent:
    """
    True agentic yield optimization system with:
    - Tool System: Dynamic tool invocation
    - Planning Loop: Multi-step reasoning
    - ReAct Pattern: Thought-Action-Observation cycles
    - Memory System: Context retention across interactions
    """

    def __init__(self):
        self.client = AsyncGroq(api_key=config.groq_api_key)
        self.data_service = DataService()
        self.memory = AgentMemory()
        self.tools = ToolRegistry()
        self.react_trace: list[ReActTrace] = []
        self.current_plan: ExecutionPlan | None = None
        self._register_tools()

    def _register_tools(self):
        """Register all available tools."""

        # Tool: Fetch Yield Pools
        self.tools.register(
            "fetch_yield_pools",
            self._tool_fetch_yields,
            {
                "description": "Fetch current DeFi yield opportunities from various protocols",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Blockchain networks to fetch yields from (e.g., 'ethereum')",
                        },
                        "min_tvl": {"type": "number", "description": "Minimum TVL in USD to filter pools"},
                        "min_apy": {"type": "number", "description": "Minimum APY percentage to filter pools"},
                    },
                },
            },
        )

        # Tool: Fetch Gas Prices
        self.tools.register(
            "fetch_gas_prices",
            self._tool_fetch_gas,
            {
                "description": "Fetch current gas prices for transaction cost estimation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chain": {
                            "type": "string",
                            "description": "Blockchain to get gas prices for (default: ethereum)",
                        }
                    },
                },
            },
        )

        # Tool: Analyze Risk
        self.tools.register(
            "analyze_risk",
            self._tool_analyze_risk,
            {
                "description": "Analyze risk factors for a specific yield pool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pool_id": {"type": "string", "description": "The pool ID to analyze"},
                        "investment_amount": {"type": "number", "description": "Amount in USD to invest"},
                    },
                    "required": ["pool_id"],
                },
            },
        )

        # Tool: Calculate Strategy
        self.tools.register(
            "calculate_strategy",
            self._tool_calculate_strategy,
            {
                "description": "Calculate optimal allocation strategy based on pools and constraints",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "investment_amount": {"type": "number", "description": "Total amount to invest in USD"},
                        "risk_tolerance": {
                            "type": "string",
                            "enum": ["conservative", "moderate", "aggressive"],
                            "description": "User's risk tolerance level",
                        },
                        "preferred_chains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Preferred blockchain networks",
                        },
                        "max_pools": {"type": "integer", "description": "Maximum number of pools to include"},
                    },
                    "required": ["investment_amount", "risk_tolerance"],
                },
            },
        )

        # Tool: Get Historical Performance
        self.tools.register(
            "get_historical_performance",
            self._tool_historical_performance,
            {
                "description": "Get historical APY performance for yield pools",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pool_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Pool IDs to get history for",
                        },
                        "days": {"type": "integer", "description": "Number of days of history (default: 30)"},
                    },
                },
            },
        )

        # Tool: Compare Protocols
        self.tools.register(
            "compare_protocols",
            self._tool_compare_protocols,
            {
                "description": "Compare different DeFi protocols by safety, yields, and features",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "protocols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Protocol names to compare",
                        }
                    },
                },
            },
        )

    # =========================================================================
    # TOOL IMPLEMENTATIONS
    # =========================================================================

    async def _tool_fetch_yields(
        self,
        chains: list[str] | None = None,
        min_tvl: float | None = None,
        min_apy: float | None = None,
    ) -> dict:
        """Fetch yield pools with filtering."""
        try:
            pools = await self.data_service.get_yield_pools()

            # Apply filters
            if chains:
                chains_lower = [c.lower() for c in chains]
                pools = [p for p in pools if p.chain.lower() in chains_lower]

            if min_tvl is not None:
                pools = [p for p in pools if p.tvl_usd >= min_tvl]

            if min_apy is not None:
                pools = [p for p in pools if p.apy >= min_apy]

            return {
                "success": True,
                "pool_count": len(pools),
                "pools": [
                    {
                        "id": p.pool_id,
                        "protocol": p.protocol,
                        "chain": p.chain,
                        "symbol": p.symbol,
                        "apy": round(p.apy, 2),
                        "tvl_usd": round(p.tvl_usd, 2),
                        "stable": p.stable_coin,
                    }
                    for p in pools[:20]  # Limit for LLM context
                ],
            }
        except Exception as e:
            logger.error(f"Tool fetch_yields error: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_fetch_gas(self, chain: str = "ethereum") -> dict:
        """Fetch gas prices."""
        try:
            gas_data = await self.data_service.get_gas_data()
            return {
                "success": True,
                "chain": chain,
                "gas_prices": {
                    "slow": gas_data.slow_gwei,
                    "standard": gas_data.standard_gwei,
                    "fast": gas_data.fast_gwei,
                },
                "eth_price_usd": gas_data.eth_price_usd,
                "recommendation": "standard" if gas_data.standard_gwei < 50 else "slow",
            }
        except Exception as e:
            logger.error(f"Tool fetch_gas error: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_analyze_risk(self, pool_id: str, investment_amount: float = 1000) -> dict:
        """Analyze risk for a specific pool."""
        try:
            pools = await self.data_service.get_yield_pools()
            pool = next((p for p in pools if p.pool_id == pool_id), None)

            if not pool:
                return {"success": False, "error": f"Pool {pool_id} not found"}

            # Risk scoring based on config thresholds
            risk_factors = []
            risk_score = 0

            # TVL risk
            if pool.tvl_usd < config.filters.min_tvl_usd:
                risk_factors.append(
                    f"Low TVL: ${pool.tvl_usd:,.0f} (below ${config.filters.min_tvl_usd:,.0f} threshold)"
                )
                risk_score += 30
            elif pool.tvl_usd < config.filters.min_tvl_usd * 5:
                risk_factors.append(f"Moderate TVL: ${pool.tvl_usd:,.0f}")
                risk_score += 15

            # APY risk (unsustainable yields)
            if pool.apy > config.filters.max_apy_percent:
                risk_factors.append(f"Very high APY: {pool.apy:.1f}% may be unsustainable")
                risk_score += 25
            elif pool.apy > config.filters.max_apy_percent * 0.5:
                risk_factors.append(f"High APY: {pool.apy:.1f}% - verify sustainability")
                risk_score += 10

            # Stablecoin preference for conservative
            if not pool.stable_coin:
                risk_factors.append("Non-stablecoin pool - exposed to price volatility")
                risk_score += 15

            # IL risk
            if pool.il_risk and pool.il_risk.lower() != "none":
                risk_factors.append(f"Impermanent loss risk: {pool.il_risk}")
                risk_score += 20

            risk_level = "low" if risk_score < 30 else "medium" if risk_score < 60 else "high"

            return {
                "success": True,
                "pool_id": pool_id,
                "protocol": pool.protocol,
                "chain": pool.chain,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "recommendation": "suitable" if risk_score < 50 else "caution" if risk_score < 70 else "avoid",
            }
        except Exception as e:
            logger.error(f"Tool analyze_risk error: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_calculate_strategy(
        self,
        investment_amount: float,
        risk_tolerance: str,
        preferred_chains: list[str] | None = None,
        max_pools: int | None = None,
    ) -> dict:
        """Calculate optimal allocation strategy."""
        try:
            pools = await self.data_service.get_yield_pools()
            gas_data = await self.data_service.get_gas_data()

            if not pools:
                return {"success": False, "error": "No yield pools available"}
            if not gas_data:
                return {"success": False, "error": "Gas data unavailable"}

            # Normalize risk tolerance (support both naming conventions)
            risk_lower = risk_tolerance.lower()
            if risk_lower in ("low", "conservative"):
                risk_level = "low"
            elif risk_lower in ("medium", "moderate"):
                risk_level = "medium"
            else:  # high, aggressive
                risk_level = "high"

            # Risk-based filtering using config
            if risk_level == "low":
                min_tvl = config.filters.min_tvl_usd * 2  # Require higher TVL for safety
                max_apy = config.filters.max_apy_percent * 0.3  # Cap at 30% of max
                prefer_stable = True
                default_max_pools = 3
            elif risk_level == "medium":
                min_tvl = config.filters.min_tvl_usd
                max_apy = config.filters.max_apy_percent * 0.6
                prefer_stable = False
                default_max_pools = 5
            else:  # high
                min_tvl = config.filters.min_tvl_usd * 0.5  # Accept lower TVL
                max_apy = config.filters.max_apy_percent
                prefer_stable = False
                default_max_pools = 7

            max_pools = max_pools or default_max_pools

            # Filter pools
            filtered = [p for p in pools if p.tvl_usd >= min_tvl and p.apy <= max_apy]

            if preferred_chains:
                chains_lower = [c.lower() for c in preferred_chains]
                filtered = [p for p in filtered if p.chain.lower() in chains_lower]

            if prefer_stable:
                stable_pools = [p for p in filtered if p.stable_coin]
                if stable_pools:
                    filtered = stable_pools

            # Calculate gas-adjusted scoring
            # For larger investments, gas costs matter less
            # For high gas, prefer pools with higher absolute yield
            gas_gwei = gas_data.standard_gwei
            gas_cost_per_tx = (gas_gwei * 150000 * gas_data.eth_price_usd) / 1e9

            # Sort by risk-adjusted, gas-aware yield
            for pool in filtered:
                tvl_factor = min(pool.tvl_usd / 1e9, 1)  # Normalize TVL (0-1)

                # Estimate annual yield from this pool
                estimated_yield = investment_amount * (pool.apy / 100)

                # Gas efficiency: higher is better (yield per gas cost)
                gas_efficiency = estimated_yield / gas_cost_per_tx if gas_cost_per_tx > 0 else estimated_yield

                # Risk-adjusted score:
                # - Higher TVL = safer (lower risk)
                # - Higher yield per gas = more efficient
                # - Risk level affects TVL weight
                if risk_level == "low":
                    tvl_weight = 0.7  # Heavily favor TVL for safety
                    yield_weight = 0.3
                elif risk_level == "medium":
                    tvl_weight = 0.5
                    yield_weight = 0.5
                else:  # high
                    tvl_weight = 0.3
                    yield_weight = 0.7  # Favor yield for aggressive

                pool._score = (pool.apy * yield_weight) + (tvl_factor * 10 * tvl_weight)
                # Bonus for gas efficiency on larger investments
                if investment_amount >= 10000:
                    pool._score *= 1 + min(gas_efficiency / 100, 0.5)

            filtered.sort(key=lambda p: p._score, reverse=True)
            selected = filtered[:max_pools]

            if not selected:
                return {"success": False, "error": "No pools match your criteria. Try adjusting filters."}

            # Calculate allocation
            total_score = sum(p._score for p in selected)
            allocations = []

            for pool in selected:
                weight = pool._score / total_score if total_score > 0 else 1 / len(selected)
                amount = investment_amount * weight

                # Estimate gas cost
                gas_cost_usd = (gas_data.standard_gwei * 150000 * gas_data.eth_price_usd) / 1e9

                allocations.append(
                    {
                        "pool_id": pool.pool_id,
                        "protocol": pool.protocol,
                        "chain": pool.chain,
                        "symbol": pool.symbol,
                        "apy": round(pool.apy, 2),
                        "tvl_usd": round(pool.tvl_usd, 2),
                        "allocation_pct": round(weight * 100, 1),
                        "allocation_usd": round(amount, 2),
                        "estimated_annual_yield": round(amount * pool.apy / 100, 2),
                        "estimated_gas_cost": round(gas_cost_usd, 2),
                    }
                )

            total_yield = sum(a["estimated_annual_yield"] for a in allocations)
            total_gas = sum(a["estimated_gas_cost"] for a in allocations)
            weighted_apy = sum(a["apy"] * a["allocation_pct"] / 100 for a in allocations)

            return {
                "success": True,
                "strategy": {
                    "risk_tolerance": risk_tolerance,
                    "investment_amount": investment_amount,
                    "pool_count": len(allocations),
                    "allocations": allocations,
                    "summary": {
                        "weighted_avg_apy": round(weighted_apy, 2),
                        "estimated_annual_yield": round(total_yield, 2),
                        "total_gas_cost": round(total_gas, 2),
                        "net_annual_yield": round(total_yield - total_gas, 2),
                    },
                },
            }
        except Exception as e:
            logger.error(f"Tool calculate_strategy error: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_historical_performance(self, pool_ids: list[str] | None = None, days: int = 30) -> dict:
        """Get historical performance data."""
        try:
            pools = await self.data_service.get_yield_pools()

            if pool_ids:
                pools = [p for p in pools if p.pool_id in pool_ids]

            # Use APY history from data service
            history_data = []
            for pool in pools[:5]:  # Limit for context
                history_data.append(
                    {
                        "pool_id": pool.pool_id,
                        "protocol": pool.protocol,
                        "current_apy": round(pool.apy, 2),
                        "apy_7d_avg": round(pool.apy_mean_7d or pool.apy, 2),
                        "apy_30d_avg": round(pool.apy_mean_30d or pool.apy, 2),
                        "volatility": "low"
                        if abs((pool.apy_mean_7d or pool.apy) - pool.apy) < 2
                        else "medium"
                        if abs((pool.apy_mean_7d or pool.apy) - pool.apy) < 5
                        else "high",
                    }
                )

            return {"success": True, "period_days": days, "pools": history_data}
        except Exception as e:
            logger.error(f"Tool historical_performance error: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_compare_protocols(self, protocols: list[str] | None = None) -> dict:
        """Compare DeFi protocols."""
        try:
            pools = await self.data_service.get_yield_pools()

            # Aggregate by protocol
            protocol_stats: dict[str, dict] = {}

            for pool in pools:
                proto = pool.protocol.lower()
                if protocols and proto not in [p.lower() for p in protocols]:
                    continue

                if proto not in protocol_stats:
                    protocol_stats[proto] = {
                        "name": pool.protocol,
                        "pool_count": 0,
                        "total_tvl": 0,
                        "avg_apy": 0,
                        "max_apy": 0,
                        "chains": set(),
                    }

                stats = protocol_stats[proto]
                stats["pool_count"] += 1
                stats["total_tvl"] += pool.tvl_usd
                stats["avg_apy"] += pool.apy
                stats["max_apy"] = max(stats["max_apy"], pool.apy)
                stats["chains"].add(pool.chain)

            # Finalize
            comparisons = []
            for _proto, stats in protocol_stats.items():
                stats["avg_apy"] = round(stats["avg_apy"] / stats["pool_count"], 2) if stats["pool_count"] > 0 else 0
                stats["total_tvl"] = round(stats["total_tvl"], 2)
                stats["max_apy"] = round(stats["max_apy"], 2)
                stats["chains"] = list(stats["chains"])
                comparisons.append(stats)

            comparisons.sort(key=lambda x: x["total_tvl"], reverse=True)

            return {"success": True, "protocol_count": len(comparisons), "protocols": comparisons[:10]}
        except Exception as e:
            logger.error(f"Tool compare_protocols error: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # CORE AGENT METHODS
    # =========================================================================

    async def _execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        """Execute a tool and record observation."""
        tool = self.tools.get_tool(tool_name)
        if not tool:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = await tool(**tool_args)
            self.memory.add_tool_result(tool_name, result)
            return result
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"error": str(e)}

    async def _create_plan(self, user_request: str) -> ExecutionPlan:
        """Create a multi-step plan for the user request."""
        system_prompt = """You are a DeFi yield optimization planning agent.

Given a user request, create a step-by-step plan to fulfill it.
Available tools: fetch_yield_pools, fetch_gas_prices, analyze_risk, calculate_strategy, get_historical_performance, compare_protocols

Respond with a JSON plan:
{
    "goal": "user's goal",
    "steps": [
        {"step": 1, "action": "tool_name or 'think' or 'respond'", "description": "what this step does", "args": {}},
        ...
    ]
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a plan for: {user_request}"},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=config.model.available_models[0], messages=messages, temperature=0.3, max_tokens=1000
            )

            content = response.choices[0].message.content
            # Parse JSON from response
            import re

            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                plan_data = json.loads(json_match.group())
                steps = [
                    PlanStep(
                        step_number=s.get("step", i + 1),
                        description=s.get("description", ""),
                        tool_name=s.get("action") if s.get("action") not in ["think", "respond"] else None,
                        tool_args=s.get("args", {}),
                    )
                    for i, s in enumerate(plan_data.get("steps", []))
                ]
                return ExecutionPlan(goal=plan_data.get("goal", user_request), steps=steps)
        except Exception as e:
            logger.error(f"Plan creation error: {e}")

        # Default single-step plan
        return ExecutionPlan(
            goal=user_request, steps=[PlanStep(step_number=1, description="Process request directly", tool_name=None)]
        )

    async def _react_loop(self, user_request: str, max_iterations: int = 5) -> str:
        """Execute ReAct loop: Thought → Action → Observation."""
        self.react_trace = []
        observations = []

        system_prompt = f"""You are a DeFi yield optimization agent using the ReAct pattern.

Available tools:
{json.dumps(self.tools.get_all_schemas(), indent=2)}

For each step, respond in this format:
Thought: [your reasoning about what to do next]
Action: [tool_name] with args [json args] OR "final_answer" if done
---

User preferences: {self.memory.get_context_summary()}
"""

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_request}]

        for _iteration in range(max_iterations):
            # Get agent's thought and action
            response = await self.client.chat.completions.create(
                model=config.model.available_models[0], messages=messages, temperature=0.4, max_tokens=1500
            )

            agent_response = response.choices[0].message.content
            messages.append({"role": "assistant", "content": agent_response})

            # Parse thought and action
            thought_match = (
                agent_response.split("Thought:")[-1].split("Action:")[0].strip() if "Thought:" in agent_response else ""
            )

            self.react_trace.append(ReActTrace(step_type=ReActStep.THOUGHT, content=thought_match))

            # Check for final answer
            if "final_answer" in agent_response.lower():
                self.react_trace.append(ReActTrace(step_type=ReActStep.FINAL_ANSWER, content=agent_response))
                break

            # Parse action
            action_match = None
            if "Action:" in agent_response:
                action_part = agent_response.split("Action:")[-1].strip()

                # Try to extract tool name and args
                for tool_name in self.tools.list_tools():
                    if tool_name in action_part:
                        action_match = tool_name
                        # Extract args
                        try:
                            args_match = re.search(r"\{[\s\S]*?\}", action_part)
                            args = json.loads(args_match.group()) if args_match else {}
                        except (json.JSONDecodeError, AttributeError):
                            args = {}
                        break

            if action_match:
                self.react_trace.append(
                    ReActTrace(
                        step_type=ReActStep.ACTION,
                        content=f"Executing {action_match}",
                        tool_name=action_match,
                        tool_args=args,
                    )
                )

                # Execute tool
                result = await self._execute_tool(action_match, args)
                observations.append(result)

                self.react_trace.append(
                    ReActTrace(
                        step_type=ReActStep.OBSERVATION,
                        content=json.dumps(result, indent=2)[:2000],  # Truncate for context
                    )
                )

                # Add observation to messages
                messages.append({"role": "user", "content": f"Observation: {json.dumps(result, indent=2)[:2000]}"})
            else:
                # No action found, likely final response
                break

        # Generate final response with all observations
        return await self._generate_final_response(user_request, observations)

    async def _generate_final_response(self, user_request: str, observations: list) -> str:
        """Generate final user-friendly response from observations."""
        system_prompt = """You are a helpful DeFi yield optimization assistant.
Based on the data gathered, provide a clear, actionable response to the user.
Include specific recommendations with numbers when available.
Be concise but thorough."""

        observation_summary = "\n\n".join(
            [f"Data {i + 1}: {json.dumps(obs, indent=2)[:1500]}" for i, obs in enumerate(observations)]
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""User request: {user_request}

Gathered data:
{observation_summary}

Provide a helpful response addressing the user's request.""",
            },
        ]

        response = await self.client.chat.completions.create(
            model=config.model.available_models[0], messages=messages, temperature=0.5, max_tokens=2000
        )

        return response.choices[0].message.content

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def process_request(self, user_request: str, use_planning: bool = True) -> dict:
        """
        Main entry point - process a user request using agentic capabilities.

        Args:
            user_request: Natural language request from user
            use_planning: Whether to create explicit plan first

        Returns:
            Dict with response, trace, and metadata
        """
        self.memory.add_message("user", user_request)

        try:
            if use_planning:
                # Create plan
                self.current_plan = await self._create_plan(user_request)
                logger.info(f"Created plan with {len(self.current_plan.steps)} steps")

            # Execute using ReAct loop
            response = await self._react_loop(user_request)

            self.memory.add_message("assistant", response)

            # Store recommendation if it was a strategy request
            if "strategy" in user_request.lower() or "recommend" in user_request.lower():
                self.memory.add_recommendation(
                    {
                        "request": user_request,
                        "response_preview": response[:500],
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            return {
                "success": True,
                "response": response,
                "plan": {
                    "goal": self.current_plan.goal if self.current_plan else None,
                    "steps": [
                        {"step": s.step_number, "description": s.description, "tool": s.tool_name}
                        for s in (self.current_plan.steps if self.current_plan else [])
                    ],
                },
                "react_trace": [
                    {"type": t.step_type.value, "content": t.content[:500], "tool": t.tool_name}
                    for t in self.react_trace
                ],
                "tools_used": list({t.tool_name for t in self.react_trace if t.tool_name}),
            }
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error processing your request: {e!s}",
            }

    async def get_quick_strategy(
        self,
        investment_amount: float,
        risk_tolerance: str = "moderate",
        preferred_chains: list[str] | None = None,
    ) -> dict:
        """
        Quick strategy generation - direct tool invocation without full ReAct loop.
        Used for simple, well-defined requests.
        """
        # Store preferences
        self.memory.set_preference("risk_tolerance", risk_tolerance)
        self.memory.set_preference("investment_amount", investment_amount)
        if preferred_chains:
            self.memory.set_preference("preferred_chains", preferred_chains)

        result = await self._tool_calculate_strategy(
            investment_amount=investment_amount, risk_tolerance=risk_tolerance, preferred_chains=preferred_chains
        )

        if result.get("success"):
            self.memory.add_recommendation(result["strategy"])

        return result

    def get_available_tools(self) -> list[dict]:
        """Get list of available tools and their descriptions."""
        return self.tools.get_all_schemas()

    def get_memory_summary(self) -> dict:
        """Get summary of agent's memory state."""
        return {
            "conversation_length": len(self.memory._conversation),
            "preferences": self.memory._preferences,
            "recommendation_count": len(self.memory._recommendations),
            "observation_count": len(self.memory._observations),
        }

    def clear_memory(self):
        """Clear agent memory for fresh start."""
        self.memory.clear()
        self.react_trace = []
        self.current_plan = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_agent_instance: YieldOptimizationAgent | None = None


def get_agent() -> YieldOptimizationAgent:
    """Get or create singleton agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = YieldOptimizationAgent()
    return _agent_instance


async def quick_strategy(
    investment_amount: float,
    risk_tolerance: str = "moderate",
    preferred_chains: list[str] | None = None,
) -> dict:
    """Convenience function for quick strategy generation."""
    agent = get_agent()
    return await agent.get_quick_strategy(investment_amount, risk_tolerance, preferred_chains)


async def chat(message: str) -> dict:
    """Convenience function for conversational interaction."""
    agent = get_agent()
    return await agent.process_request(message)
