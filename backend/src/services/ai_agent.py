import asyncio
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
import os
from groq import AsyncGroq
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.schema import AgentAction, AgentFinish
from ..models.recommendation import (
    RecommendationResponse, 
    AllocationItem, 
    YieldData, 
    RiskProfile
)

class YieldOptimizationAgent:
    """
    Advanced AI Agent for DeFi Yield Optimization powered by Groq (Free & Fast)
    - Uses tools to gather market intelligence
    - Reasons through risk/reward scenarios
    - Makes adaptive decisions based on market conditions
    - Learns from previous recommendations
    """
    
    def __init__(self):
        # Initialize Groq LLM - using currently supported model
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",  # Updated to supported model
            temperature=0.1,  # Low temperature for consistent reasoning
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=4096
        )
        
        # Agent's memory and context
        self.market_memory = {}
        self.reasoning_history = []
        
        # Create agent tools
        self.tools = self._create_agent_tools()
        
        # Agent prompt with reasoning capabilities
        self.agent_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.agent_prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def _get_system_prompt(self) -> str:
        return """
        You are YieldGuard AI Agent - an expert DeFi yield optimization agent powered by Groq's lightning-fast Llama 3.1.

        CORE CAPABILITIES:
        1. **Analytical Reasoning**: Break down complex DeFi scenarios step by step
        2. **Risk Assessment**: Evaluate protocol risks, market conditions, and portfolio theory
        3. **Tool Usage**: Use available tools to gather market intelligence and validate decisions
        4. **Adaptive Strategy**: Adjust recommendations based on changing market conditions
        5. **Learning**: Remember insights from previous analyses to improve future recommendations

        REASONING PROCESS:
        1. Analyze the user's investment parameters (capital, risk profile)
        2. Use tools to gather comprehensive market data
        3. Assess each opportunity using multiple risk factors
        4. Apply portfolio optimization principles
        5. Consider gas costs and execution efficiency
        6. Generate and validate the optimal allocation strategy
        7. Provide clear reasoning for each decision

        DECISION FRAMEWORK:
        - Low Risk: Prioritize capital preservation, mature protocols, high TVL
        - Medium Risk: Balance yield and stability, diversification focus
        - High Risk: Optimize for maximum yield, accept higher volatility

        Always think step by step and use tools when you need additional information.
        You are powered by Groq for ultra-fast inference and can handle complex DeFi analysis efficiently.
        """

    def _create_agent_tools(self) -> List[Tool]:
        """Create tools that the agent can use for analysis"""
        
        def analyze_protocol_risk(protocol_name: str) -> str:
            """Analyze the risk profile of a DeFi protocol"""
            try:
                risk_data = {
                    "uniswap-v3": {
                        "maturity_score": 9.5,
                        "security_audits": "Multiple audits by top firms",
                        "tvl_stability": "High",
                        "smart_contract_risk": "Low",
                        "impermanent_loss_risk": "Medium-High"
                    },
                    "aave-v3": {
                        "maturity_score": 9.8,
                        "security_audits": "Extensive audit history",
                        "tvl_stability": "Very High",
                        "smart_contract_risk": "Very Low",
                        "impermanent_loss_risk": "None"
                    },
                    "compound-v3": {
                        "maturity_score": 9.0,
                        "security_audits": "Well audited",
                        "tvl_stability": "High",
                        "smart_contract_risk": "Low",
                        "impermanent_loss_risk": "None"
                    }
                }
                return json.dumps(risk_data.get(protocol_name, {"error": "Protocol not in database"}))
            except Exception as e:
                return json.dumps({"error": f"Error analyzing protocol risk: {str(e)}"})

        def calculate_portfolio_metrics(allocations: str) -> str:
            """Calculate portfolio risk metrics and diversification score"""
            try:
                allocs = json.loads(allocations)
                if not allocs:
                    return json.dumps({"error": "Empty allocations provided"})
                
                total_weight = sum(alloc.get('percentage', 0) for alloc in allocs)
                
                # Calculate Herfindahl index for concentration
                hhi = sum((alloc.get('percentage', 0) / 100) ** 2 for alloc in allocs)
                diversification_score = 1 - hhi
                
                # Risk-weighted portfolio metrics
                weighted_yield = sum(
                    alloc.get('apy', 0) * alloc.get('percentage', 0) / 100 
                    for alloc in allocs
                )
                
                return json.dumps({
                    "diversification_score": round(diversification_score, 3),
                    "concentration_risk": "High" if hhi > 0.5 else "Medium" if hhi > 0.33 else "Low",
                    "weighted_yield": round(weighted_yield, 2),
                    "total_allocation": round(total_weight, 2)
                })
            except Exception as e:
                return json.dumps({"error": f"Invalid allocation format: {str(e)}"})

        def assess_market_conditions(gas_price: float, avg_yield: float) -> str:
            """Assess current market conditions for optimal timing"""
            try:
                conditions = {
                    "gas_assessment": "Favorable" if gas_price < 20 else "Moderate" if gas_price < 40 else "Unfavorable",
                    "yield_environment": "High yield" if avg_yield > 15 else "Moderate yield" if avg_yield > 8 else "Low yield",
                    "market_timing": "Good" if gas_price < 25 and avg_yield > 10 else "Moderate",
                    "execution_recommendation": "Execute immediately" if gas_price < 20 else "Wait for lower gas" if gas_price > 50 else "Execute when ready"
                }
                return json.dumps(conditions)
            except Exception as e:
                return json.dumps({"error": f"Error assessing market conditions: {str(e)}"})

        def validate_allocation_strategy(strategy: str, risk_profile: str) -> str:
            """Validate if the allocation strategy aligns with risk profile"""
            try:
                allocs = json.loads(strategy)
                
                # Check allocation rules based on risk profile
                if not allocs:
                    return json.dumps({"error": "Empty allocation list provided"})
                
                max_single_allocation = max(alloc.get('percentage', 0) for alloc in allocs)
                num_positions = len(allocs)
                
                validation = {
                    "risk_alignment": "Good",
                    "diversification_adequate": num_positions >= (2 if risk_profile == "high" else 3),
                    "concentration_acceptable": max_single_allocation <= (80 if risk_profile == "high" else 60 if risk_profile == "medium" else 50),
                    "recommendations": []
                }
                
                if max_single_allocation > 70:
                    validation["recommendations"].append("Consider reducing concentration in top position")
                
                if num_positions < 3 and risk_profile == "low":
                    validation["recommendations"].append("Add more positions for better diversification")
                
                return json.dumps(validation)
            except Exception as e:
                return json.dumps({"error": f"Invalid strategy format: {str(e)}"})

        return [
            Tool(
                name="analyze_protocol_risk",
                description="Analyze the risk profile of a specific DeFi protocol",
                func=analyze_protocol_risk
            ),
            Tool(
                name="calculate_portfolio_metrics",
                description="Calculate portfolio diversification and risk metrics for a given allocation",
                func=calculate_portfolio_metrics
            ),
            Tool(
                name="assess_market_conditions",
                description="Assess current market conditions based on gas prices and yields",
                func=assess_market_conditions
            ),
            Tool(
                name="validate_allocation_strategy",
                description="Validate if an allocation strategy aligns with the specified risk profile",
                func=validate_allocation_strategy
            )
        ]

    async def analyze_and_recommend(
        self,
        capital: float,
        risk_profile: RiskProfile,
        yield_data: List[YieldData],
        gas_data: Dict
    ) -> RecommendationResponse:
        """
        Use AI agent to reason through market data and generate intelligent recommendations
        """
        try:
            print(f"🤖 AI Agent starting analysis for ${capital:,} with {risk_profile.value} risk profile...")
            
            # Prepare context for the agent
            market_context = self._prepare_agent_context(yield_data, gas_data, capital, risk_profile)
            avg_yield = sum(asset.apy for asset in yield_data) / len(yield_data) if yield_data else 0
            total_tvl = sum(asset.tvl for asset in yield_data) if yield_data else 0
            len_yield = len(yield_data)

            print("checkpoint 0")
            
            # Agent reasoning prompt
            agent_input = f"""
            I need to optimize a DeFi yield strategy with the following parameters:
            
            Investment Capital: ${capital:,}
            Risk Profile: {risk_profile.value}
            Current Gas Price: {gas_data.get('standard', 25)} gwei
            
            Available Opportunities:
            {json.dumps(market_context['opportunities'], indent=2)}

            Market Summary:
            - {len_yield} yield opportunities available
            - Average yield: {avg_yield}%
            - Total market TVL: ${total_tvl}
            
            Please analyze this data step by step and provide an optimal allocation strategy:
            
            1. First, assess the current market conditions using available tools
            2. Analyze the risk profile of the top protocols
            3. Create an optimal allocation strategy based on the risk profile
            4. Validate the strategy using portfolio metrics
            5. Provide a final recommendation with clear reasoning
            
            Return your final recommendation as a JSON object with:
            - allocations: [{{asset, percentage, reasoning, expected_yield, risk_score}}]
            - total_expected_yield: number
            - portfolio_risk_score: number
            - confidence_level: number
            - gas_cost_estimate: number
            - reasoning_summary: string
            """

            # Execute the agent
            result = await self.agent_executor.ainvoke({"input": agent_input})
            print("checkpoint 1")

            # Parse agent response
            recommendation = self._parse_agent_response(result, capital, risk_profile)
            print("checkpoint 2")

            # Store reasoning for learning
            self.reasoning_history.append({
                "timestamp": datetime.now(),
                "input_params": {"capital": capital, "risk_profile": risk_profile.value},
                "agent_reasoning": result.get("output", ""),
                "recommendation": recommendation.model_dump()
            })
            print("checkpoint 3")

            print(f"🧠 AI Agent completed analysis: {recommendation.total_expected_yield:.2f}% yield, {recommendation.confidence_score:.0%} confidence")
            
            return recommendation
            
        except Exception as e:
            print(f"❌ AI Agent error: {e}")
            # Fallback to simple allocation if agent fails
            return self._generate_fallback_recommendation(capital, risk_profile, yield_data, gas_data)

    def _prepare_agent_context(
        self, 
        yield_data: List[YieldData], 
        gas_data: Dict, 
        capital: float, 
        risk_profile: RiskProfile
    ) -> Dict:
        """Prepare structured context for the agent"""
        opportunities = []
        for asset in yield_data:
            opportunities.append({
                "asset": asset.asset,
                "protocol": asset.protocol,
                "apy": asset.apy,
                "tvl_usd": asset.tvl,
                "yield_category": "High" if asset.apy > 20 else "Medium" if asset.apy > 10 else "Conservative"
            })
        
        return {
            "opportunities": opportunities,
            "market_conditions": {
                "gas_price": gas_data.get('standard', 25),
                "total_opportunities": len(yield_data),
                "capital_efficiency": capital / gas_data.get('standard', 25)
            }
        }

    def _parse_agent_response(
        self, 
        agent_result: Dict, 
        capital: float, 
        risk_profile: RiskProfile
    ) -> RecommendationResponse:
        """Parse the agent's reasoning output into a recommendation"""
        try:
            # Extract JSON from agent output
            output = agent_result.get("output", "")
            
            # Try to find JSON in the output
            import re
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                recommendation_data = json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in agent output")
            
            # Parse allocations
            allocations = []
            for alloc in recommendation_data.get("allocations", []):
                allocations.append(AllocationItem(
                    asset=alloc["asset"],
                    percentage=float(alloc["percentage"]),
                    expected_yield=float(alloc["expected_yield"]),
                    risk_score=float(alloc.get("risk_score", 0.5))
                ))
            
            return RecommendationResponse(
                timestamp=datetime.now(),
                capital=capital,
                risk_profile=risk_profile,
                allocations=allocations,
                total_expected_yield=float(recommendation_data.get("total_expected_yield", 0)),
                total_risk_score=float(recommendation_data.get("portfolio_risk_score", 0.5)),
                gas_cost_estimate=float(recommendation_data.get("gas_cost_estimate", 100)),
                confidence_score=float(recommendation_data.get("confidence_level", 0.8))
            )
            
        except Exception as e:
            print(f"Error parsing agent response: {e}")
            raise e

    def _generate_fallback_recommendation(
        self, 
        capital: float, 
        risk_profile: RiskProfile,
        yield_data: List[YieldData],
        gas_data: Dict
    ) -> RecommendationResponse:
        """Fallback recommendation if agent fails"""
        print("🔄 Using fallback recommendation logic...")
        
        # Simple safe allocation
        safe_assets = sorted(yield_data, key=lambda x: x.tvl, reverse=True)[:3]
        
        allocations = []
        percentages = [50.0, 30.0, 20.0]
        
        for i, asset in enumerate(safe_assets):
            allocations.append(AllocationItem(
                asset=asset.asset,
                percentage=percentages[i],
                expected_yield=asset.apy,
                risk_score=0.4
            ))
        
        total_yield = sum(alloc.expected_yield * alloc.percentage / 100 for alloc in allocations)
        
        return RecommendationResponse(
            timestamp=datetime.now(),
            capital=capital,
            risk_profile=risk_profile,
            allocations=allocations,
            total_expected_yield=total_yield,
            total_risk_score=0.4,
            gas_cost_estimate=gas_data.get('standard', 25) * 10,
            confidence_score=0.6
        )

    def _create_simple_recommendation(
        self, 
        capital: float, 
        risk_profile: RiskProfile,
        yield_data: List[YieldData],
        gas_data: Dict
    ) -> RecommendationResponse:
        """Create a smart recommendation using real yield data"""
        print("🔍 Creating smart recommendation with real DeFi data...")
        
        # Use real data if available, otherwise use mock data
        if yield_data and len(yield_data) > 0:
            # Sort by TVL (safety) and APY (yield) for balanced approach
            safe_assets = sorted(yield_data, key=lambda x: (x.tvl * 0.7 + x.apy * 1000000000 * 0.3), reverse=True)[:3]
            print(f"🔍 Selected top 3 assets: {[asset.asset for asset in safe_assets]}")
        else:
            # Mock safe assets as fallback
            mock_assets = [
                type('MockAsset', (), {'asset': 'USDC', 'apy': 5.2, 'tvl': 2000000000})(),
                type('MockAsset', (), {'asset': 'STETH', 'apy': 3.8, 'tvl': 1500000000})(),
                type('MockAsset', (), {'asset': 'USDT', 'apy': 4.1, 'tvl': 1200000000})()
            ]
            safe_assets = mock_assets
            print("🔍 Using mock assets as fallback")
        
        # Risk-based allocation percentages
        if risk_profile == RiskProfile.LOW:
            percentages = [60.0, 25.0, 15.0]  # Conservative
        elif risk_profile == RiskProfile.MEDIUM:
            percentages = [50.0, 30.0, 20.0]  # Balanced
        else:  # HIGH
            percentages = [40.0, 35.0, 25.0]  # Aggressive
        
        print(f"🔍 Using {risk_profile.value} risk allocation: {percentages}")
        
        # Create allocations
        allocations = []
        for i, asset in enumerate(safe_assets[:len(percentages)]):
            risk_score = 0.3 if risk_profile == RiskProfile.LOW else 0.5 if risk_profile == RiskProfile.MEDIUM else 0.7
            allocations.append(AllocationItem(
                asset=asset.asset,
                percentage=percentages[i],
                expected_yield=asset.apy,
                risk_score=risk_score
            ))
        
        # Calculate portfolio metrics
        total_yield = sum(alloc.expected_yield * alloc.percentage / 100 for alloc in allocations)
        avg_risk = sum(alloc.risk_score * alloc.percentage / 100 for alloc in allocations)
        
        print(f"🔍 Portfolio calculated - Total yield: {total_yield:.2f}%, Avg risk: {avg_risk:.2f}")
        
        return RecommendationResponse(
            timestamp=datetime.now(),
            capital=capital,
            risk_profile=risk_profile,
            allocations=allocations,
            total_expected_yield=total_yield,
            total_risk_score=avg_risk,
            gas_cost_estimate=gas_data.get('standard', 25) * 12,  # Estimated gas cost
            confidence_score=0.85  # High confidence for data-driven recommendations
        )