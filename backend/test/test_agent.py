#!/usr/bin/env python3
"""
YieldGuard Lite - Agentic System Test Suite
Tests for Tool System, Planning Loop, ReAct Pattern, and Memory System.
"""

import asyncio

import pytest
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def agent():
    """Create a YieldOptimizationAgent instance for testing."""
    from src.services.agent import YieldOptimizationAgent

    return YieldOptimizationAgent()


# ============================================================================
# Tests
# ============================================================================


def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    from src.utils.config import config

    assert config.api.defillama_yields is not None
    assert len(config.model.available_models) > 0
    assert config.filters.min_tvl_usd > 0

    print(f"  API endpoints: {len(vars(config.api))} configured")
    print(f"  Models available: {len(config.model.available_models)}")
    print(f"  Min TVL filter: ${config.filters.min_tvl_usd:,.0f}")
    print("Config test PASSED")


def test_agent_initialization(agent):
    """Test agent initialization with all 4 agentic features."""
    print("\nTesting agent initialization...")

    # Check core components exist
    assert agent.client is not None, "Groq client not initialized"
    assert agent.data_service is not None, "Data service not initialized"
    assert agent.memory is not None, "Memory system not initialized"
    assert agent.tools is not None, "Tool registry not initialized"

    # Check tools are registered
    tools = agent.tools.list_tools()
    expected_tools = [
        "fetch_yield_pools",
        "fetch_gas_prices",
        "analyze_risk",
        "calculate_strategy",
        "get_historical_performance",
        "compare_protocols",
    ]
    for tool in expected_tools:
        assert tool in tools, f"Tool {tool} not registered"

    print(f"  Agent initialized with {len(tools)} tools")
    print(f"  Tools: {', '.join(tools)}")
    print("Agent initialization test PASSED")


def test_memory_system(agent):
    """Test the memory system."""
    print("\nTesting memory system...")

    # Test conversation memory
    agent.memory.add_message("user", "What are the best yield options?")
    agent.memory.add_message("assistant", "Let me check the current yields...")

    conv = agent.memory.get_conversation_for_llm()
    assert len(conv) == 2, "Conversation not stored"
    print(f"  Conversation history: {len(conv)} messages")

    # Test preference storage
    agent.memory.set_preference("risk_tolerance", "moderate")
    agent.memory.set_preference("investment_amount", 10000)
    assert agent.memory.get_preference("risk_tolerance") == "moderate"
    print("  Preferences stored correctly")

    # Test recommendation memory
    agent.memory.add_recommendation({"strategy": "test", "apy": 5.0})
    summary = agent.memory.get_context_summary()
    assert "Preferences" in summary
    print(f"  Context summary: {summary[:100]}...")

    # Test clear
    agent.memory.clear()
    assert len(agent.memory._conversation) == 0
    print("  Memory cleared successfully")

    print("Memory system test PASSED")


def test_tool_registry(agent):
    """Test the tool registry system."""
    print("\nTesting tool registry...")

    # Get all tool schemas
    schemas = agent.tools.get_all_schemas()
    assert len(schemas) >= 6, "Not enough tools registered"
    print(f"  Tool schemas: {len(schemas)} tools")

    # Check schema format
    for schema in schemas:
        assert "type" in schema
        assert "function" in schema
        assert "name" in schema["function"]
        assert "description" in schema["function"]

    print("  Schema format validated")

    # Test tool retrieval
    tool = agent.tools.get_tool("fetch_yield_pools")
    assert tool is not None, "Tool not found"
    assert callable(tool), "Tool not callable"
    print("  Tool retrieval working")

    print("Tool registry test PASSED")


@pytest.mark.asyncio
async def test_tool_execution(agent):
    """Test direct tool execution."""
    print("\nTesting tool execution...")

    # Test fetch_yield_pools tool
    result = await agent._tool_fetch_yields()
    assert result.get("success"), f"Yield fetch failed: {result.get('error')}"
    print(f"  fetch_yield_pools: {result.get('pool_count', 0)} pools fetched")

    # Test fetch_gas_prices tool
    result = await agent._tool_fetch_gas()
    if result.get("success"):
        print(f"  fetch_gas_prices: standard={result['gas_prices']['standard']} gwei")
    else:
        print(f"  fetch_gas_prices: {result.get('error', 'unavailable')}")

    # Test calculate_strategy tool
    result = await agent._tool_calculate_strategy(investment_amount=10000, risk_tolerance="moderate")
    if result.get("success"):
        strategy = result.get("strategy", {})
        print(
            f"  calculate_strategy: {strategy.get('pool_count', 0)} pools, "
            f"{strategy.get('summary', {}).get('weighted_avg_apy', 0):.2f}% APY"
        )
    else:
        print(f"  calculate_strategy: {result.get('error', 'failed')}")

    print("Tool execution test PASSED")


@pytest.mark.asyncio
async def test_quick_strategy(agent):
    """Test quick strategy generation (direct tool invocation)."""
    print("\nTesting quick strategy...")

    result = await agent.get_quick_strategy(investment_amount=5000, risk_tolerance="conservative")

    if result.get("success"):
        strategy = result.get("strategy", {})
        allocations = strategy.get("allocations", [])
        summary = strategy.get("summary", {})

        print("  Investment: $5,000 (conservative)")
        print(f"  Pools selected: {len(allocations)}")
        print(f"  Weighted APY: {summary.get('weighted_avg_apy', 0):.2f}%")
        print(f"  Est. annual yield: ${summary.get('estimated_annual_yield', 0):.2f}")
        print(f"  Gas cost: ${summary.get('total_gas_cost', 0):.2f}")

        for alloc in allocations[:3]:
            print(f"    - {alloc['symbol']}: {alloc['allocation_pct']}% @ {alloc['apy']:.1f}% APY")

        print("Quick strategy test PASSED")
    else:
        pytest.fail(f"Quick strategy failed: {result.get('error')}")


@pytest.mark.asyncio
async def test_full_agent_request(agent):
    """Test full agentic request with planning and ReAct."""
    print("\nTesting full agent request (ReAct loop)...")

    # Clear memory first
    agent.memory.clear()

    # Process a natural language request
    result = await agent.process_request(
        "I have $10,000 to invest. What are the best yield opportunities for moderate risk?",
        use_planning=True,
    )

    if result.get("success"):
        print("  Response generated successfully")
        print(f"  Plan goal: {result.get('plan', {}).get('goal', 'N/A')}")
        print(f"  Plan steps: {len(result.get('plan', {}).get('steps', []))}")
        print(f"  ReAct trace: {len(result.get('react_trace', []))} steps")
        print(f"  Tools used: {result.get('tools_used', [])}")
        print(f"  Response preview: {result.get('response', '')[:200]}...")
        print("Full agent request test PASSED")
    else:
        pytest.fail(f"Full agent request failed: {result.get('error')}")


# ============================================================================
# Manual Test Runner (for direct execution)
# ============================================================================


def run_tests():
    """Run all tests manually (without pytest)."""
    print("=" * 60)
    print("YieldGuard Lite - Agentic System Test Suite")
    print("Testing: Tool System, Planning Loop, ReAct Pattern, Memory")
    print("=" * 60)

    from src.services.agent import YieldOptimizationAgent

    results = {}

    # Test 1: Config
    try:
        test_config()
        results["config"] = True
    except Exception as e:
        print(f"Config test FAILED: {e}")
        results["config"] = False

    # Create agent for remaining tests
    try:
        agent_instance = YieldOptimizationAgent()
        results["initialization"] = True
    except Exception as e:
        print(f"Agent initialization FAILED: {e}")
        results["initialization"] = False
        agent_instance = None

    if agent_instance:
        # Test 2: Agent initialization
        try:
            test_agent_initialization(agent_instance)
            results["agent_init_full"] = True
        except Exception as e:
            print(f"Agent initialization test FAILED: {e}")
            results["agent_init_full"] = False

        # Test 3: Memory system
        try:
            test_memory_system(agent_instance)
            results["memory_system"] = True
        except Exception as e:
            print(f"Memory system test FAILED: {e}")
            results["memory_system"] = False

        # Reinitialize agent after memory test
        agent_instance = YieldOptimizationAgent()

        # Test 4: Tool registry
        try:
            test_tool_registry(agent_instance)
            results["tool_registry"] = True
        except Exception as e:
            print(f"Tool registry test FAILED: {e}")
            results["tool_registry"] = False

        # Test 5: Tool execution (async)
        try:
            asyncio.run(test_tool_execution(agent_instance))
            results["tool_execution"] = True
        except Exception as e:
            print(f"Tool execution test FAILED: {e}")
            results["tool_execution"] = False

        # Test 6: Quick strategy (async)
        try:
            asyncio.run(test_quick_strategy(agent_instance))
            results["quick_strategy"] = True
        except Exception as e:
            print(f"Quick strategy test FAILED: {e}")
            results["quick_strategy"] = False

        # Test 7: Full agent request with ReAct (async)
        agent_instance = YieldOptimizationAgent()
        try:
            asyncio.run(test_full_agent_request(agent_instance))
            results["full_agent_request"] = True
        except Exception as e:
            print(f"Full agent request test FAILED: {e}")
            results["full_agent_request"] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All tests PASSED! Agentic system is ready for use.")
    else:
        print("⚠️  Some tests FAILED. Please review errors above.")

    return all_passed


if __name__ == "__main__":
    run_tests()
