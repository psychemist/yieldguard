"""
YieldGuard Lite API
Production-ready FastAPI backend for DeFi yield optimization.
"""

from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()


from src.models.chat import ChatRequest, ChatResponse
from src.models.recommendation import RecommendationRequest, RecommendationResponse, YieldData
from src.services.agent import get_agent
from src.services.analyzer import yield_analyzer
from src.services.data_service import data_service
from src.services.model_runner import ModelRunner
from src.utils.config import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management - cleanup on shutdown."""
    yield
    await data_service.close()


app = FastAPI(
    title="YieldGuard Lite API", description="AI-powered DeFi yield optimization", version="2.0.0", lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_runner = ModelRunner()
agent = get_agent()


@app.get("/")
async def root():
    return {"name": "YieldGuard Lite API", "version": "2.0.0", "status": "operational"}


@app.get("/health")
async def health_check():
    """Health check with data source status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {"data_service": "available", "model_runner": "available"},
    }


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get AI-powered yield optimization recommendations.
    Uses live data from DeFiLlama, CoinGecko, and gas oracles.
    """
    try:
        print(f"📊 Recommendation request: ${request.capital:,.0f}, {request.risk_profile}")

        # Fetch live market data
        pools = await data_service.get_yield_pools()
        gas = await data_service.get_gas_data()

        if not pools:
            raise HTTPException(status_code=503, detail="No yield data available - DeFiLlama API may be down")

        # Convert to format expected by model runner
        yield_data = [
            YieldData(
                protocol=p.protocol,
                asset=p.symbol,
                pool_id=p.pool_id,
                apy=p.apy,
                tvl=p.tvl_usd,
                timestamp=datetime.now(),
            )
            for p in pools
        ]

        gas_data = gas.to_dict() if gas else {}

        # Generate recommendation
        recommendation = await model_runner.generate_recommendation(
            capital=request.capital, risk_profile=request.risk_profile, yield_data=yield_data, gas_data=gas_data
        )

        print(
            f"✅ Recommendation: {recommendation.total_expected_yield:.1f}% yield, {recommendation.confidence_score:.0%} confidence"
        )
        return recommendation

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/yields")
async def get_yields():
    """Get current yield pools."""
    try:
        pools = await data_service.get_yield_pools()
        return {"timestamp": datetime.now().isoformat(), "count": len(pools), "pools": [p.to_dict() for p in pools]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/yields/historical")
async def get_historical_yields(pool_id: str | None = None, pool_ids: str | None = None, days: int = 30):
    """
    Get historical yield data for charting.
    If pool_id is provided, returns single pool history.
    If pool_ids (comma-separated) is provided, returns history for those pools.
    Otherwise returns top 5 pool histories.
    """
    try:
        if pool_id:
            ts = await data_service.get_yield_history(pool_id, days)
            trend = data_service.compute_trend(ts)
            return {"pool_id": pool_id, "data": ts.to_dict(), "trend": trend}

        # Determine which pools to fetch
        target_pools = []
        if pool_ids:
            # Fetch specific pools requested by frontend
            requested_ids = [pid.strip() for pid in pool_ids.split(",") if pid.strip()]
            all_pools = await data_service.get_yield_pools()
            target_pools = [p for p in all_pools if p.pool_id in requested_ids]
        else:
            # Default to top 5 pools
            pools = await data_service.get_yield_pools()
            target_pools = pools[:5]

        # Collect all timestamps and yields
        all_timestamps: set[str] = set()
        pool_data: dict[str, dict[str, float]] = {}  # display key -> {timestamp: value}

        for pool in target_pools:
            if pool.pool_id:
                ts = await data_service.get_yield_history(pool.pool_id, days)
                if ts.values and ts.timestamps:
                    pool_key = f"{pool.protocol} ({pool.symbol})"
                    pool_data[pool_key] = dict(zip(ts.timestamps, ts.values, strict=False))
                    all_timestamps.update(ts.timestamps)

        # Sort timestamps and build aligned response
        sorted_dates = sorted(all_timestamps)

        # Build yields dict: { symbol: [values aligned to sorted_dates] }
        yields: dict[str, list[float]] = {}
        for symbol, data in pool_data.items():
            yields[symbol] = [data.get(date, 0) for date in sorted_dates]

        return {
            "timestamp": datetime.now().isoformat(),
            "days": days,
            "dates": sorted_dates,
            "yields": yields,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gas")
async def get_gas():
    """Get current gas prices with USD estimates."""
    try:
        gas = await data_service.get_gas_data()
        if not gas:
            raise HTTPException(status_code=503, detail="Gas data unavailable")

        return {
            **gas.to_dict(),
            "estimates_usd": {
                "swap": gas.estimate_cost_usd(config.gas.swap_gas_limit),
                "deposit": gas.estimate_cost_usd(config.gas.deposit_gas_limit),
                "withdraw": gas.estimate_cost_usd(config.gas.withdraw_gas_limit),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market")
async def get_market_snapshot():
    """Get complete market snapshot for dashboard."""
    try:
        snapshot = await data_service.get_market_snapshot()
        return snapshot
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis")
async def get_market_analysis():
    """
    Get comprehensive market analysis including:
    - Trend detection for top pools
    - Volatility analysis
    - Market stance recommendation
    """
    try:
        # Get pools and their history
        pools = await data_service.get_yield_pools()
        if not pools:
            raise HTTPException(status_code=503, detail="No yield data available")

        # Get gas data for cost analysis
        gas = await data_service.get_gas_data()

        # Aggregate APY values for analysis
        apy_values = [p.apy for p in pools]
        avg_apy = sum(apy_values) / len(apy_values) if apy_values else 0

        # Run trend analysis on aggregate
        trend = yield_analyzer.analyze_trend(apy_values)

        # Run volatility analysis
        volatility = yield_analyzer.analyze_volatility(apy_values)

        # Calculate gas cost as % of $10k capital (reference)
        gas_cost_pct = 0
        if gas:
            gas_cost = gas.estimate_cost_usd(config.gas.deposit_gas_limit * config.gas.estimated_transactions)
            gas_cost_pct = (gas_cost / 10000) * 100

        # Compute stance
        stance = yield_analyzer.compute_stance(trend, volatility, avg_apy, gas_cost_pct)

        # Pool-level risk analysis for top 10
        pool_risks = []
        for pool in pools[:10]:
            risk_score, factors = yield_analyzer.analyze_pool_risk(pool.apy, pool.tvl_usd, pool.il_risk)
            pool_risks.append(
                {
                    "symbol": pool.symbol,
                    "protocol": pool.protocol,
                    "apy": pool.apy,
                    "tvl_usd": pool.tvl_usd,
                    "risk_score": round(risk_score, 2),
                    "risk_factors": factors,
                }
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "market_overview": {
                "pool_count": len(pools),
                "avg_apy": round(avg_apy, 2),
                "max_apy": max(apy_values) if apy_values else 0,
                "min_apy": min(apy_values) if apy_values else 0,
            },
            "trend": trend.to_dict(),
            "volatility": volatility.to_dict(),
            "stance": stance.to_dict(),
            "top_pools": pool_risks,
            "gas_context": {"current_gwei": gas.standard_gwei if gas else 0, "cost_pct_of_10k": round(gas_cost_pct, 2)},
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat with the AI agent for tailored advice.
    """
    try:
        # If context is provided (e.g. wallet balances), add it to agent memory
        if request.context and "wallet_assets" in request.context:
            agent.memory.add_message("system", f"User wallet assets: {request.context['wallet_assets']}")

        result = await agent.process_request(request.message)

        return ChatResponse(
            response=result["response"],
            metadata={
                "tools_used": result.get("tools_used", []),
                "plan": result.get("plan")
            }
        )
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
