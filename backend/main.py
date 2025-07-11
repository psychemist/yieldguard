from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import os
from datetime import datetime, date

# Import our services
from src.services.yield_fetcher import YieldFetcher
from src.services.gas_fetcher import GasFetcher
from src.services.model_runner import ModelRunner
from src.models.recommendation import RecommendationRequest, RecommendationResponse

app = FastAPI(title="YieldGuard Lite API", version="1.0.0")

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
yield_fetcher = YieldFetcher()
gas_fetcher = GasFetcher()
model_runner = ModelRunner()

@app.get("/")
async def root():
    return {"message": "YieldGuard Lite API - DeFi Yield Optimization MVP"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get AI-powered yield optimization recommendations
    """
    try:
        # Fetch current data
        yield_data = await yield_fetcher.get_current_yields()
        gas_price = await gas_fetcher.get_current_gas_price()
        
        # Generate recommendations using AI model
        recommendation = await model_runner.generate_recommendation(
            capital=request.capital,
            risk_profile=request.risk_profile,
            yield_data=yield_data,
            gas_price=gas_price
        )
        
        return recommendation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/yields/historical")
async def get_historical_yields(days: int = 30):
    """
    Get historical yield data for the dashboard
    """
    try:
        data = await yield_fetcher.get_historical_yields(days)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gas/current")
async def get_current_gas():
    """
    Get current gas prices
    """
    try:
        gas_data = await gas_fetcher.get_current_gas_price()
        return gas_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/protocols")
async def get_supported_protocols():
    """
    Get list of supported protocols for the MVP
    """
    return {
        "protocols": [
            {
                "id": "uniswap-v3",
                "name": "Uniswap V3",
                "type": "AMM",
                "description": "Decentralized exchange with concentrated liquidity"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)