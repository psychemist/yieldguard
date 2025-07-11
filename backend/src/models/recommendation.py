from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

class RiskProfile(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class RecommendationRequest(BaseModel):
    wallet_address: Optional[str] = None
    capital: float
    risk_profile: RiskProfile = RiskProfile.MEDIUM
    preferred_assets: Optional[List[str]] = None

class AllocationItem(BaseModel):
    asset: str
    percentage: float
    expected_yield: float
    risk_score: float

class RecommendationResponse(BaseModel):
    timestamp: datetime
    capital: float
    risk_profile: RiskProfile
    allocations: List[AllocationItem]
    total_expected_yield: float
    total_risk_score: float
    gas_cost_estimate: float
    confidence_score: float

class YieldData(BaseModel):
    protocol: str
    asset: str
    apy: float
    tvl: float
    timestamp: datetime

class GasData(BaseModel):
    slow: float
    standard: float
    fast: float
    timestamp: datetime