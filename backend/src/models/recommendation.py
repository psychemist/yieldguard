from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class RiskProfile(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RecommendationRequest(BaseModel):
    wallet_address: str | None = None
    capital: float
    risk_profile: RiskProfile = RiskProfile.MEDIUM
    preferred_assets: list[str] | None = None


class AllocationItem(BaseModel):
    asset: str
    pool_id: str | None = None
    percentage: float
    expected_yield: float
    risk_score: float


class RecommendationResponse(BaseModel):
    timestamp: datetime
    capital: float
    risk_profile: RiskProfile
    allocations: list[AllocationItem]
    total_expected_yield: float
    total_risk_score: float
    gas_cost_estimate: float
    confidence_score: float


class YieldData(BaseModel):
    protocol: str
    asset: str
    pool_id: str | None = None
    apy: float
    tvl: float
    timestamp: datetime


class GasData(BaseModel):
    slow: float
    standard: float
    fast: float
    timestamp: datetime
