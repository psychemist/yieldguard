/**
 * YieldGuard Lite - API Service
 * Centralized API client for backend communication
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface YieldPool {
  protocol: string;
  symbol: string;
  chain: string;
  apy: number;
  tvl_usd: number;
  pool_id: string;
  il_risk: string;
  stable_coin: boolean;
  timestamp: string;
}

export interface GasData {
  slow_gwei: number;
  standard_gwei: number;
  fast_gwei: number;
  instant_gwei: number;
  eth_price_usd: number;
  timestamp: string;
  estimates_usd?: {
    swap: number;
    deposit: number;
    withdraw: number;
  };
}

export interface TrendAnalysis {
  direction: 'rising' | 'stable' | 'falling' | 'unknown';
  slope: number;
  normalized_slope: number;
  strength: number;
  r_squared: number;
}

export interface VolatilityAnalysis {
  bucket: 'low' | 'moderate' | 'high' | 'extreme';
  std_dev: number;
  cv: number;
  max_drawdown: number;
  range_pct: number;
}

export interface MarketStance {
  stance: 'favorable' | 'neutral' | 'caution' | 'wait';
  confidence: number;
  reasoning: string[];
  recommended_stable_pct: number;
}

export interface MarketAnalysis {
  timestamp: string;
  market_overview: {
    pool_count: number;
    avg_apy: number;
    max_apy: number;
    min_apy: number;
  };
  trend: TrendAnalysis;
  volatility: VolatilityAnalysis;
  stance: MarketStance;
  top_pools: Array<{
    symbol: string;
    protocol: string;
    apy: number;
    tvl_usd: number;
    risk_score: number;
    risk_factors: string[];
  }>;
  gas_context: {
    current_gwei: number;
    cost_pct_of_10k: number;
  };
}

export interface AllocationItem {
  asset: string;
  percentage: number;
  expected_yield: number;
  risk_score: number;
}

export interface Recommendation {
  timestamp: string;
  capital: number;
  risk_profile: string;
  allocations: AllocationItem[];
  total_expected_yield: number;
  total_risk_score: number;
  gas_cost_estimate: number;
  confidence_score: number;
}

export interface HistoricalYields {
  pool_id?: string;
  dates: string[];
  yields: Record<string, number[]>;
}

export interface MarketSnapshot {
  timestamp: string;
  total_tvl: number;
  avg_apy: number;
  pool_count: number;
  top_protocols: Array<{ name: string; tvl: number; pool_count: number }>;
}

export interface Protocol {
  name: string;
  slug: string;
  tvl: number;
  pool_count: number;
  chains: string[];
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API Error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  // Health check
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return this.request('/health');
  }

  // Get current yield pools
  async getYieldPools(): Promise<{ timestamp: string; count: number; pools: YieldPool[] }> {
    return this.request('/yields');
  }

  // Get historical yield data
  async getHistoricalYields(poolId?: string, days: number = 30): Promise<HistoricalYields> {
    const params = new URLSearchParams({ days: days.toString() });
    if (poolId) params.append('pool_id', poolId);
    return this.request(`/yields/historical?${params}`);
  }

  // Get current gas data
  async getGasData(): Promise<GasData> {
    return this.request('/gas');
  }

  // Get market analysis
  async getMarketAnalysis(): Promise<MarketAnalysis> {
    return this.request('/analysis');
  }

  // Get market snapshot
  async getMarketSnapshot(): Promise<MarketSnapshot> {
    return this.request('/market');
  }

  // Get AI recommendation
  async getRecommendation(
    capital: number,
    riskProfile: 'low' | 'medium' | 'high',
    walletAddress?: string
  ): Promise<Recommendation> {
    return this.request('/recommendations', {
      method: 'POST',
      body: JSON.stringify({
        capital,
        risk_profile: riskProfile,
        wallet_address: walletAddress,
      }),
    });
  }

  // Get supported protocols
  async getProtocols(): Promise<{ protocols: Protocol[] }> {
    return this.request('/protocols');
  }
}

// Export singleton instance
export const api = new ApiService();

export default ApiService;
