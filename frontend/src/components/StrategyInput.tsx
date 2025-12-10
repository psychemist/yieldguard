'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Loader2, Calculator, TrendingUp, AlertTriangle, CheckCircle, Eye, Info } from 'lucide-react';

interface AllocationItem {
  asset: string;
  pool_id: string;
  percentage: number;
  expected_yield: number;
  risk_score: number;
}

interface Recommendation {
  timestamp: string;
  capital: number;
  risk_profile: string;
  allocations: AllocationItem[];
  total_expected_yield: number;
  total_risk_score: number;
  gas_cost_estimate: number;
  confidence_score: number;
}

interface Props {
  onRecommendation: (recommendation: Recommendation) => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
  isConnected: boolean;
  onViewDashboard: () => void; // New prop for navigation
}

export default function StrategyInput({ onRecommendation, loading, setLoading, isConnected, onViewDashboard }: Props) {
  const [capital, setCapital] = useState('1000');
  const [riskProfile, setRiskProfile] = useState('medium');
  const [gasPreference, setGasPreference] = useState([25]);
  const [error, setError] = useState('');
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [showSuccessModal, setShowSuccessModal] = useState(false);

  // Helper function to make asset names user-friendly
  const getAssetDisplayName = (asset: string): { name: string; description: string } => {
    const assetMap: Record<string, { name: string; description: string }> = {
      'STETH': { 
        name: 'Staked Ethereum (Lido)', 
        description: 'Earn staking rewards on your ETH while keeping it liquid' 
      },
      'WSTETH': { 
        name: 'Wrapped Staked ETH', 
        description: 'Lido\'s rebasing-free version of staked ETH' 
      },
      'WEETH': { 
        name: 'Wrapped eETH', 
        description: 'Ether.fi\'s liquid staking token for Ethereum' 
      },
      'WBETH': { 
        name: 'Wrapped Beacon ETH', 
        description: 'Binance\'s liquid staking derivative for ETH' 
      },
      'SUSDE': { 
        name: 'Staked USDe', 
        description: 'Ethena\'s synthetic dollar with staking rewards' 
      },
      'SUSDS': { 
        name: 'Staked USDS', 
        description: 'Ethena\'s staked synthetic dollar token' 
      },
      'USDC': { 
        name: 'USD Coin', 
        description: 'Circle\'s fully-backed USD stablecoin' 
      },
      'USDT': { 
        name: 'Tether USD', 
        description: 'Tether\'s USD-pegged stablecoin' 
      },
      'EZETH': { 
        name: 'Renzo Restaked ETH', 
        description: 'Liquid restaking token through Renzo protocol' 
      }
    };
    
    return assetMap[asset.toUpperCase()] || { 
      name: asset, 
      description: 'DeFi yield-bearing asset' 
    };
  };

  const handleGetRecommendation = async () => {
    if (!capital || parseFloat(capital) <= 0) {
      setError('Please enter a valid capital amount');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/recommendations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          capital: parseFloat(capital),
          risk_profile: riskProfile,
          wallet_address: isConnected ? 'connected' : null,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get recommendation');
      }

      const recommendationData = await response.json();
      setRecommendation(recommendationData);
      onRecommendation(recommendationData);
      setShowSuccessModal(true);
    } catch (err) {
      setError('Unable to get recommendation. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleViewDashboard = () => {
    setShowSuccessModal(false);
    onViewDashboard();
  };

  const getRiskDescription = (profile: string) => {
    switch (profile) {
      case 'low':
        return 'Conservative approach with stable, lower-risk assets';
      case 'medium':
        return 'Balanced approach with moderate risk and returns';
      case 'high':
        return 'Aggressive approach focusing on higher yield opportunities';
      default:
        return '';
    }
  };

  const getGasDescription = (gwei: number) => {
    if (gwei <= 20) return 'Slow but economical';
    if (gwei <= 30) return 'Standard speed and cost';
    return 'Fast execution, higher cost';
  };

  return (
    <div className="space-y-6">
      {/* Success Modal */}
      {showSuccessModal && recommendation && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <CheckCircle className="h-8 w-8 text-green-600" />
                <div>
                  <h2 className="text-2xl font-bold text-green-600">AI Recommendation Complete!</h2>
                  <p className="text-gray-600">Your optimized yield strategy is ready</p>
                </div>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4 p-4 bg-green-50 rounded-lg">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {recommendation.total_expected_yield.toFixed(2)}%
                    </div>
                    <div className="text-sm text-green-700">Expected Annual Yield</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {(recommendation.confidence_score * 100).toFixed(0)}%
                    </div>
                    <div className="text-sm text-blue-700">AI Confidence</div>
                  </div>
                </div>

                <div>
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    <Info className="h-4 w-4" />
                    Recommended Allocation ({recommendation.allocations.length} assets)
                  </h3>
                  <div className="space-y-3">
                    {recommendation.allocations.map((allocation, index) => {
                      const assetInfo = getAssetDisplayName(allocation.asset);
                      return (
                        <div key={index} className="border rounded-lg p-3">
                          <div className="flex justify-between items-start mb-2">
                            <div className="flex-1">
                              <div className="font-medium">{assetInfo.name}</div>
                              <div className="text-sm text-gray-600">{assetInfo.description}</div>
                            </div>
                            <div className="text-right">
                              <div className="font-bold">{allocation.percentage.toFixed(1)}%</div>
                              <div className="text-sm text-green-600">{allocation.expected_yield.toFixed(2)}% APY</div>
                            </div>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-blue-600 h-2 rounded-full" 
                              style={{ width: `${allocation.percentage}%` }}
                            ></div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="flex gap-3">
                  <Button 
                    onClick={handleViewDashboard}
                    className="flex-1"
                    size="lg"
                  >
                    <Eye className="mr-2 h-4 w-4" />
                    View Dashboard
                  </Button>
                  <Button 
                    onClick={() => setShowSuccessModal(false)}
                    variant="destructive"
                    size="lg"
                  >
                    Close
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calculator className="h-5 w-5" />
            Strategy Configuration
          </CardTitle>
          <CardDescription>
            Configure your investment parameters to get AI-powered yield optimization recommendations
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Capital Input */}
          <div className="space-y-2">
            <Label htmlFor="capital">Investment Capital (USD)</Label>
            <Input
              id="capital"
              type="number"
              value={capital}
              onChange={(e) => setCapital(e.target.value)}
              placeholder="Enter amount in USD"
              min="1"
              step="0.01"
            />
            <p className="text-sm text-muted-foreground">
              The amount you want to allocate across DeFi protocols
            </p>
          </div>

          {/* Risk Profile */}
          <div className="space-y-2">
            <Label>Risk Profile</Label>
            <Select value={riskProfile} onValueChange={setRiskProfile}>
              <SelectTrigger>
                <SelectValue placeholder="Select risk profile" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="low">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="bg-green-50 text-green-700">
                      Low Risk
                    </Badge>
                  </div>
                </SelectItem>
                <SelectItem value="medium">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="bg-yellow-50 text-yellow-700">
                      Medium Risk
                    </Badge>
                  </div>
                </SelectItem>
                <SelectItem value="high">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="bg-red-50 text-red-700">
                      High Risk
                    </Badge>
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
            <p className="text-sm text-muted-foreground">
              {getRiskDescription(riskProfile)}
            </p>
          </div>

          {/* Gas Preference */}
          <div className="space-y-2">
            <Label>Gas Price Preference: {gasPreference[0]} gwei</Label>
            <Slider
              value={gasPreference}
              onValueChange={setGasPreference}
              max={50}
              min={15}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-sm text-muted-foreground">
              <span>15 gwei (Slow)</span>
              <span>30 gwei (Standard)</span>
              <span>50 gwei (Fast)</span>
            </div>
            <p className="text-sm text-muted-foreground">
              {getGasDescription(gasPreference[0])}
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-md">
              <AlertTriangle className="h-4 w-4 text-red-600" />
              <span className="text-sm text-red-700">{error}</span>
            </div>
          )}

          {/* Action Button */}
          <Button 
            onClick={handleGetRecommendation} 
            disabled={loading || !capital}
            className="w-full"
            size="lg"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <TrendingUp className="mr-2 h-4 w-4" />
                Get AI Recommendation
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Protocol Info */}
      <Card>
        <CardHeader>
          <CardTitle>Supported Protocol</CardTitle>
          <CardDescription>
            YieldGuard Lite focuses on Uniswap V3 for the MVP
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between p-4 border rounded-lg">
            <div>
              <h4 className="font-semibold">Uniswap V3</h4>
              <p className="text-sm text-muted-foreground">
                Concentrated liquidity AMM on Ethereum
              </p>
            </div>
            <Badge variant="outline" className="bg-blue-50 text-blue-700">
              Active
            </Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}