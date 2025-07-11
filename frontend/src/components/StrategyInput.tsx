'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Loader2, Calculator, TrendingUp, AlertTriangle } from 'lucide-react';

interface Props {
  onRecommendation: (recommendation: any) => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
  isConnected: boolean;
}

export default function StrategyInput({ onRecommendation, loading, setLoading, isConnected }: Props) {
  const [capital, setCapital] = useState('1000');
  const [riskProfile, setRiskProfile] = useState('medium');
  const [gasPreference, setGasPreference] = useState([25]); // Standard gas
  const [error, setError] = useState('');

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

      const recommendation = await response.json();
      onRecommendation(recommendation);
    } catch (err) {
      setError('Unable to get recommendation. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
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