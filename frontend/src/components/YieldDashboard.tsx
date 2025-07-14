'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Clock } from 'lucide-react';

interface AllocationItem {
  asset: string;
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
  recommendation: Recommendation | null;
  isConnected: boolean;
  walletAddress?: string;
}

export default function YieldDashboard({ recommendation, isConnected, walletAddress }: Props) {
  const [historicalData, setHistoricalData] = useState([]);
  const [gasData, setGasData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistoricalData();
    fetchGasData();
  }, []);

  const fetchHistoricalData = async () => {
    try {
      console.log('Fetching REAL historical data from backend...');
      const response = await fetch('http://localhost:8000/yields/historical?days=30');
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Received real historical data:', data);
      
      // Transform REAL data for chart
      const chartData = data.dates?.map((date: string, index: number) => ({
        date,
        ...Object.keys(data.yields || {}).reduce((acc, assetName) => {
          acc[assetName] = data.yields[assetName]?.[index] || 0;
          return acc;
        }, {} as Record<string, number>)
      })) || [];
      
      if (chartData.length === 0) {
        throw new Error('No real data received from API');
      }
      
      setHistoricalData(chartData);
      console.log(`Successfully loaded ${chartData.length} days of real historical data`);
      
    } catch (error) {
      console.error('Failed to fetch real historical data:', error);
      throw new Error(`Real data fetch failed: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchGasData = async () => {
    try {
      console.log('Fetching REAL gas data from backend...');
      const response = await fetch('http://localhost:8000/gas/current');
      
      if (!response.ok) {
        throw new Error(`Gas API error: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Received real gas data:', data);
      
      if (!data.slow || !data.standard || !data.fast) {
        throw new Error('Invalid gas data received');
      }
      
      setGasData(data);
    } catch (error) {
      console.error('Failed to fetch real gas data:', error);
      throw new Error(`Real gas data fetch failed: ${error}`);
    }
  };

  const getRiskColor = (score: number) => {
    if (score <= 0.3) return 'text-green-600';
    if (score <= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getRiskLabel = (score: number) => {
    if (score <= 0.3) return 'Low Risk';
    if (score <= 0.6) return 'Medium Risk';
    return 'High Risk';
  };

  const pieColors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader className="pb-2">
                <div className="h-4 bg-gray-200 rounded w-3/4"></div>
              </CardHeader>
              <CardContent>
                <div className="h-8 bg-gray-200 rounded w-1/2"></div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Expected Yield</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {recommendation ? `${recommendation.total_expected_yield.toFixed(2)}%` : '12.5%'}
            </div>
            <p className="text-xs text-muted-foreground">
              Annual percentage yield
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Risk Score</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${getRiskColor(recommendation?.total_risk_score || 0.4)}`}>
              {getRiskLabel(recommendation?.total_risk_score || 0.4)}
            </div>
            <p className="text-xs text-muted-foreground">
              Overall portfolio risk
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Gas Cost</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${recommendation ? recommendation.gas_cost_estimate.toFixed(2) : '45.00'}
            </div>
            <p className="text-xs text-muted-foreground">
              Estimated transaction cost
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Confidence</CardTitle>
            <CheckCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">
              {recommendation ? `${(recommendation.confidence_score * 100).toFixed(0)}%` : '85%'}
            </div>
            <p className="text-xs text-muted-foreground">
              AI model confidence
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Historical Yields Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Historical Yields</CardTitle>
            <CardDescription>Past 30 days yield performance</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="USDC/ETH" stroke="#3B82F6" strokeWidth={2} />
                <Line type="monotone" dataKey="USDT/ETH" stroke="#10B981" strokeWidth={2} />
                <Line type="monotone" dataKey="WBTC/ETH" stroke="#F59E0B" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Allocation Pie Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Recommended Allocation</CardTitle>
            <CardDescription>
              {recommendation ? `Based on ${recommendation.capital} capital` : 'Sample allocation'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={recommendation?.allocations || [
                    { asset: 'USDC/ETH', percentage: 40 },
                    { asset: 'USDT/ETH', percentage: 35 },
                    { asset: 'WBTC/ETH', percentage: 25 }
                  ]}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ asset, percentage }) => `${asset}: ${percentage}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="percentage"
                >
                  {(recommendation?.allocations || []).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={pieColors[index % pieColors.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Gas Tracker */}
      {gasData && (
        <Card>
          <CardHeader>
            <CardTitle>Gas Tracker</CardTitle>
            <CardDescription>Current Ethereum gas prices</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-lg font-semibold text-green-600">{gasData.slow} gwei</div>
                <div className="text-sm text-muted-foreground">Slow</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold text-yellow-600">{gasData.standard} gwei</div>
                <div className="text-sm text-muted-foreground">Standard</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold text-red-600">{gasData.fast} gwei</div>
                <div className="text-sm text-muted-foreground">Fast</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}