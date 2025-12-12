'use client';

import { useState, useEffect } from 'react';
import { TrendingUp, AlertTriangle, CheckCircle, Clock, Info, MessageSquare, Sparkles } from 'lucide-react'; 
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Label } from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';

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

interface GasDataType {
  slow: number;
  standard: number;
  fast: number;
}

interface ChartDataPoint {
  date: string;
  [key: string]: string | number;
}

interface Props {
  recommendation: Recommendation | null;
  onNavigateToStrategy: () => void;
  onNavigateToChat: () => void;
}

export default function YieldDashboard({ recommendation, onNavigateToStrategy, onNavigateToChat }: Props) {
  const [historicalData, setHistoricalData] = useState<ChartDataPoint[]>([]);
  const [gasData, setGasData] = useState<GasDataType | null>(null);
  const [loading, setLoading] = useState(true);
  const [recommendedAssets, setRecommendedAssets] = useState<string[]>([]);

  useEffect(() => {
    fetchGasData();
  }, []);

  // Fetch historical data when recommendation changes
  useEffect(() => {
    if (recommendation?.allocations) {
      const assets = recommendation.allocations.map(a => a.asset);
      const poolIds = recommendation.allocations.map(a => a.pool_id);
      setRecommendedAssets(assets);
      fetchHistoricalData(poolIds, assets);
    } else {
      // Show top pools if no recommendation yet
      fetchHistoricalData();
    }
  }, [recommendation]);

  const fetchHistoricalData = async (poolIds?: string[], assetNames?: string[]) => {
    try {
      let url = `${process.env.NEXT_PUBLIC_API_URL}/yields/historical?days=30`;
      if (poolIds && poolIds.length > 0) {
        url += `&pool_ids=${poolIds.join(',')}`;
      }

      console.log('Fetching REAL historical data from backend...', poolIds ? `for pools: ${poolIds.join(', ')}` : 'for top pools');

      const response = await fetch(url);
      if (!response.ok) {
        console.error(`API error: ${response.status}`);
        setHistoricalData([]);
        return;
      }

      const data = await response.json();
      console.log('Received real historical data:', data);

      // Transform data for chart
      if (data.dates && data.yields) {
        // If we requested specific pools, use backend response keys
        const keysToRender = Object.keys(data.yields);
        
        const chartData: ChartDataPoint[] = data.dates.map((date: string, index: number) => {
          const dataPoint: ChartDataPoint = { date };
          
          keysToRender.forEach(key => {
            if (data.yields[key]) {
              dataPoint[key] = data.yields[key]?.[index] || 0;
            }
          });
          return dataPoint;
        });
        
        setHistoricalData(chartData);
        // If specific assets were recommendeded, keep them for the legend title
        // otherwise default to empty to show "top pools" text
        setRecommendedAssets(assetNames || []);
        console.log(`Successfully loaded ${chartData.length} days of historical data`);
      } else {
        console.warn('No historical data structure found, using empty array');
        setHistoricalData([]);
      }
      
    } catch (error) {
      console.error('Failed to fetch real historical data:', error);
      setHistoricalData([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchGasData = async () => {
    try {
      console.log('Fetching REAL gas data from backend...');
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/gas`);
      if (!response.ok) {
        console.log(`Gas API error: ${response.status}`);
        return;
      }

      const data = await response.json();
      console.log('Received real gas data:', data);

      // Transform API response to match expected format
      setGasData({
        slow: data.slow_gwei,
        standard: data.standard_gwei,
        fast: data.fast_gwei
      });
    } catch (error) {
      console.error('Failed to fetch real gas data:', error);
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

  // Helper function to format protocol names cleanly
  const formatProtocolName = (name: string): string => {
    // Extract protocol and token parts
    const match = name.match(/^([^(]+)\s*\(([^)]+)\)$/);
    if (!match) {
      // No parentheses, just capitalize first letter
      return name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();
    }

    const [, protocolPart, tokenSymbol] = match;
    const tokenUpper = tokenSymbol.toUpperCase();
    
    // Clean up protocol name - remove token suffix if present
    let cleanProtocol = protocolPart.trim();
    
    // Check if protocol ends with a token-like suffix (e.g., "-usde", "-uscc", "-usdt")
    const tokenLower = tokenSymbol.toLowerCase();
    const suffixPattern = new RegExp(`[-_]?${tokenLower}$`, 'i');
    cleanProtocol = cleanProtocol.replace(suffixPattern, '');
    
    // Check if protocol contains any part of the token
    const tokenBase = tokenLower.replace(/[0-9]/g, ''); // Remove numbers from token
    const basePattern = new RegExp(`[-_]?${tokenBase}[a-z]*$`, 'i');
    cleanProtocol = cleanProtocol.replace(basePattern, '');
    
    // Capitalize protocol name (handle dashes)
    cleanProtocol = cleanProtocol.split('-').map(part => 
      part.charAt(0).toUpperCase() + part.slice(1).toLowerCase()
    ).join('-');
    
    // Remove trailing dash if any
    cleanProtocol = cleanProtocol.replace(/-$/, '');
    
    return `${cleanProtocol} (${tokenUpper})`;
  };

  // Create a mapping from pool_id or asset to index for consistent colors
    const getColorIndex = (key: string): number => {
      if (recommendation?.allocations && recommendation.allocations.length > 0) {
        // If recommendation exists, find index in allocations array
        const index = recommendation.allocations.findIndex(alloc =>
          key.toLowerCase().includes(alloc.asset.toLowerCase())
        );
        return index >= 0 ? index : 0;
      } else {
        // If no recommendation, assign colors based on the order of assets in historicalData
        const historicalAssetKeys = Object.keys(historicalData[0] || {}).filter(k => k !== 'date');
        const index = historicalAssetKeys.indexOf(key);
        return index >= 0 ? index : 0;
      }
    };
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
              {recommendation ? `${recommendation.total_expected_yield.toFixed(2)}%` : '--%'}
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
            <div className={`text-2xl font-bold ${getRiskColor(recommendation?.total_risk_score || 0.0)}`}>
              {getRiskLabel(recommendation?.total_risk_score || 0.0)}
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
              ${recommendation ? recommendation.gas_cost_estimate.toFixed(2) : '--'}
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
              {recommendation ? `${(recommendation.confidence_score * 100).toFixed(0)}%` : '--%'}
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
            <CardDescription>
              {recommendedAssets.length > 0 
                ? `30-day APY trends for your AI-recommended assets` 
                : `Past 30 days yield performance from top DeFi protocols`}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                <div className="text-center">
                  <TrendingUp className="h-8 w-8 mx-auto mb-2 opacity-50 animate-pulse" />
                  <p>Loading real historical data...</p>
                </div>
              </div>
            ) : historicalData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={historicalData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="date"
                      stroke="#888888"
                      fontSize={12}
                      tickLine={true}
                      axisLine={true}
                    >
                    <Label value="DATE" position="insideBottom" dy={10} className="fill-foreground" /> 
                    </XAxis>
                    <YAxis
                      tickFormatter={(value) => `${value}%`}
                      stroke="#888888"
                      fontSize={12}
                      tickLine={true}
                      axisLine={true}
                    >
                      <Label value="APY (%)" angle={-90} position="insideLeft" dy={-10} className="fill-foreground" />
                    </YAxis>
                    <Tooltip
                      formatter={(value: number) => [`${value}%`, "APY"]}
                      labelStyle={{ color: "black" }}
                      contentStyle={{ borderRadius: "8px", border: "none", boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)" }}
                    /> 
                    {/* Dynamically render lines for all real assets - colors match pie chart order */}
                    {Object.keys(historicalData[0] || {}).filter(key => key !== 'date').map((assetName) => {
                      const colorIndex = getColorIndex(assetName);
                      return (
                        <Line 
                          key={assetName}
                          type="monotone" 
                          dataKey={assetName} 
                          stroke={pieColors[colorIndex % pieColors.length]} 
                          strokeWidth={2}
                          dot={false}
                        />
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>

                {/* Asset Legend & Info */}
                <div className="mt-4 space-y-2">
                  <div className="flex flex-wrap gap-2">
                    {Object.keys(historicalData[0] || {}).filter(key => key !== 'date').map((assetName) => {
                      const colorIndex = getColorIndex(assetName);
                      return (
                        <div key={assetName} className="flex items-center gap-1 text-xs">
                          <div 
                            className="w-3 h-3 rounded-full" 
                            style={{ backgroundColor: pieColors[colorIndex % pieColors.length] }}
                          />
                          <span className="font-medium text-foreground">{formatProtocolName(assetName)}</span>
                        </div>
                      );
                    })}
                  </div>
                  
                  {recommendedAssets.length > 0 ? (
                    <p className="text-xs text-muted-foreground">
                      📊 <strong>Showing historical performance</strong> for assets recommended by our AI based on your ${recommendation?.capital.toLocaleString()} investment and {recommendation?.risk_profile} risk profile. 
                      This real on-chain data helps you understand yield stability and trends before deploying capital.
                    </p>
                  ) : (
                    <p className="text-xs text-muted-foreground">
                      📊 <strong>Showing top DeFi protocols on Ethereum by TVL.</strong> Get a personalized AI recommendation to see historical trends for assets matched to your risk profile and capital.
                    </p>
                  )}
                </div>
              </>
            ) : (
              <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                <div className="text-center">
                  <AlertTriangle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No historical data available.</p>
                  <p className="text-xs mt-1">Try refreshing or checking back later.</p> 
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Allocation Pie Chart */}
        <Card>
          <CardHeader>
            <CardTitle>AI Recommended Allocation</CardTitle>
            <CardDescription>
              {recommendation ? 
                `Real market analysis for $${recommendation.capital.toLocaleString()} investment` : 
                'Connect and get AI recommendation based on real DeFi data'
              }
            </CardDescription>
          </CardHeader>
          <CardContent>
            {recommendation?.allocations ? (
              <div className="space-y-4">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={recommendation.allocations.map(alloc => ({
                        ...alloc,
                        name: getAssetDisplayName(alloc.asset).name
                      }))}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      // eslint-disable-next-line @typescript-eslint/no-explicit-any
                      label={({ name, percentage }: any) => `${name}: ${percentage}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="percentage"
                    >
                      {recommendation.allocations.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={pieColors[index % pieColors.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>

                {/* Asset Details Modal Trigger */}
                <div className="flex justify-center mt-4">
                  <Dialog>
                    <DialogTrigger asChild>
                      <Button variant="outline" className="w-full">
                        <Info className="mr-2 h-4 w-4" />
                        View Asset Details
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
                      <DialogHeader>
                        <DialogTitle>Portfolio Allocation Details</DialogTitle>
                        <DialogDescription>
                          Detailed breakdown of the recommended assets and their risk/reward profiles.
                        </DialogDescription>
                      </DialogHeader>
                      <div className="space-y-3 mt-4">
                        {recommendation.allocations.map((allocation, index) => {
                          const assetInfo = getAssetDisplayName(allocation.asset);
                            return (
                              <div key={index} className="flex flex-col sm:flex-row sm:items-center p-4 bg-secondary/50 rounded-lg gap-4">
                                {/* Asset Info - fixed width */}
                                <div className="space-y-1 sm:w-1/3 sm:min-w-[200px]">
                                  <div className="flex items-center gap-2">
                                    <div
                                      className="w-3 h-3 rounded-full shrink-0"
                                      style={{ backgroundColor: pieColors[index % pieColors.length] }}
                                    />
                                    <span className="font-semibold text-foreground">
                                      {assetInfo.name} ({allocation.asset})
                                    </span>
                                  </div>
                                  <p className="text-sm text-muted-foreground pl-5">{assetInfo.description}</p>
                                </div>
                                {/* Stats Grid - equal width columns, centered */}
                                <div className="grid grid-cols-4 gap-2 text-sm flex-1">
                                  <div className="text-center">
                                    <div className="text-muted-foreground text-xs">Allocation</div>
                                    <div className="font-medium text-foreground">{allocation.percentage.toFixed(1)}%</div>
                                  </div>
                                  <div className="text-center">
                                    <div className="text-muted-foreground text-xs">Value</div>
                                    <div className="font-medium text-foreground">
                                      ${((recommendation.capital * allocation.percentage) / 100).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                    </div>
                                  </div>
                                  <div className="text-center">
                                    <div className="text-muted-foreground text-xs">Yield</div>
                                    <div className="font-medium text-green-600 dark:text-green-400">{allocation.expected_yield.toFixed(2)}% APY</div>
                                  </div>
                                  <div className="text-center">
                                    <div className="text-muted-foreground text-xs">Risk</div>
                                    <div className={`font-medium ${getRiskColor(allocation.risk_score)}`}>
                                      {getRiskLabel(allocation.risk_score)}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </DialogContent>
                    </Dialog>
                  </div>
                </div>
            ) : (
            <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                <div className="text-center">
                <AlertTriangle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>Get AI recommendation to see optimal allocation</p>
                <div className="flex gap-3 justify-center mt-4">
                  <Button 
                    variant="default" 
                    size="sm" 
                    onClick={onNavigateToStrategy}
                    className="gap-2"
                  >
                    <Sparkles className="h-4 w-4" />
                    Build Strategy
                  </Button>
                  <Button 
                    variant="secondary" 
                    size="sm" 
                    onClick={onNavigateToChat}
                    className="gap-2"
                  >
                    <MessageSquare className="h-4 w-4" />
                    Chat with AI
                  </Button>
                </div>
                </div>
            </div>
            )}
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

  // Helper function to make asset names user-friendly (add this before the return statement)
  function getAssetDisplayName(asset: string): { name: string; description: string } {
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
}