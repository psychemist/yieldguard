'use client';

import { useState } from 'react';
import { useAppKit } from '@reown/appkit/react';
import { useAccount } from 'wagmi';
import { ThemeToggle } from '@/components/theme-toggle';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { TrendingUp, Shield, Zap, Wallet } from 'lucide-react';
import YieldDashboard from '@/components/YieldDashboard';
import StrategyInput from '@/components/StrategyInput';
import ChatInterface from '@/components/ChatInterface';

interface Recommendation {
  timestamp: string;
  capital: number;
  risk_profile: string;
  allocations: Array<{
    asset: string;
    pool_id: string;
    percentage: number;
    expected_yield: number;
    risk_score: number;
  }>;
  total_expected_yield: number;
  total_risk_score: number;
  gas_cost_estimate: number;
  confidence_score: number;
}

export default function Home() {
  const { isConnected } = useAccount();
  const { open } = useAppKit();
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');

  // Function to navigate to dashboard after getting recommendation
  const handleViewDashboard = () => {
    setActiveTab('dashboard');
  };

  return (
    <div className="min-h-screen bg-linear-to-b from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold text-slate-900 dark:text-slate-100">
              YieldGuard
            </h1>
            <p className="text-slate-600 dark:text-slate-400 mt-2">
              AI-Powered DeFi Yield Optimization Engine
            </p>
          </div>
          <div className="flex items-center gap-4">
            <ThemeToggle />
            {isConnected ? (
              <w3m-button />
            ) : (
              <Button onClick={() => open()} className="flex items-center gap-2">
                <Wallet className="h-4 w-4" />
                Connect Wallet
              </Button>
            )}
          </div>
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center space-y-0 pb-2">
              <TrendingUp className="h-5 w-5 text-green-600" />
              <CardTitle className="text-sm font-medium ml-2">
                Yield Optimization
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-slate-600 dark:text-slate-400">
                AI-powered allocation across DeFi protocols with real market data
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center space-y-0 pb-2">
              <Shield className="h-5 w-5 text-blue-600" />
              <CardTitle className="text-sm font-medium ml-2">
                Risk Management
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-slate-600 dark:text-slate-400">
                Customizable risk profiles with AI agent analysis
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center space-y-0 pb-2">
              <Zap className="h-5 w-5 text-yellow-600" />
              <CardTitle className="text-sm font-medium ml-2">
                Gas Optimization
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-slate-600 dark:text-slate-400">
                Real-time gas cost estimation and optimization
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Main Dashboard */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
            <TabsTrigger value="strategy">Strategy Builder</TabsTrigger>
            <TabsTrigger value="chat">AI Chat</TabsTrigger>
          </TabsList>
          
          <TabsContent value="dashboard">
            <YieldDashboard 
              recommendation={recommendation}
              onNavigateToStrategy={() => setActiveTab('strategy')}
              onNavigateToChat={() => setActiveTab('chat')}
            />
          </TabsContent>
          
          <TabsContent value="strategy">
            <StrategyInput 
              onRecommendation={setRecommendation}
              loading={loading}
              setLoading={setLoading}
              isConnected={isConnected}
              onViewDashboard={handleViewDashboard}
            />
          </TabsContent>

          <TabsContent value="chat">
            <ChatInterface />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
