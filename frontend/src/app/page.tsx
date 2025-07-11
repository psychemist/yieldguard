'use client';

import { useState, useEffect } from 'react';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import { useAccount } from 'wagmi';
import YieldDashboard from '@/components/YieldDashboard';
import StrategyInput from '@/components/StrategyInput';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { TrendingUp, Shield, Zap } from 'lucide-react';

export default function Home() {
  const { address, isConnected } = useAccount();
  const [recommendation, setRecommendation] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold text-slate-900 dark:text-slate-100">
              YieldGuard Lite
            </h1>
            <p className="text-slate-600 dark:text-slate-400 mt-2">
              AI-Powered DeFi Yield Optimization MVP
            </p>
          </div>
          <ConnectButton />
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
                AI-powered allocation across Uniswap V3 pools
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
                Customizable risk profiles with real-time assessment
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
        <Tabs defaultValue="dashboard" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
            <TabsTrigger value="strategy">Strategy Builder</TabsTrigger>
          </TabsList>
          
          <TabsContent value="dashboard">
            <YieldDashboard 
              recommendation={recommendation}
              isConnected={isConnected}
              walletAddress={address}
            />
          </TabsContent>
          
          <TabsContent value="strategy">
            <StrategyInput 
              onRecommendation={setRecommendation}
              loading={loading}
              setLoading={setLoading}
              isConnected={isConnected}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
