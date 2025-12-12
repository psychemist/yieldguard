'use client';

import { useState, useRef, useEffect } from 'react';
import { Bot, User, Send, Wallet, Loader2 } from 'lucide-react';
import { useAccount, useBalance, useReadContracts } from 'wagmi';
import { formatEther, formatUnits } from 'viem';
import { erc20Abi } from 'viem';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

const COMMON_TOKENS = [
  { symbol: 'USDC', address: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48' as const, decimals: 6 },
  { symbol: 'USDT', address: '0xdAC17F958D2ee523a2206206994597C13D831ec7' as const, decimals: 6 },
  { symbol: 'DAI', address: '0x6B175474E89094C44Da98b954EedeAC495271d0F' as const, decimals: 18 },
  { symbol: 'WBTC', address: '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599' as const, decimals: 8 },
  { symbol: 'WETH', address: '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2' as const, decimals: 18 },
];

export default function ChatInterface() {
  const { address, isConnected } = useAccount();
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: "Hello! I'm your YieldGuard AI assistant. I can help you find the best DeFi yields, analyze risks, or suggest strategies based on your wallet. How can I help you today?",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Wallet Balance Hooks
  const { data: ethBalance } = useBalance({ address });
  
  const { data: tokenBalances } = useReadContracts({
    contracts: COMMON_TOKENS.map((token) => ({
      address: token.address,
      abi: erc20Abi,
      functionName: 'balanceOf',
      args: [address!],
    })),
    query: {
      enabled: isConnected && !!address,
    }
  });

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleSendMessage = async (content: string, context?: unknown) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      role: 'user',
      content,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: content,
          context: context 
        }),
      });

      if (!response.ok) throw new Error('Failed to send message');

      const data = await response.json();
      
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.response,
          timestamp: new Date(),
        },
      ]);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          content: 'Sorry, I encountered an error processing your request.',
          timestamp: new Date(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleWalletScan = () => {
    if (!isConnected || !address) return;

    // Format ETH Balance
    let assetString = `ETH: ${ethBalance ? parseFloat(formatEther(ethBalance.value)).toFixed(4) : '0'}\n`;

    // Format Token Balances
    if (tokenBalances) {
      tokenBalances.forEach((result, index) => {
        if (result.status === 'success' && result.result) {
          const token = COMMON_TOKENS[index];
          const amount = parseFloat(formatUnits(result.result as bigint, token.decimals));
          if (amount > 0) {
            assetString += `${token.symbol}: ${amount.toFixed(4)}\n`;
          }
        }
      });
    }

    const message = `I've scanned my wallet (${address.slice(0, 6)}...${address.slice(-4)}). Here are my assets:\n\n${assetString}\nBased on these holdings, what yield strategies do you recommend?`;
    
    // Send with context
    handleSendMessage(message, { 
      wallet_assets: assetString,
      wallet_address: address
    });
  };

  return (
    <Card className="h-[600px] flex flex-col">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bot className="h-5 w-5 text-blue-600" />
          AI Advisor
        </CardTitle>
        <CardDescription>
          Ask about yields, risks, or get tailored advice for your wallet
        </CardDescription>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col gap-4 overflow-hidden">
        <div className="flex-1 overflow-y-auto pr-4 space-y-4">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`flex gap-3 ${ 
                  msg.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {msg.role !== 'user' && (
                  <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center shrink-0">
                    <Bot className="h-5 w-5 text-blue-600" />
                  </div>
                )}
                <div
                  className={`rounded-lg p-3 max-w-[80%] ${ 
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : msg.role === 'system'
                      ? 'bg-red-50 text-red-600 border border-red-200'
                      : 'bg-slate-100 dark:bg-slate-800'
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                </div>
                {msg.role === 'user' && (
                  <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center shrink-0">
                    <User className="h-5 w-5 text-slate-600" />
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                  <Bot className="h-5 w-5 text-blue-600" />
                </div>
                <div className="bg-slate-100 dark:bg-slate-800 rounded-lg p-3">
                  <Loader2 className="h-4 w-4 animate-spin text-slate-500" />
                </div>
              </div>
            )}
            <div ref={scrollRef} />
        </div>

        {/* Suggested Actions */}
        {isConnected && messages.length < 3 && (
          <div className="flex gap-2 justify-center">
            <Button 
              variant="outline" 
              size="sm" 
              className="bg-purple-50 hover:bg-purple-100 text-purple-700 border-purple-200"
              onClick={handleWalletScan}
              disabled={loading}
            >
              <Wallet className="w-3 h-3 mr-2" />
              Scan My Wallet & Advise
            </Button>
          </div>
        )}

        <div className="flex gap-2 pt-2 border-t">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage(inputValue)}
            placeholder="Ask anything about DeFi yields..."
            disabled={loading}
          />
          <Button 
            onClick={() => handleSendMessage(inputValue)} 
            disabled={loading || !inputValue.trim()}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
