'use client';

import '@rainbow-me/rainbowkit/styles.css';
import { useEffect, useState } from 'react';
import {
  getDefaultConfig,
  RainbowKitProvider,
} from '@rainbow-me/rainbowkit';
import { WagmiProvider } from 'wagmi';
import {
  mainnet,
  polygon,
  optimism,
  arbitrum,
  base,
} from 'wagmi/chains';
import {
  QueryClientProvider,
  QueryClient,
} from '@tanstack/react-query';
import { ThemeProvider } from '@/contexts/theme-context';

const config = getDefaultConfig({
  appName: 'YieldGuard Lite',
  projectId: process.env.NEXT_PUBLIC_REOWN_PROJECT_ID || '', // Use env variable
  chains: [mainnet, polygon, optimism, arbitrum, base],
  ssr: true,
});

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
    },
  },
});

export function Providers({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return (
    <WagmiProvider config={config}>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider
          defaultTheme="system"
          storageKey="yieldguard-ui-theme"
        >
          <RainbowKitProvider>
            {children}
          </RainbowKitProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </WagmiProvider>
  );
}