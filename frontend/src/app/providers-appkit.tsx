'use client';

import { createAppKit } from '@reown/appkit/react';
import { WagmiProvider } from 'wagmi';
import { arbitrum, mainnet, polygon, optimism, base } from '@reown/appkit/networks';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { WagmiAdapter } from '@reown/appkit-adapter-wagmi';
import { ThemeProvider } from '@/contexts/theme-context';

// 0. Setup queryClient
const queryClient = new QueryClient();

// 1. Get projectId from https://cloud.reown.com
const projectId = 'ca6398f2043be1d89cc89ac1dc12c56f';

// 2. Create a metadata object - optional
const metadata = {
  name: 'YieldGuard',
  description: 'AI-Powered DeFi Yield Optimization MVP',
  url: 'https://yieldguard.app', // origin must match your domain & subdomain
  icons: ['https://assets.reown.com/reown-profile-pic.png']
};

// 3. Set the networks
const networks = [mainnet, arbitrum, polygon, optimism, base];

// 4. Create Wagmi Adapter
const wagmiAdapter = new WagmiAdapter({
  networks,
  projectId,
  ssr: true
});

// 5. Create modal
createAppKit({
  adapters: [wagmiAdapter],
  networks,
  projectId,
  metadata,
  features: {
    analytics: true // Optional - defaults to your Cloud configuration
  }
});

export function AppKitProvider({ children }: { children: React.ReactNode }) {
  return (
    <WagmiProvider config={wagmiAdapter.wagmiConfig}>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider
          defaultTheme="system"
          storageKey="yieldguard-ui-theme"
        >
          {children}
        </ThemeProvider>
      </QueryClientProvider>
    </WagmiProvider>
  );
}