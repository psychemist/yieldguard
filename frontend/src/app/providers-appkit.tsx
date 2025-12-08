'use client';

import { createAppKit } from '@reown/appkit/react';
import type { AppKitNetwork } from '@reown/appkit/networks';
import { WagmiProvider } from 'wagmi';
import { arbitrum, mainnet, polygon, optimism, base } from '@reown/appkit/networks';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { WagmiAdapter } from '@reown/appkit-adapter-wagmi';
import { ThemeProvider } from '@/contexts/theme-context';

const queryClient = new QueryClient();
const projectId = process.env.NEXT_PUBLIC_REOWN_PROJECT_ID || '';

const metadata = {
  name: 'YieldGuard Lite',
  description: 'AI-Powered DeFi Yield Optimization MVP',
  url: 'https://yieldguard.app',
  icons: ['https://assets.reown.com/reown-profile-pic.png']
};

const networks: [AppKitNetwork, ...AppKitNetwork[]] = [mainnet, arbitrum, polygon, optimism, base];

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
    analytics: true
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