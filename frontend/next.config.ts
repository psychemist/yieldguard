import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Transpile problematic packages
  transpilePackages: [
    "@reown/appkit",
    "@reown/appkit-adapter-wagmi",
    "@reown/appkit-controllers",
    "@walletconnect/universal-provider",
    "@walletconnect/utils",
    "viem",
    "wagmi",
    "@wagmi/core",
    "@wagmi/connectors",
    "@base-org/account",
  ],

  webpack: (config, { isServer }) => {
    // Handle node modules that aren't browser compatible
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
        crypto: false,
        stream: false,
        path: false,
        os: false,
      };
    }

    config.resolve.alias = {
      ...config.resolve.alias,
      "pino-pretty": false,
      "@react-native-async-storage/async-storage": false,
    };

    // Handle pino/thread-stream test imports
    config.module.rules.push({
      test: /node_modules\/(pino|thread-stream)\/.*test.*\.js$/,
      loader: "ignore-loader",
    });

    // Ignore problematic optional dependencies
    config.externals = [
      ...(config.externals || []),
      "why-is-node-running",
      "tap",
    ];

    return config;
  },

  // Skip type checking during build for faster builds (we check separately)
  typescript: {
    ignoreBuildErrors: false,
  },

  // Suppress hydration warnings from wallet libraries
  reactStrictMode: true,
};

export default nextConfig;
