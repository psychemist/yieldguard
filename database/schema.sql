-- YieldGuard Lite Database Schema
-- PostgreSQL/Supabase compatible

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    wallet_address TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Yield data table for historical tracking
CREATE TABLE yield_data (
    id SERIAL PRIMARY KEY,
    protocol TEXT NOT NULL,
    asset TEXT NOT NULL,
    date DATE NOT NULL,
    apy NUMERIC(10, 4) NOT NULL,
    tvl NUMERIC(20, 2) NOT NULL,
    gas_cost NUMERIC(10, 4),
    price NUMERIC(12, 6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Create unique constraint to prevent duplicate entries
    UNIQUE(protocol, asset, date)
);

-- Recommendations table
CREATE TABLE recommendations (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    capital NUMERIC(12, 2) NOT NULL,
    risk_profile TEXT NOT NULL CHECK (risk_profile IN ('low', 'medium', 'high')),
    allocation JSONB NOT NULL,
    expected_yield NUMERIC(10, 4) NOT NULL,
    risk_score NUMERIC(3, 2) NOT NULL,
    gas_cost_estimate NUMERIC(10, 2) NOT NULL,
    confidence_score NUMERIC(3, 2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Gas price tracking
CREATE TABLE gas_prices (
    id SERIAL PRIMARY KEY,
    slow NUMERIC(8, 2) NOT NULL,
    standard NUMERIC(8, 2) NOT NULL,
    fast NUMERIC(8, 2) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_yield_data_protocol_asset ON yield_data(protocol, asset);
CREATE INDEX idx_yield_data_date ON yield_data(date);
CREATE INDEX idx_recommendations_user_id ON recommendations(user_id);
CREATE INDEX idx_recommendations_date ON recommendations(date);
CREATE INDEX idx_gas_prices_timestamp ON gas_prices(timestamp);

-- Sample data for development
INSERT INTO yield_data (protocol, asset, date, apy, tvl, gas_cost, price) VALUES
('uniswap-v3', 'USDC/ETH', CURRENT_DATE, 12.5, 50000000, 25.0, 2000.0),
('uniswap-v3', 'USDT/ETH', CURRENT_DATE, 8.3, 30000000, 25.0, 2000.0),
('uniswap-v3', 'WBTC/ETH', CURRENT_DATE, 15.2, 20000000, 25.0, 2000.0);

INSERT INTO gas_prices (slow, standard, fast) VALUES (20.0, 25.0, 30.0);