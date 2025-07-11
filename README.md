# YieldGuard Lite MVP

AI-Powered DeFi Yield Optimization Platform - MVP Version

## 🚀 Overview

YieldGuard Lite is a simplified MVP that demonstrates AI-powered yield optimization for DeFi protocols. It focuses on Uniswap V3 pools and provides intelligent allocation recommendations based on risk profiles, gas costs, and market conditions.

### Key Features

- **AI-Powered Recommendations**: Smart allocation suggestions based on historical yield data
- **Risk Management**: Customizable risk profiles (Low, Medium, High)
- **Gas Optimization**: Real-time gas price tracking and cost estimation
- **Interactive Dashboard**: Visual charts and metrics for yield tracking
- **Wallet Integration**: Connect with MetaMask and other wallets via RainbowKit

## 🏗️ Architecture

```
├── backend/          # Python FastAPI backend
│   ├── src/
│   │   ├── api/      # API endpoints
│   │   ├── models/   # Data models
│   │   ├── services/ # Business logic
│   │   └── utils/    # Utilities
│   ├── main.py       # FastAPI app
│   └── requirements.txt
├── frontend/         # Next.js frontend
│   ├── src/
│   │   ├── app/      # App router pages
│   │   ├── components/ # React components
│   │   └── lib/      # Utilities
│   └── package.json
└── database/         # PostgreSQL schema
    └── schema.sql
```

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **PostgreSQL** - Database for storing yield data and recommendations
- **Pandas/NumPy** - Data processing and analysis
- **XGBoost** - Machine learning for yield optimization
- **httpx** - HTTP client for external APIs

### Frontend
- **Next.js 15** - React framework with App Router
- **TailwindCSS** - Utility-first CSS framework
- **shadcn/ui** - Beautiful component library
- **RainbowKit + Wagmi** - Web3 wallet integration
- **Recharts** - Data visualization

### Data Sources
- **DefiLlama** - Yield and TVL data
- **Etherscan** - Gas price data
- **CoinGecko** - Token price data

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+
- PostgreSQL 13+

### Setup

1. **Clone and setup the project:**
```bash
git clone <repository-url>
cd yieldguard
chmod +x setup.sh
./setup.sh
```

2. **Configure environment variables:**
```bash
# Backend
cp backend/.env.example backend/.env
# Edit backend/.env with your database credentials and API keys

# Frontend
# Get WalletConnect project ID from https://cloud.walletconnect.com
# Update frontend/src/app/providers.tsx with your project ID
```

3. **Set up database:**
```bash
createdb yieldguard_lite
psql yieldguard_lite < database/schema.sql
```

4. **Start the services:**
```bash
# Terminal 1: Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py

# Terminal 2: Frontend
cd frontend
npm run dev
```

5. **Access the application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📊 MVP Features

### Dashboard
- Real-time yield metrics display
- Historical yield performance charts
- Risk assessment indicators
- Gas cost tracking

### Strategy Builder
- Capital amount input
- Risk profile selection (Low/Medium/High)
- Gas preference slider
- AI-powered allocation recommendations

### AI Model
- Risk-adjusted yield scoring
- Portfolio optimization algorithms
- Gas cost impact analysis
- Confidence scoring

## 🔧 API Endpoints

### Core Endpoints
- `GET /` - Health check
- `POST /recommendations` - Get AI recommendations
- `GET /yields/historical` - Historical yield data
- `GET /gas/current` - Current gas prices
- `GET /protocols` - Supported protocols

### Example Request
```bash
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "capital": 1000,
    "risk_profile": "medium",
    "wallet_address": "0x..."
  }'
```

## 🎯 Development Roadmap

### Week 1-2: Core MVP
- [x] Backend API with FastAPI
- [x] Frontend dashboard with Next.js
- [x] Basic AI model for recommendations
- [x] Wallet connection integration

### Week 3-4: Enhancement
- [ ] Real-time data integration
- [ ] Advanced risk calculations
- [ ] User session management
- [ ] Enhanced UI/UX

### Week 5-6: Polish
- [ ] Performance optimization
- [ ] Error handling improvements
- [ ] Testing and validation
- [ ] Deployment preparation

## 🔮 Future Enhancements

### Full YieldGuard Features
- Multi-protocol support (Curve, Balancer, etc.)
- Layer 2 integration (Arbitrum, Optimism)
- Advanced ML models (reinforcement learning)
- Impermanent loss calculations
- Automated execution via MPC wallets
- Portfolio rebalancing strategies

## 📈 Performance Metrics

The MVP tracks several key metrics:
- **Expected Yield**: Projected annual percentage yield
- **Risk Score**: Portfolio risk assessment (0-1 scale)
- **Gas Cost**: Estimated transaction costs
- **Confidence Score**: AI model confidence level

## 🔒 Security Considerations

- No private key handling (wallet-only integration)
- Read-only blockchain interactions
- Rate limiting on API endpoints
- Input validation and sanitization
- Secure environment variable management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Check the GitHub Issues
- Review the API documentation at `/docs`
- Ensure all environment variables are properly configured

---

**YieldGuard Lite** - Democratizing DeFi yield optimization through AI 🚀