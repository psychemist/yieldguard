# YieldGuard

**AI-Powered DeFi Yield Optimization Platform**  
YieldGuard is an advanced DeFi analytics and automation platform that leverages autonomous AI agents to help users optimize their yield farming strategies. By combining real-time market data, risk analysis, and Large Language Models (LLMs) via Groq, YieldGuard provides personalized, risk-adjusted investment recommendations through an interactive chat interface and a comprehensive dashboard.

---

## 🚀 Key Features

### 🤖 Autonomous AI Agent

- **Natural Language Interface:** Chat with the agent to request strategies, analyze pools, or ask about market conditions.
- **ReAct Pattern:** The agent uses a "Reasoning + Acting" loop to break down complex queries, plan steps, and execute tools (e.g., "Fetch yields", "Check gas", "Analyze risk") to generate accurate answers.
- **Memory & Context:** Remembers your risk tolerance, capital constraints, and previous interactions to provide tailored advice.

### 📊 Intelligent Market Analysis

- **Trend Detection:** Automatically identifies rising or falling yield trends using linear regression.
- **Volatility Scoring:** Categorizes pools into Low, Moderate, High, or Extreme volatility buckets.
- **Market Stance:** Computes a global "Stance" (Favorable, Neutral, Caution, Wait) based on aggregated trends, volatility, and gas costs.

### 🛡️ Risk Management & Strategy

- **Risk Profiling:** Customizes strategies for Conservative, Moderate, or Aggressive risk profiles.
- **Gas Optimization:** Factors in real-time gas prices (Ethereum L1) to ensure strategies are profitable net of fees.
- **Safety Scores:** Evaluates pools based on TVL, Impermanent Loss (IL) risk, and protocol maturity.

### 💻 Modern Dashboard

- **Real-time Metrics:** View live APY, TVL, and gas prices.
- **Visualizations:** Interactive charts for yield history and portfolio composition.
- **Wallet Connection:** Seamlessly connect via RainbowKit (MetaMask, WalletConnect, etc.) to manage your portfolio.

---

## 🏗️ Architecture

YieldGuard operates as a modern full-stack application:

```
├── backend/                       # Python FastAPI Backend
│   ├── src/
│   │   ├── services/              # Core Logic
│   │   │   ├── agent.py           # AI Agent (Groq integration, Tool Registry)
│   │   │   ├── analyzer.py        # Statistical Analysis Engine
│   │   │   ├── data_service.py    # Data fetching (DeFiLlama, etc.)
│   │   │   └── model_runner.py    # Strategy Execution
│   │   ├── models/                # Pydantic Models
│   │   └── test/                  # Test files
│   ├── main.py                    # Application Entry Point
│   └── requirements.txt
│
├── frontend/                      # Next.js 15 Frontend
│   ├── src/
│   │   ├── app/                   # App Router Pages
│   │   ├── components/            # React Components (Shadcn UI)
│   │   ├── contexts/              # App Context (Theme)
│   │   └── lib/                   # Utilities
│   ├── package.json
│   └── next.config.ts
│
└── database/                      # Database Resources
    └── schema.sql                 # PostgreSQL Schema
```

---

## 🛠️ Tech Stack

### Backend

- **Framework:** FastAPI (Python)
- **AI/LLM:** Groq API (Llama 3 / Mixtral)
- **Data Processing:** Pandas, NumPy
- **Async I/O:** `httpx`, `asyncio`

### Frontend

- **Framework:** Next.js 15 (App Router)
- **Language:** TypeScript
- **Styling:** TailwindCSS, Shadcn UI
- **Web3:** RainbowKit, Wagmi, Viem
- **State/Query:** React Query (TanStack Query)
- **Charts:** Recharts

### Data Sources

- **Token Prices:** CoinGecko
- **DeFi Data:** DeFiLlama API
- **Gas Data:** Etherscan / Owlracle

---

## 🏁 Getting Started

### Prerequisites

- **Node.js:** v18+
- **Python:** v3.9+
- **API Keys:**
  - **Groq API Key:** For the AI agent.
  - **WalletConnect Project ID:** For the frontend wallet connection.
  - **Etherscan API Key:** (Optional) For precise gas data.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd yieldguard
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Environment
cp .env.example .env
# ⚠️ Open .env and add your GROQ_API_KEY
```

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Configure Environment
# Create a .env.local file or update providers.tsx with your WalletConnect ID
```

### 4. Run the Application

**Terminal 1 (Backend):**

```bash
cd backend
source venv/bin/activate
python main.py
# Server runs at http://localhost:8000
# API Docs at http://localhost:8000/docs
```

**Terminal 2 (Frontend):**

```bash
cd frontend
npm run dev
# App runs at http://localhost:3000
```

---

## 🧪 Development & Testing

- **Backend Tests:** Run `pytest` in the `backend/` directory.
- **Linting:**
  - Backend: Uses `ruff`.
  - Frontend: Uses `eslint`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
