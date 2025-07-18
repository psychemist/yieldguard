#!/bin/bash

# YieldGuard Lite Development Setup Script

echo "🚀 Setting up YieldGuard Lite MVP..."

# Backend Setup
echo "📦 Setting up backend..."
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
echo "✅ Backend setup complete"

# Frontend Setup
echo "📦 Setting up frontend..."
cd ../frontend

# Install dependencies (already done, but ensuring completeness)
npm install

echo "✅ Frontend setup complete"

# Database Setup
echo "🗄️  Setting up database..."
cd ../database

echo "📝 Database schema created. Please run schema.sql in your PostgreSQL instance."

echo "🎉 YieldGuard Lite MVP setup complete!"
echo ""
echo "Next steps:"
echo "1. Set up PostgreSQL database and run database/schema.sql"
echo "2. Update backend/.env with your database credentials"
echo "3. Get API keys for Etherscan and update .env"
echo "4. Get WalletConnect project ID and update frontend/src/app/providers.tsx"
echo ""
echo "To start development:"
echo "Backend: cd backend && python main.py"
echo "Frontend: cd frontend && npm run dev"