#!/bin/bash
# YieldGuard Lite - Development Startup Script
# Run this to start both backend and frontend

set -e

echo "🚀 Starting YieldGuard Lite Development Environment"
echo "=================================================="

# Check for required environment variables
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  GROQ_API_KEY not set. AI recommendations will use fallback logic."
    echo "   Set it with: export GROQ_API_KEY='your-key-here'"
fi

# Start backend
echo ""
echo "📡 Starting Backend API..."
cd backend
if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
pip3 install -q -r requirements.txt

# Run in background
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
echo "   Backend started on http://localhost:8000 (PID: $BACKEND_PID)"

# Wait for backend to be ready
echo "   Waiting for backend..."
sleep 3

# Start frontend
echo ""
echo "🎨 Starting Frontend..."
cd ../frontend
if [ ! -d "node_modules" ]; then
    echo "   Installing dependencies..."
    npm install
fi
npm run build &
FRONTEND_PID=$!
echo "   Frontend started on http://localhost:3000 (PID: $FRONTEND_PID)"

echo ""
echo "=================================================="
echo "✅ YieldGuard Lite is running!"
echo ""
echo "   Backend API: http://localhost:8000"
echo "   Frontend UI: http://localhost:3000"
echo "   API Docs:    http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "=================================================="

# Wait and handle shutdown
trap "echo ''; echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
