#!/bin/bash

# AgentSilex Web Demo - Quick Start Script

echo "🚀 Starting AgentSilex Web Demo..."
echo ""

# Check if .env file exists in parent directory
if [ ! -f ../.env ]; then
    echo "⚠️  Warning: .env file not found in parent directory!"
    echo "   Please create ../.env and add your API keys"
    echo "   Example: OPENAI_API_KEY=sk-your-key-here"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Found .env file in parent directory"
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed!"
    echo "   Please install uv first: https://github.com/astral-sh/uv"
    exit 1
fi

echo ""
echo "📦 Installing dependencies with uv..."
uv pip install -q -r requirements.txt

echo ""
echo "🌐 Starting server on http://localhost:8080"
echo "   Press Ctrl+C to stop"
echo ""

# Use uv run to execute with the correct environment
uv run python app.py
