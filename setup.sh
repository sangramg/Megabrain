#!/bin/bash
# Megabrain Setup Script
# Creates virtual environment and installs dependencies

set -e

SKILL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"; pwd )"
cd "$SKILL_DIR"

echo "🧠 Megabrain Setup"
echo "=================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "📍 Python version: $PYTHON_VERSION"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "📦 Virtual environment exists"
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
echo "📥 Upgrading pip..."
pip install -q --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Check for config.json
echo ""
if [ ! -f "config.json" ]; then
    echo "⚠️  config.json not found!"
    echo "   Run: cp config.json.template config.json"
    echo "   Then edit with your Zotero user ID"
else
    echo "✅ config.json exists"
fi

# Check environment variables
echo ""
echo "🔑 Checking environment variables..."
MISSING_KEYS=0

if [ -z "$ZOTERO_API_KEY" ]; then
    echo "   ❌ ZOTERO_API_KEY not set"
    MISSING_KEYS=1
else
    echo "   ✅ ZOTERO_API_KEY"
fi

if [ -z "$VOYAGE_API_KEY" ]; then
    echo "   ❌ VOYAGE_API_KEY not set"
    MISSING_KEYS=1
else
    echo "   ✅ VOYAGE_API_KEY"
fi

if [ -z "$VENICE_API_KEY" ]; then
    echo "   ❌ VENICE_API_KEY not set"
    MISSING_KEYS=1
else
    echo "   ✅ VENICE_API_KEY"
fi

if [ $MISSING_KEYS -eq 1 ]; then
    echo ""
    echo "   Add missing keys to ~/.bashrc:"
    echo "   export ZOTERO_API_KEY=\"your-key\""
    echo "   export VOYAGE_API_KEY=\"your-key\""
    echo "   export VENICE_API_KEY=\"your-key\""
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Configure:     cp config.json.template config.json"
echo "  2. Set API keys:  Add to ~/.bashrc (see above)"
echo "  3. Initial sync:  bash sync.sh"
echo "  4. Test search:   bash run.sh \"print(zotero_search('your query'))\""
echo ""
