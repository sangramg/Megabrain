#!/bin/bash
# Quick Zotero sync
# Usage: bash sync.sh

SKILL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"; pwd )"
cd "$SKILL_DIR"

source venv/bin/activate

python3 -c "
import sys
sys.path.insert(0, '.')
from zotero_tools import zotero_sync
print(zotero_sync())
"
