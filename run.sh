#!/bin/bash
# Megabrain Python runner
# Usage: bash run.sh "python_code_here"
#
# Examples:
#   bash run.sh "print(zotero_search('digital transformation'))"
#   bash run.sh "print(universal_search('productivity'))"
#   bash run.sh "zotero_sync()"

SKILL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"; pwd )"
cd "$SKILL_DIR"

# Activate venv
source venv/bin/activate

# Import and run
python3 -c "
import sys
sys.path.insert(0, '.')
from zotero_tools import *
from second_brain import *
from megabrain_tools import *
$1
"
