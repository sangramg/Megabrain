#!/usr/bin/env python3
"""
OpenClaw tool integration for Megabrain
Unified API exports for library/, mind/, second_brain/
"""
import sys
import os

# Add skill dir to path
sys.path.insert(0, os.path.dirname(__file__))

# Set DB path from env or default
if 'DB_PATH' not in os.environ:
    os.environ['DB_PATH'] = os.path.expanduser('~/.openclaw/workspace/research-vector-db')

# Import unified API
from megabrain_tools import (
    # Library (Research)
    add_library,
    search_library,
    zotero_search,
    zotero_sync,
    zotero_sync_notes,
    
    # Mind (Personal)
    add_mind,
    add_note,
    search_mind,
    add_memory,  # Deprecated alias
    
    # Second Brain (Interests)
    add_brain,
    search_brain,
    
    # Universal
    universal_search,
    brain_search
)

# Legacy exports (optional - backward compatibility)
from user_content import add_url, add_video

# Export to OpenClaw
__all__ = [
    # Library
    'add_library',
    'search_library',
    'zotero_search',
    'zotero_sync',
    'zotero_sync_notes',
    
    # Mind
    'add_mind',
    'add_note',
    'search_mind',
    'add_memory',  # Deprecated
    
    # Second Brain
    'add_brain',
    'search_brain',
    
    # Universal
    'universal_search',
    'brain_search',
    
    # Legacy
    'add_url',
    'add_video'
]
