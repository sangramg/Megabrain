#!/usr/bin/env python3
"""
Unified Megabrain API
Clean partition-aware interface for library/, mind/, second_brain/

Author: Percy (OpenClaw Agent)
Date: 2026-02-22
Version: 3.1
"""
import json
import os
import logging
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)
SKILL_DIR = Path(__file__).parent
CONFIG_PATH = SKILL_DIR / "config.json"

with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# Helper to get API keys from environment
def get_api_key(config_section: str, key_name: str = 'api_key_env') -> Optional[str]:
    """Get API key from environment variable specified in config."""
    if config_section not in CONFIG:
        return None
    section = CONFIG[config_section]
    if 'api_key' in section and not section['api_key'].startswith('PASTE_YOUR'):
        return section['api_key']
    if key_name not in section:
        return None
    env_var = section[key_name]
    return os.environ.get(env_var)

VOYAGE_API_KEY = get_api_key('voyage')

# Existing imports
from zotero_tools import (
    zotero_search, 
    zotero_sync, 
    zotero_sync_notes,
    get_embeddings
)
from user_content import add_note as _add_note_raw
from second_brain import (
    add_to_brain as _add_to_brain_raw,
    search_brain,
    universal_search
)


# ============================================================
# LIBRARY PARTITION (Research: Zotero + Manual)
# ============================================================

def add_library(
    content: str,
    title: str,
    authors: str,
    year: int,
    metadata: Optional[Dict] = None
) -> str:
    """
    Manually add research document to library/ (alongside Zotero papers)
    
    Args:
        content: Full text or abstract
        title: Document title
        authors: Comma-separated author names
        year: Publication year
        metadata: {
            methodology: "qualitative|quantitative|mixed|conceptual|review|design-science" (REQUIRED),
            topic: str (primary topic, REQUIRED),
            subtopics: str (comma-separated),
            publication: str (journal/conference),
            doi: str (optional),
            source_type: "pdf|report|thesis|preprint|book-chapter" (default: pdf)
        }
    
    Returns: Success message with doc ID
    """
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Validate required metadata
    if not metadata:
        metadata = {}
    
    if 'methodology' not in metadata:
        return "❌ ERROR: 'methodology' is required for library/ documents.\nChoose one: qualitative, quantitative, mixed, conceptual, review, design-science"
    
    if 'topic' not in metadata:
        return "❌ ERROR: 'topic' is required. Provide primary research topic."
    
    # Build document metadata
    doc_metadata = {
        'title': title,
        'authors': authors,
        'year': year,
        'methodology': metadata['methodology'],
        'topic': metadata['topic'],
        'subtopics': metadata.get('subtopics', ''),
        'publication': metadata.get('publication', 'Manual Entry'),
        'doi': metadata.get('doi', ''),
        'source': 'manual',
        'source_type': metadata.get('source_type', 'pdf'),
        'date_added': datetime.now().isoformat(),
        'partition': 'library'
    }
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = splitter.split_text(content)
    logger.info(f"Split '{title}' into {len(chunks)} chunks")
    
    # Create Documents
    docs = [
        Document(
            page_content=chunk,
            metadata={**doc_metadata, 'chunk_id': i}
        )
        for i, chunk in enumerate(chunks)
    ]
    
    # Add to library FAISS index
    from langchain_community.vectorstores import FAISS
    
    lit_path = Path(CONFIG['chromadb']['persist_directory']).expanduser() / "literature"
    lit_path.mkdir(parents=True, exist_ok=True)
    
    embeddings = get_embeddings()
    
    if (lit_path / "index.faiss").exists():
        logger.info("Loading existing literature FAISS index...")
        db = FAISS.load_local(str(lit_path), embeddings, allow_dangerous_deserialization=True)
    else:
        logger.info("Creating new literature FAISS index...")
        db = FAISS.from_documents([], embeddings)
    
    db.add_documents(docs)
    db.save_local(str(lit_path))
    
    logger.info(f"✅ Added '{title}' to library/ ({len(chunks)} chunks)")
    return f"✅ Added to library/: '{title}' ({authors}, {year})\n   Methodology: {metadata['methodology']}\n   Topic: {metadata['topic']}\n   Chunks indexed: {len(chunks)}"


def search_library(query: str) -> str:
    """Search library/ partition (Zotero + manual research)"""
    return zotero_search(query)


# ============================================================
# MIND PARTITION (Personal Notes)
# ============================================================

def add_mind(content: str, metadata: Optional[Dict] = None) -> str:
    """
    Add personal note to mind/
    
    Args:
        content: Note text
        metadata: {
            topic: str (REQUIRED - primary topic),
            subtopics: str (comma-separated),
            context: str (when/why written),
            sentiment: "supportive|critical|exploratory|neutral",
            supersedes: str (title/ID of note this updates),
            action_items: str (comma-separated todos),
            related_to: str (cross-links to other notes/papers)
        }
    
    Returns: Success message with doc ID
    """
    if not metadata or 'topic' not in metadata:
        return "❌ ERROR: 'topic' is required for mind/ notes.\nProvide metadata={'topic': 'your_topic', ...}"
    
    title = metadata.get('topic', 'Note')
    
    # Pass metadata to underlying function
    return _add_note_raw(content, title=title, metadata=metadata)


# Deprecated alias for backward compatibility
def add_memory(content: str, metadata: Optional[Dict] = None) -> str:
    """DEPRECATED: Use add_mind() instead. Kept for backward compatibility."""
    logger.warning("add_memory() is deprecated. Use add_mind() to avoid confusion with MEMORY.md")
    return add_mind(content, metadata)


def add_note(note: str, title: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
    """
    Simplified add_memory (auto-infers topic if missing)
    
    Args:
        note: Note text
        title: Optional title (defaults to first line or "Note")
        metadata: Optional metadata dict (topic will be inferred if missing)
    
    Returns: Success message
    """
    return _add_note_raw(note, title=title, metadata=metadata)


def search_mind(query: str, top_k: int = 10) -> str:
    """
    Search mind/ partition (personal notes) using FAISS + Voyage rerank
    
    Args:
        query: Search query
        top_k: Number of results to return (default: 10)
    
    Returns:
        Formatted search results from mind/ index
    """
    from langchain_community.vectorstores import FAISS
    import voyageai
    
    # Load mind FAISS index
    mind_path = Path(CONFIG['chromadb']['persist_directory']).expanduser() / "mind"
    if not (mind_path / "index.faiss").exists():
        return "🔍 Mind index is empty. Add notes with add_memory() or add_note()."
    
    embeddings = get_embeddings()
    db = FAISS.load_local(str(mind_path), embeddings, allow_dangerous_deserialization=True)
    
    # Initial FAISS search (top 100)
    docs_scores = db.similarity_search_with_score(query, k=min(100, top_k * 10))
    
    if not docs_scores:
        return f"🔍 No results found in mind/ for: {query}"
    
    # Rerank with Voyage
    try:
        client = voyageai.Client(api_key=VOYAGE_API_KEY)
        doc_texts = [doc.page_content for doc, _ in docs_scores]
        
        rerank_result = client.rerank(
            query=query,
            documents=doc_texts,
            model="rerank-2",
            top_k=top_k
        )
        
        # Map reranked results back to docs
        reranked_docs = []
        for result in rerank_result.results:
            orig_doc, orig_score = docs_scores[result.index]
            reranked_docs.append((orig_doc, result.relevance_score))
        
        docs_scores = reranked_docs
        
    except Exception as e:
        logger.warning(f"Reranking failed, using FAISS results: {e}")
        docs_scores = docs_scores[:top_k]
    
    # Format results
    output = [f"🧩 Mind Search Results for: '{query}'\n"]
    output.append(f"Found {len(docs_scores)} relevant notes:\n")
    
    for i, (doc, score) in enumerate(docs_scores, 1):
        meta = doc.metadata
        title = meta.get('title', 'Untitled')
        topic = meta.get('topic', '')
        context = meta.get('context', '')
        added = meta.get('added_at', '')
        
        output.append(f"\n{'='*60}")
        output.append(f"[{i}] {title} (Score: {score:.3f})")
        if topic:
            output.append(f"    Topic: {topic}")
        if context:
            output.append(f"    Context: {context}")
        if added:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(added.replace('Z', '+00:00'))
                output.append(f"    Added: {dt.strftime('%Y-%m-%d %H:%M')}")
            except:
                output.append(f"    Added: {added}")
        
        output.append(f"\n{doc.page_content[:300]}...")
    
    return '\n'.join(output)


# ============================================================
# SECOND BRAIN PARTITION (General Interests)
# ============================================================

def add_brain(
    content: str,
    category: str,
    tags: str,
    source_author: Optional[str] = None,
    title: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    Add general interest content to second_brain/
    
    Args:
        content: Text content
        category: Enforced (e.g., 'tech', 'philosophy', 'cooking', 'health')
        tags: Comma-separated keywords (REQUIRED)
        source_author: Original content creator
        title: Optional title
        metadata: {
            topic: str,
            subtopics: str,
            related_to: str
        }
    
    Returns: Success message
    """
    if not metadata:
        metadata = {}
    
    return _add_to_brain_raw(
        content, 
        category=category, 
        tags=tags,
        source_author=source_author,
        title=title,
        **metadata
    )


# search_brain imported directly from second_brain.py (no wrapper needed)


# ============================================================
# CROSS-PARTITION SEARCH
# ============================================================

# Imported directly from second_brain.py:
# - universal_search(query) → All three partitions
# - search_brain(query, category) → Second brain only


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Library (Research)
    'add_library',
    'search_library',
    'zotero_search',  # Alias for search_library
    'zotero_sync',
    'zotero_sync_notes',
    
    # Mind (Personal)
    'add_mind',       # PRIMARY
    'add_note',
    'search_mind',
    'add_memory',     # DEPRECATED (backward compat)
    
    # Second Brain (Interests)
    'add_brain',
    'search_brain',
    
    # Universal
    'universal_search'
]
