#!/usr/bin/env python3
"""
Second Brain Module for Megabrain
Handles general interests and non-research content

This is the third index in the Megabrain system:
- literature/ → Research papers from Zotero
- mind/ → Personal notes and user-contributed content
- second_brain/ → General interests, ideas, web clippings (this module)

Author: Percy (OpenClaw Agent)
Date: 2026-02-19
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import voyageai
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setup
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

DB_PATH = Path(CONFIG['chromadb']['persist_directory']).expanduser()
SECOND_BRAIN_PATH = DB_PATH / "second_brain"


def get_embeddings():
    """Lazy load Voyage embeddings"""
    from zotero_tools import get_embeddings as _get_embeddings
    return _get_embeddings()


def get_second_brain_db():
    """Get or create second brain FAISS index"""
    embeddings = get_embeddings()
    SECOND_BRAIN_PATH.mkdir(parents=True, exist_ok=True)
    
    faiss_file = SECOND_BRAIN_PATH / "index.faiss"
    if faiss_file.exists():
        logger.info("Loading second brain FAISS index...")
        db = FAISS.load_local(
            str(SECOND_BRAIN_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        logger.info("Creating new second brain FAISS index...")
        # Initialize with dummy document
        dummy_doc = Document(
            page_content="Second brain initialized",
            metadata={"type": "system", "created": datetime.now().isoformat()}
        )
        db = FAISS.from_documents([dummy_doc], embeddings)
        db.save_local(str(SECOND_BRAIN_PATH))
        logger.info(f"Second brain index created at {SECOND_BRAIN_PATH}")
    
    return db


def add_to_brain(
    content: str,
    source_type: str = 'note',
    url: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    title: Optional[str] = None
) -> str:
    """
    Add content to second brain FAISS index
    
    Args:
        content: Text content (or will be extracted from URL/video)
        source_type: 'note', 'url', 'video', 'idea', 'snippet'
        url: Source URL (if applicable)
        category: Category/topic (e.g., 'tech', 'philosophy', 'cooking')
        tags: List of tags
        title: Title for the content
    
    Returns:
        Confirmation message
    """
    try:
        # Extract from URL/video if needed
        if source_type == 'url' and url:
            from user_content import extract_url_content
            content, extracted_meta = extract_url_content(url)
            title = title or extracted_meta.get('title')
        elif source_type == 'video' and url:
            from user_content import extract_youtube_content
            content, extracted_meta, _ = extract_youtube_content(url)
            title = title or extracted_meta.get('title')
        
        # Build metadata
        metadata = {
            'source_type': source_type,
            'added_at': datetime.now().isoformat(),
            'added_by': 'user',
            'index': 'second_brain'
        }
        
        if title:
            metadata['title'] = title
        if url:
            metadata['url'] = url
        if category:
            metadata['category'] = category
        if tags:
            metadata['tags'] = ', '.join(tags)
        
        # Extract source_author from yt-dlp metadata for video source_type
        if source_type == 'video' and url:
            metadata['source_author'] = extracted_meta.get('author', extracted_meta.get('channel', ''))
        
        # Auto-infer structured tags if missing
        if not metadata.get('topic') or not metadata.get('subtopics'):
            try:
                from tag_inference import infer_tags_second_brain
                inferred = infer_tags_second_brain(content, metadata)
                for k in ('topic', 'subtopics', 'category'):
                    if inferred.get(k) and not metadata.get(k):
                        metadata[k] = inferred[k]
                logger.info(f"Auto-inferred tags: topic={metadata.get('topic')}")
            except Exception as e:
                logger.warning(f"Tag inference failed (non-fatal): {e}")
        
        # Flag missing required fields
        missing_fields = []
        if not metadata.get('category'):
            missing_fields.append('category')
        if not metadata.get('tags'):
            missing_fields.append('tags')
        if not metadata.get('source_author'):
            missing_fields.append('source_author')
        
        # Chunk if long
        if len(content) > 2000:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_text(content)
            logger.info(f"Split into {len(chunks)} chunks")
        else:
            chunks = [content]
        
        # Create documents
        docs = []
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta['chunk_id'] = i
            chunk_meta['total_chunks'] = len(chunks)
            docs.append(Document(page_content=chunk, metadata=chunk_meta))
        
        # Store in second brain DB
        db = get_second_brain_db()
        db.add_documents(docs)
        db.save_local(str(SECOND_BRAIN_PATH))
        
        # Build response
        result = f"✅ Added to second brain ({len(chunks)} chunks)"
        if title:
            result += f"\n   📝 {title}"
        if category:
            result += f"\n   🏷️  Category: {category}"
        if tags:
            result += f"\n   🔖 Tags: {', '.join(tags)}"
        result += f"\n   📊 Content: {len(content)} characters"
        if metadata.get('topic'):
            result += f"\n   🎯 Topic: {metadata['topic']}"
        if metadata.get('subtopics'):
            result += f"\n   📌 Subtopics: {metadata['subtopics']}"
        
        if missing_fields:
            result += f"\n\n⚠️ **Missing required fields**: {', '.join(missing_fields)}"
            result += "\nPlease provide so I can update the entry."
        
        logger.info(f"Added to second brain: {source_type} ({len(docs)} docs)")
        return result
        
    except Exception as e:
        logger.error(f"add_to_brain failed: {e}")
        return f"❌ Failed to add to second brain: {e}"


def search_brain(query: str, category: Optional[str] = None, top_k: int = 8) -> str:
    """
    Search second brain only (excludes research papers)
    
    Args:
        query: Search query
        category: Optional category filter
        top_k: Number of results
    
    Returns:
        Formatted search results
    """
    try:
        db = get_second_brain_db()
        
        # Get top-50 for reranking
        raw_docs = db.similarity_search(query, k=50)
        
        # Filter by category if specified
        if category:
            raw_docs = [
                doc for doc in raw_docs
                if doc.metadata.get('category', '').lower() == category.lower()
            ]
        
        # Rerank with Voyage if configured
        if VOYAGE_API_KEY:
            try:
                logger.info(f"Reranking {len(raw_docs)} results with Voyage rerank-2...")
                vo = voyageai.Client(api_key=VOYAGE_API_KEY)
                
                rerank_texts = [doc.page_content for doc in raw_docs]
                rerank_result = vo.rerank(
                    query=query,
                    documents=rerank_texts,
                    model="rerank-2",
                    top_k=min(top_k, len(raw_docs))
                )
                
                results = [raw_docs[r.index] for r in rerank_result.results]
                logger.info(f"Reranked to top-{len(results)}")
            except Exception as e:
                logger.warning(f"Reranking failed, using FAISS results: {e}")
                results = raw_docs[:top_k]
        else:
            results = raw_docs[:top_k]
        
        # Format response
        if not results:
            return f"🔍 No results found in second brain for: {query}"
        
        response = f"🧠 **Second Brain Search Results** ({len(results)} items)\n\n"
        
        seen = set()
        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            # Deduplicate by title or first 50 chars
            dedup_key = meta.get('title', doc.page_content[:50])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            
            response += f"**{i}. "
            if meta.get('title'):
                response += f"{meta['title']}**\n"
            else:
                response += f"{doc.page_content[:60]}...**\n"
            
            if meta.get('category'):
                response += f"   🏷️  {meta['category']}\n"
            if meta.get('tags'):
                response += f"   🔖 {meta['tags']}\n"
            if meta.get('url'):
                response += f"   🔗 {meta['url']}\n"
            
            # Show snippet
            snippet = doc.page_content[:200]
            if len(doc.page_content) > 200:
                snippet += "..."
            response += f"   {snippet}\n\n"
        
        logger.info(f"Second brain search returned {len(results)} results")
        return response
        
    except Exception as e:
        logger.error(f"brain_search failed: {e}")
        return f"❌ Second brain search failed: {e}"


def universal_search(query: str, top_k: int = 8) -> str:
    """
    Search ALL indexes: literature + mind + second_brain
    
    Args:
        query: Search query
        top_k: Number of results per index
    
    Returns:
        Formatted combined search results with source labels
    """
    try:
        embeddings = get_embeddings()
        
        # Load all three indexes
        logger.info("Loading all three indexes for universal search...")
        
        db_lit = FAISS.load_local(
            str(DB_PATH / "literature"),
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        db_mem = FAISS.load_local(
            str(DB_PATH / "mind"),
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        db_brain = get_second_brain_db()
        
        # Search each index (get top-50 for reranking)
        logger.info("Searching across all indexes...")
        lit_docs = db_lit.similarity_search(query, k=50)
        mem_docs = db_mem.similarity_search(query, k=50)
        brain_docs = db_brain.similarity_search(query, k=50)
        
        # Label sources
        for doc in lit_docs:
            doc.metadata['search_source'] = 'literature'
        for doc in mem_docs:
            doc.metadata['search_source'] = 'mind'
        for doc in brain_docs:
            doc.metadata['search_source'] = 'second_brain'
        
        # Combine all results
        all_docs = lit_docs + mem_docs + brain_docs
        logger.info(f"Combined {len(all_docs)} results from all indexes")
        
        # Rerank combined results with Voyage
        if VOYAGE_API_KEY:
            try:
                logger.info(f"Reranking {len(all_docs)} combined results...")
                vo = voyageai.Client(api_key=VOYAGE_API_KEY)
                
                rerank_texts = [doc.page_content for doc in all_docs]
                rerank_result = vo.rerank(
                    query=query,
                    documents=rerank_texts,
                    model="rerank-2",
                    top_k=top_k
                )
                
                results = [all_docs[r.index] for r in rerank_result.results]
                logger.info(f"Universal search reranked to top-{len(results)}")
            except Exception as e:
                logger.warning(f"Reranking failed, using top results: {e}")
                results = all_docs[:top_k]
        else:
            results = all_docs[:top_k]
        
        # Format response with source labels
        if not results:
            return f"🔍 No results found across all indexes for: {query}"
        
        response = f"🌐 **Universal Search Results** ({len(results)} from all sources)\n\n"
        
        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            source = meta.get('search_source', 'unknown')
            
            # Source icon
            icon = {
                'literature': '📚',
                'mind': '🧩',
                'second_brain': '🧠'
            }.get(source, '📄')
            
            response += f"**{i}. {icon} [{source.upper()}]** "
            
            # Title
            if meta.get('title'):
                response += f"{meta['title']}\n"
            else:
                response += f"{doc.page_content[:60]}...\n"
            
            # Source-specific metadata
            if source == 'literature':
                if meta.get('authors'):
                    response += f"   👥 {meta['authors']} ({meta.get('year', 'N/A')})\n"
                if meta.get('doi'):
                    response += f"   🔗 DOI: {meta['doi']}\n"
            elif source in ['mind', 'second_brain']:
                if meta.get('category'):
                    response += f"   🏷️  {meta['category']}\n"
                if meta.get('tags'):
                    response += f"   🔖 {meta['tags']}\n"
            
            # Snippet
            snippet = doc.page_content[:150]
            if len(doc.page_content) > 150:
                snippet += "..."
            response += f"   {snippet}\n\n"
        
        logger.info(f"Universal search returned {len(results)} results")
        return response
        
    except Exception as e:
        logger.error(f"universal_search failed: {e}")
        return f"❌ Universal search failed: {e}"


# CLI
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python second_brain.py add-note 'My idea' --title 'Title' --category 'tech'")
        print("  python second_brain.py search 'my query'")
        print("  python second_brain.py universal 'search all indexes'")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == 'add-note':
        content = sys.argv[2] if len(sys.argv) > 2 else ""
        title = None
        category = None
        
        # Parse optional args
        for i, arg in enumerate(sys.argv[3:], 3):
            if arg == '--title' and i + 1 < len(sys.argv):
                title = sys.argv[i + 1]
            elif arg == '--category' and i + 1 < len(sys.argv):
                category = sys.argv[i + 1]
        
        print(add_to_brain(content, source_type='note', title=title, category=category))
    
    elif cmd == 'search':
        query = sys.argv[2] if len(sys.argv) > 2 else ""
        print(brain_search(query))
    
    elif cmd == 'universal':
        query = sys.argv[2] if len(sys.argv) > 2 else ""
        print(universal_search(query))
    
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
