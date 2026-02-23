#!/usr/bin/env python3
"""
Megabrain Research Tools for OpenClaw (formerly Zotero Research)
- Sync: Pull new items via Zotero API (literature/ index)
- Search: Natural language queries with RAG
- Memory: Add personal knowledge (mind/ index)
- Second Brain: General interests (second_brain/ index via second_brain.py)

Author: Percy (OpenClaw Agent)
Date: 2026-02-18 | Megabrain v2.0: 2026-02-19
"""

import json
import os
import re
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging
import requests
import torch

# Limit CPU threads to prevent SIGKILL from resource spikes
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

from pyzotero import zotero
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
import voyageai
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SKILL_DIR = Path(__file__).parent
CONFIG_PATH = SKILL_DIR / "config.json"
STATE_PATH = SKILL_DIR / "state.json"

# Load config
try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    logger.error(f"Config file not found: {CONFIG_PATH}")
    logger.error("Please create config.json with your Zotero credentials")
    raise

def get_api_key(config_section: str, key_name: str = 'api_key_env') -> str:
    """
    Get API key from environment variable specified in config.
    
    Args:
        config_section: Section in CONFIG (e.g., 'zotero', 'voyage')
        key_name: Key in that section containing env var name (default: 'api_key_env')
    
    Returns:
        API key from environment
        
    Raises:
        ValueError: If env var not set or empty
    """
    if config_section not in CONFIG:
        raise ValueError(f"Config section '{config_section}' not found")
    
    section = CONFIG[config_section]
    
    # Check for direct api_key first (backward compatibility)
    if 'api_key' in section and not section['api_key'].startswith('PASTE_YOUR'):
        logger.warning(f"Using hardcoded API key from config (insecure). Please use environment variables.")
        return section['api_key']
    
    # Get environment variable name
    if key_name not in section:
        raise ValueError(f"'{key_name}' not found in config['{config_section}']")
    
    env_var = section[key_name]
    api_key = os.environ.get(env_var)
    
    if not api_key:
        raise ValueError(f"Environment variable '{env_var}' not set. Please set it with: export {env_var}='your-key-here'")
    
    return api_key

# Expand paths
DB_PATH = Path(CONFIG['chromadb']['persist_directory']).expanduser()
PDF_CACHE = Path(CONFIG['sync']['pdf_cache']).expanduser()
PDF_CACHE.mkdir(parents=True, exist_ok=True)

# Get API keys from environment
ZOTERO_API_KEY = get_api_key('zotero')
VOYAGE_API_KEY = get_api_key('voyage') if 'voyage' in CONFIG else None

# Initialize Zotero client
logger.info(f"Initializing Zotero client for user {CONFIG['zotero']['user_id']}")
zot = zotero.Zotero(
    CONFIG['zotero']['user_id'],
    CONFIG['zotero']['library_type'],
    ZOTERO_API_KEY
)

# Initialize embeddings (lazy load to save memory)
_embeddings = None

def get_embeddings():
    """Lazy load embedding model (Voyage AI or HuggingFace fallback)"""
    global _embeddings
    if _embeddings is None:
        # Check if Voyage API key is available
        if VOYAGE_API_KEY:
            logger.info(f"Using VoyageAI: {CONFIG['voyage']['embedding_model']}")
            _embeddings = VoyageAIEmbeddings(
                voyage_api_key=VOYAGE_API_KEY,
                model=CONFIG['voyage']['embedding_model']
            )
        else:
            logger.info(f"Falling back to local: {CONFIG['chromadb']['embedding_model']}")
            _embeddings = HuggingFaceEmbeddings(
                model_name=CONFIG['chromadb']['embedding_model']
            )
    return _embeddings

# Initialize LLM (lazy load)
_llm = None

def get_llm():
    """Lazy load LLM"""
    global _llm
    if _llm is None:
        logger.info(f"Initializing LLM: {CONFIG['llm']['model']}")
        api_key = os.getenv(CONFIG['llm']['api_key_env'])
        if not api_key:
            logger.error(f"Environment variable {CONFIG['llm']['api_key_env']} not set")
            raise EnvironmentError(f"Missing {CONFIG['llm']['api_key_env']}")
        
        _llm = ChatOpenAI(
            model=CONFIG['llm']['model'],
            base_url=CONFIG['llm']['base_url'],
            api_key=api_key,
            temperature=CONFIG['llm']['temperature']
        )
    return _llm


# ── Citation helpers ──────────────────────────────────────────────

CITATION_BOOST_ALPHA = 0.15  # Tuning weight: 0 = ignore citations, 0.3 = heavy weight

def parse_citation_count(extra: str) -> int:
    """Parse citation count from Zotero 'extra' field.
    
    Formats handled:
        '377 citations (Crossref/DOI) [2026-02-16]'
        '5808 citations (Crossref) [2025-11-03]'
        'PMID: 12345\n42 citations (Crossref/DOI) [2026-01-01]'
    
    Returns 0 if no citation count found.
    """
    if not extra:
        return 0
    match = re.search(r'(\d+)\s+citations?\s*\(', extra, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def citation_boost_score(voyage_score: float, citation_count: int, alpha: float = CITATION_BOOST_ALPHA) -> float:
    """Apply citation-weighted boost to a relevance score.
    
    Formula: final = voyage_score × (1 + α × log(1 + citations))
    
    - α=0.15: gentle boost (seminal papers get ~50% lift at 5000 cites)
    - log() prevents mega-cited papers from dominating
    - Papers with 0 citations get no boost (multiplier = 1.0)
    """
    return voyage_score * (1.0 + alpha * math.log(1 + citation_count))


def backfill_citation_counts():
    """One-time backfill: fetch citation counts from Zotero and patch FAISS metadata.
    
    Fetches all items from Zotero, parses citation counts from 'extra' field,
    and updates the corresponding chunks in the FAISS literature index.
    No re-embedding needed — only metadata is patched.
    """
    import pickle
    
    logger.info("=" * 60)
    logger.info("BACKFILLING citation counts into FAISS metadata")
    logger.info("=" * 60)
    
    lit_path = DB_PATH / "literature"
    pkl_path = lit_path / "index.pkl"
    
    if not pkl_path.exists():
        return "❌ Literature index not found. Run zotero_sync() first."
    
    # Step 1: Fetch citation counts from Zotero (paginated to avoid OOM)
    logger.info("Fetching items from Zotero API (paginated)...")
    citation_map = {}  # zotero_key → citation_count
    start = 0
    batch_size = 100
    
    while True:
        items = zot.items(limit=batch_size, start=start)
        if not items:
            break
        for item in items:
            data = item['data']
            if data.get('itemType') in ('journalArticle', 'conferencePaper', 'book', 'bookSection', 'report', 'thesis'):
                count = parse_citation_count(data.get('extra', ''))
                citation_map[item['key']] = count
        start += batch_size
        logger.info(f"  Fetched {start} items...")
    
    logger.info(f"Found citation data for {len(citation_map)} papers")
    has_citations = sum(1 for v in citation_map.values() if v > 0)
    logger.info(f"  {has_citations} papers have citation counts, {len(citation_map) - has_citations} have 0")
    
    # Step 2: Patch FAISS metadata
    logger.info("Patching FAISS metadata...")
    with open(pkl_path, 'rb') as f:
        docstore, id_map = pickle.load(f)
    
    patched = 0
    for faiss_id, doc_id in id_map.items():
        doc = docstore.search(doc_id)
        zkey = doc.metadata.get('zotero_key')
        if zkey and zkey in citation_map:
            doc.metadata['citation_count'] = citation_map[zkey]
            patched += 1
    
    # Step 3: Save patched index
    with open(pkl_path, 'wb') as f:
        pickle.dump((docstore, id_map), f)
    
    logger.info(f"✅ Patched {patched} chunks with citation counts")
    return f"✅ Backfilled citation counts: {patched} chunks patched ({has_citations} papers with citations)"


class ZoteroSync:
    """Handles incremental sync from Zotero API"""
    
    def __init__(self):
        self.state = self._load_state()
        self.embeddings = get_embeddings()
    
    def _load_state(self) -> Dict:
        """Load sync state (last_synced timestamp)"""
        if STATE_PATH.exists():
            with open(STATE_PATH, 'r') as f:
                return json.load(f)
        return {
            'last_synced': None,
            'item_versions': {},
            'sync_history': []
        }
    
    def _save_state(self):
        """Save sync state"""
        with open(STATE_PATH, 'w') as f:
            json.dump(self.state, f, indent=2)
        logger.info(f"State saved: last_synced={self.state['last_synced']}")
    
    def fetch_new_items(self) -> List[Dict]:
        """Fetch items added/modified since last sync"""
        
        logger.info("Fetching new items from Zotero API...")
        
        try:
            # Fetch items modified since last sync
            if self.state['last_synced']:
                logger.info(f"Incremental sync from version {self.state['last_synced']}")
                items = zot.everything(zot.items(since=self.state['last_synced']))
            else:
                # Initial sync - get everything
                logger.info("Initial sync - fetching all items")
                items = zot.everything(zot.items())
        except Exception as e:
            logger.error(f"Zotero API error: {e}")
            raise
        
        # Filter to research items (articles, papers, books)
        research_types = [
            'journalArticle', 'conferencePaper', 'book',
            'bookSection', 'report', 'thesis'
        ]
        new_items = [
            item for item in items
            if item['data'].get('itemType') in research_types
        ]
        
        logger.info(f"Fetched {len(new_items)} new/modified research items")
        return new_items
    
    def fetch_standalone_notes(self) -> List[Dict]:
        """Fetch standalone notes (not attached to items) for mind index"""
        
        logger.info("Fetching standalone notes from Zotero...")
        
        try:
            if self.state['last_synced']:
                items = zot.everything(zot.items(since=self.state['last_synced']))
            else:
                items = zot.everything(zot.items())
        except Exception as e:
            logger.error(f"Zotero API error fetching notes: {e}")
            return []
        
        # Filter to standalone notes (no parentItem)
        standalone_notes = [
            item for item in items
            if item['data'].get('itemType') == 'note' 
            and not item['data'].get('parentItem')
        ]
        
        logger.info(f"Fetched {len(standalone_notes)} standalone notes")
        return standalone_notes
    
    def fetch_item_notes(self, item_key: str) -> List[str]:
        """Fetch notes attached to a specific item"""
        
        try:
            children = zot.children(item_key)
            notes = []
            for child in children:
                if child['data'].get('itemType') == 'note':
                    note_content = child['data'].get('note', '')
                    # Strip HTML tags from Zotero note
                    import re
                    clean_note = re.sub(r'<[^>]+>', '', note_content)
                    clean_note = clean_note.strip()
                    if clean_note:
                        notes.append(clean_note)
            return notes
        except Exception as e:
            logger.warning(f"Failed to fetch notes for {item_key}: {e}")
            return []
    
    def process_standalone_note(self, note_item: Dict) -> List[Document]:
        """Convert standalone Zotero note → Documents for mind index"""
        
        data = note_item['data']
        note_content = data.get('note', '')
        
        # Strip HTML tags
        import re
        clean_note = re.sub(r'<[^>]+>', '', note_content)
        clean_note = clean_note.strip()
        
        if not clean_note:
            return []
        
        metadata = {
            'source': 'zotero_note',
            'zotero_key': note_item['key'],
            'zotero_url': f"zotero://select/items/{note_item['key']}",
            'tags': ', '.join([t['tag'] for t in data.get('tags', [])]),
            'date_added': data.get('dateAdded', ''),
            'date_modified': data.get('dateModified', ''),
            'ingested_at': datetime.now().isoformat()
        }
        
        # Extract title from first line or first 50 chars
        first_line = clean_note.split('\n')[0][:100]
        metadata['title'] = first_line if first_line else 'Zotero Note'
        
        # Chunk if long
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )
        chunks = splitter.split_text(clean_note)
        
        docs = []
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta['chunk_id'] = i
            docs.append(Document(page_content=chunk, metadata=chunk_meta))
        
        logger.info(f"Created {len(docs)} chunks from standalone note: {metadata['title'][:40]}...")
        return docs
    
    def download_pdf(self, item_key: str) -> Optional[str]:
        """Download PDF attachment for item"""
        
        # Check cache first
        cached_pdf = PDF_CACHE / f"{item_key}.pdf"
        if cached_pdf.exists():
            logger.debug(f"PDF cached: {item_key}")
            return str(cached_pdf)
        
        # Get attachments
        try:
            attachments = zot.children(item_key)
        except Exception as e:
            logger.warning(f"Failed to fetch attachments for {item_key}: {e}")
            return None
        
        for att in attachments:
            if att['data'].get('contentType') == 'application/pdf':
                # Download PDF
                try:
                    logger.info(f"Downloading PDF: {item_key}")
                    # Note: zot.file() returns download URL, not binary
                    # We need to construct the URL manually
                    pdf_url = f"https://api.zotero.org/users/{CONFIG['zotero']['user_id']}/items/{att['key']}/file"
                    headers = {'Zotero-API-Key': ZOTERO_API_KEY}
                    response = requests.get(pdf_url, headers=headers, timeout=30)
                    
                    if response.status_code == 200:
                        with open(cached_pdf, 'wb') as f:
                            f.write(response.content)
                        logger.info(f"PDF saved: {cached_pdf}")
                        return str(cached_pdf)
                    else:
                        logger.warning(f"PDF download failed (status {response.status_code}): {item_key}")
                except Exception as e:
                    logger.warning(f"PDF download error for {item_key}: {e}")
        
        return None
    
    def process_item(self, item: Dict) -> List[Document]:
        """Convert Zotero item → LangChain Documents"""
        
        data = item['data']
        
        # Extract metadata
        creators = data.get('creators', [])
        authors = ', '.join([
            f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()
            for c in creators if c.get('creatorType') == 'author'
        ])
        
        metadata = {
            'title': data.get('title', 'Untitled'),
            'authors': authors or 'Unknown',
            'year': data.get('date', '')[:4] if data.get('date') else 'N/A',
            'abstract': data.get('abstractNote', ''),
            'doi': data.get('DOI', ''),
            'url': data.get('url', ''),
            'journal': data.get('publicationTitle', ''),
            'item_type': data.get('itemType', ''),
            'zotero_key': item['key'],
            'zotero_url': f"zotero://select/items/{item['key']}",
            'tags': ', '.join([t['tag'] for t in data.get('tags', [])]),
            'collections': ', '.join(data.get('collections', [])),
            'citation_count': parse_citation_count(data.get('extra', '')),
            'ingested_at': datetime.now().isoformat()
        }
        
        # Get text content
        text = metadata['abstract'] if metadata['abstract'] else metadata['title']
        
        # Download and extract PDF if available
        if CONFIG['sync']['download_pdfs']:
            pdf_path = self.download_pdf(item['key'])
            if pdf_path:
                try:
                    logger.info(f"Extracting text from PDF: {pdf_path}")
                    loader = PyMuPDFLoader(pdf_path)
                    pages = loader.load()
                    text = '\n\n'.join([p.page_content for p in pages])
                    metadata['pdf_path'] = pdf_path
                    metadata['pages'] = len(pages)
                    logger.info(f"Extracted {len(pages)} pages")
                except Exception as e:
                    logger.warning(f"PDF extraction failed: {e}")
        
        # Derive structured tags from Zotero tags
        zotero_tags = [t['tag'] for t in data.get('tags', [])]
        if zotero_tags:
            metadata['topic'] = zotero_tags[0]
            metadata['subtopics'] = ', '.join(zotero_tags[1:]) if len(zotero_tags) > 1 else zotero_tags[0]
        
        # Infer methodology from content heuristics
        try:
            from tag_inference import infer_methodology_heuristic
            meth = infer_methodology_heuristic(text)
            if meth:
                metadata['methodology'] = meth
        except Exception as e:
            logger.warning(f"Methodology heuristic failed: {e}")
        
        # Chunk text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )
        chunks = splitter.split_text(text)
        
        # Create Documents
        docs = []
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta['chunk_id'] = i
            docs.append(Document(page_content=chunk, metadata=chunk_meta))
        
        # Fetch and append item notes (attached to this paper)
        item_notes = self.fetch_item_notes(item['key'])
        if item_notes:
            logger.info(f"Found {len(item_notes)} attached notes for: {metadata['title'][:40]}...")
            combined_notes = "\n\n---\n\n".join(item_notes)
            note_chunks = splitter.split_text(combined_notes)
            for i, chunk in enumerate(note_chunks):
                note_meta = metadata.copy()
                note_meta['chunk_id'] = len(docs) + i
                note_meta['content_type'] = 'item_note'
                docs.append(Document(page_content=chunk, metadata=note_meta))
            logger.info(f"Added {len(note_chunks)} note chunks")
        
        logger.info(f"Created {len(docs)} chunks for: {metadata['title'][:50]}...")
        return docs
    
    def sync(self) -> str:
        """Main sync: fetch new items → process → store in ChromaDB"""
        
        logger.info("=" * 60)
        logger.info("Starting Zotero sync")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Fetch new research items
            items = self.fetch_new_items()
            
            # Process research items
            all_docs = []
            processed_titles = []
            errors = []
            
            for i, item in enumerate(items, 1):
                try:
                    logger.info(f"Processing [{i}/{len(items)}]...")
                    docs = self.process_item(item)
                    all_docs.extend(docs)
                    title = item['data'].get('title', 'Untitled')
                    processed_titles.append(title[:50])
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    errors.append(str(e))
            
            # Store research items in literature index
            if all_docs:
                logger.info(f"Storing {len(all_docs)} chunks in literature index...")
                lit_path = DB_PATH / "literature"
                if lit_path.exists():
                    db = FAISS.load_local(str(lit_path), self.embeddings, allow_dangerous_deserialization=True)
                else:
                    db = FAISS.from_documents([], self.embeddings)
                db.add_documents(all_docs)
                db.save_local(str(lit_path))
                logger.info("✅ Research documents stored successfully")
            
            # === STANDALONE NOTES → mind index ===
            standalone_notes = self.fetch_standalone_notes()
            note_docs = []
            note_count = 0
            
            for note in standalone_notes:
                try:
                    docs = self.process_standalone_note(note)
                    note_docs.extend(docs)
                    note_count += 1
                except Exception as e:
                    logger.error(f"Error processing standalone note: {e}")
                    errors.append(str(e))
            
            if note_docs:
                logger.info(f"Storing {len(note_docs)} chunks from {note_count} standalone notes in mind index...")
                mind_path = DB_PATH / "mind"
                if mind_path.exists():
                    mind_db = FAISS.load_local(str(mind_path), self.embeddings, allow_dangerous_deserialization=True)
                else:
                    mind_db = FAISS.from_documents([], self.embeddings)
                mind_db.add_documents(note_docs)
                mind_db.save_local(str(mind_path))
                logger.info("✅ Standalone notes stored in mind index")
            
            # Update state
            if all_docs or note_docs:
                self.state['last_synced'] = zot.last_modified_version()
                self.state['sync_history'].append({
                    'timestamp': datetime.now().isoformat(),
                    'items_added': len(items),
                    'chunks_added': len(all_docs),
                    'notes_synced': note_count,
                    'note_chunks': len(note_docs),
                    'errors': len(errors)
                })
                self._save_state()
            
            # Build result message
            duration = (datetime.now() - start_time).total_seconds()
            
            if not items and not standalone_notes:
                return "✅ No new items to sync"
            
            result = f"✅ **Synced in {duration:.1f}s**\n\n"
            
            if items:
                result += f"**Papers:** {len(items)} ({len(all_docs)} chunks)\n"
                for title in processed_titles[:3]:
                    result += f"• {title}...\n"
                if len(processed_titles) > 3:
                    result += f"• ... and {len(processed_titles) - 3} more\n"
            
            if note_count > 0:
                result += f"\n**Standalone Notes:** {note_count} ({len(note_docs)} chunks) → mind index\n"
            
            if errors:
                result += f"\n⚠️ **{len(errors)} errors** (check logs)"
            
            logger.info("Sync complete")
            return result
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return f"❌ Sync failed: {e}"


class ZoteroSearch:
    """Semantic search with RAG"""
    
    def __init__(self):
        self.embeddings = get_embeddings()
        self.llm = get_llm()
        
        # Load literature DB (FAISS)
        logger.info("Loading literature database...")
        lit_path = DB_PATH / "literature"
        if lit_path.exists():
            self.db_lit = FAISS.load_local(str(lit_path), self.embeddings, allow_dangerous_deserialization=True)
        else:
            logger.error("Literature DB not found. Run zotero_sync() first.")
            raise FileNotFoundError("Literature DB not found")
        
        # Load personal knowledge DB (FAISS)
        logger.info("Loading personal knowledge database...")
        mem_path = DB_PATH / "mind"
        if mem_path.exists():
            self.db_mem = FAISS.load_local(str(mem_path), self.embeddings, allow_dangerous_deserialization=True)
        else:
            # Create empty FAISS index
            logger.info("Personal mind DB not found; creating empty index...")
            self.db_mem = None
        
        # RAG prompt - STRUCTURED 4-PART format (user-preferred style from 2026-02-20)
        prompt_template = """You are an expert research synthesizer. Provide a STRUCTURED 4-part analysis.

## RESPONSE FORMAT (STRICT):

**Core Concept:** One-sentence definition answering the question directly.

**PART 1: THEMATIC ANALYSIS**
For each major theme (number them):

**Theme N: [Theme Name]**
• Brief explanation (1-2 sentences)
• Full citation: Author(s) (Year). Title. Journal, Volume(Issue), Pages. DOI: xxx
• Evidence: Key finding with [Author, Year] inline citation

**Library Ties:** Connect to other papers in the user's library (e.g., "Links to [Tiwana, 2013] on platforms...")

**PART 2: SYNTHESIS**
• **Interactions:** How themes connect (e.g., "Mimetic + normative amplify in tech")
• **Gaps:** What's missing from the literature

**PART 3: IMPLICATIONS**
• Bullet points with actionable insights
• How to apply, measure, or avoid pitfalls

**PART 4: SUMMARY**
One punchy sentence capturing the core takeaway.

**APPENDIX (Core Sources)**
For each key source:
N. Author(s) (Year) – Journal. Method: X. Contrib: Y. DOI: xxx

---

## CONSTRAINTS:
- **FULL CITATIONS**: Include Journal, Volume, Pages, DOI where available
- **NUMBERED THEMES**: Theme 1, Theme 2, etc.
- **LIBRARY TIES**: Reference other papers from user's indexed library
- **IMPLICATIONS**: Include practical/actionable points
- **PUNCHY SUMMARY**: One sentence max
- **LENGTH**: 500-800 words

## Retrieved Context:
{context}

## Research Question:
{question}

## Your Structured Analysis:"""
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db_lit.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        
        logger.info("Search system initialized")


# Standalone filter function for metadata filtering
def filter_docs(docs_with_scores, where):
    """Post-filter FAISS docs by metadata (Chroma-like where syntax).
    
    Supports:
        - Exact match: {"field": "value"} or {"field": {"$eq": "value"}}
        - Contains (case-insensitive): {"field": {"$contains": "substring"}}
        - Greater than: {"field": {"$gt": number}} or {"$gte": number}
        - In list: {"field": {"$in": ["val1", "val2"]}}
    """
    filtered = []
    for doc, score in docs_with_scores:
        meta = doc.metadata
        match = True
        for k, v in where.items():
            val = meta.get(k)
            if val is None:
                match = False
                break
            if isinstance(v, dict):
                if '$gte' in v:
                    try:
                        num_val = float(val) if isinstance(val, (str, int, float)) else 0
                        if num_val < v['$gte']:
                            match = False
                    except ValueError:
                        match = False
                elif '$gt' in v:
                    try:
                        num_val = float(val) if isinstance(val, (str, int, float)) else 0
                        if num_val <= v['$gt']:
                            match = False
                    except ValueError:
                        match = False
                elif '$eq' in v:
                    if str(val).lower() != str(v['$eq']).lower():
                        match = False
                elif '$contains' in v:
                    if str(v['$contains']).lower() not in str(val).lower():
                        match = False
                elif '$in' in v:
                    if str(val).lower() not in [str(x).lower() for x in v['$in']]:
                        match = False
            else:
                if str(val).lower() != str(v).lower():
                    match = False
            if not match:
                break
        if match:
            filtered.append((doc, score))
    return filtered


def chunk_llm_context(docs, max_docs=8):
    """Chunk large context into LLM-friendly batches."""
    if len(docs) <= max_docs:
        return [docs]
    batches = []
    for i in range(0, len(docs), max_docs):
        batches.append(docs[i:i + max_docs])
    return batches


# ── ZoteroSearch.search() — restored as proper class method via monkey-patch ──

def _zotero_search_method(self, query, where=None):
        """Execute search with RAG + Voyage reranking + metadata filtering + large output handling."""
        logger.info(f"Searching for: {query} {where}")
        
        try:
            # Step 1: Get top-100 candidates from FAISS with scores
            logger.info("Fetching top-100 candidates from FAISS...")
            raw_docs_scores = self.db_lit.similarity_search_with_score(query, k=100)
            raw_docs_scores = [(doc, 1.0 - score) for doc, score in raw_docs_scores]  # Normalize to [0,1] relevance
            
            # Step 1b: Apply metadata filter (post-retrieval)
            if where:
                logger.info(f"Filtering {len(raw_docs_scores)} candidates with where={where}")
                raw_docs_scores = filter_docs(raw_docs_scores, where)
                logger.info(f"After filter: {len(raw_docs_scores)} candidates")
            
            raw_docs = [doc for doc, score in raw_docs_scores]
            
            # Step 2: Rerank with Voyage (if configured)
            reranked_docs = raw_docs[:20]  # Increased pool for large results
            
            if VOYAGE_API_KEY:
                try:
                    logger.info(f"Reranking with Voyage rerank-2 ({len(raw_docs)} -> top-20)...")
                    vo = voyageai.Client(api_key=VOYAGE_API_KEY)
                    
                    rerank_texts = [doc.page_content for doc in raw_docs]
                    rerank_result = vo.rerank(
                        query=query,
                        documents=rerank_texts,
                        model="rerank-2",
                        top_k=20  # Larger for robust handling
                    )
                    
                    # Citation boost
                    boosted = []
                    for r in rerank_result.results:
                        doc = raw_docs[r.index]
                        cites = doc.metadata.get('citation_count', 0)
                        if isinstance(cites, str):
                            try:
                                cites = int(cites)
                            except (ValueError, TypeError):
                                cites = 0
                        boosted_score = citation_boost_score(r.relevance_score, cites)
                        boosted.append((doc, r.relevance_score, boosted_score, cites))
                    
                    boosted.sort(key=lambda x: x[2], reverse=True)
                    reranked_docs = [b[0] for b in boosted]
                    
                    # Log top-3
                    log_parts = []
                    for doc, orig, final, cites in boosted[:3]:
                        cite_str = f" [{cites} cites]" if cites > 0 else ""
                        log_parts.append(f"{orig:.3f}→{final:.3f}{cite_str}")
                    logger.info(f"Reranked top-20 with citation boost: {', '.join(log_parts)}")
                    
                except Exception as e:
                    logger.warning(f"Reranking failed, using FAISS top-20: {e}")
                    reranked_docs = raw_docs[:20]
            
            # Large output handling
            full_sources_path = None
            if len(reranked_docs) > 10:
                full_sources_path = SKILL_DIR / f"search_results_{int(datetime.now().timestamp())}.md"
                with open(full_sources_path, 'w') as f:
                    f.write(f"# Full Sources: '{query}' (filtered: {where})\n\n")
                    for i, doc in enumerate(reranked_docs):
                        meta = doc.metadata
                        f.write(f"## Source {i+1}: {meta.get('title', 'N/A')}\n")
                        f.write(f"**Authors/Year**: {meta.get('authors', 'N/A')} ({meta.get('year', 'N/A')})\n")
                        f.write(f"**Content**: {doc.page_content}\n\n")
                logger.info(f"Saved full sources to {full_sources_path}")
            
            # Single-pass synthesis with top-8 docs (verbose format)
            logger.info(f"Generating verbose synthesis from top-8 docs...")
            top_docs = reranked_docs[:8]  # Top 8 for detailed analysis
            
            context_text = "\n\n---\n\n".join([
                f"[Source {i+1}] {doc.page_content[:1200]}"  # More text per source for quotes
                for i, doc in enumerate(top_docs)
            ])
            
            answer = self.llm.invoke(self.prompt.format(context=context_text, question=query), timeout=120).content
            
            # Build response with full source references
            response = answer
            
            # Add all Zotero library references from top 20
            response += "\n\n---\n\n### 📚 ALL REFERENCES FROM LIBRARY\n"
            seen = set()
            for doc in reranked_docs[:20]:
                meta = doc.metadata
                key = (meta.get('title'), meta.get('year'))
                if key not in seen:
                    seen.add(key)
                    response += f"\n• **{meta.get('title', 'N/A')}**"
                    response += f"\n  {meta.get('authors', 'N/A')} ({meta.get('year', 'N/A')})"
                    if meta.get('journal'):
                        response += f" — {meta.get('journal')}"
                    cites = meta.get('citation_count', 0)
                    if int(cites) > 0:
                        response += f" [📊 {cites} citations]"
                    if meta.get('doi'):
                        response += f"\n  DOI: {meta.get('doi')}"
                    response += "\n"
            
            # Add full sources file reference if created
            if full_sources_path:
                response += f"\n💾 **Full text chunks**: `{full_sources_path.name}`"
            
            # Personal notes from Mind partition (verbose, relevance threshold)
            try:
                if self.db_mem:
                    mem_docs_scores = self.db_mem.similarity_search_with_score(query, k=5)
                    relevant_docs = [doc for doc, score in mem_docs_scores if score < 0.4]  # FAISS distance threshold (lower = better)
                    if relevant_docs:
                        response += "\n\n### 🧠 RELATED NOTES FROM MIND PARTITION\n"
                        for idx, doc in enumerate(relevant_docs[:3], 1):
                            meta = doc.metadata
                            response += f"\n**Note {idx}**: {meta.get('title', 'Untitled')}\n"
                            if meta.get('date_added'):
                                response += f"Date: {meta.get('date_added')[:10]}\n"
                            if meta.get('topic'):
                                response += f"Topic: {meta.get('topic')}\n"
                            snippet = doc.page_content[:300].replace('\n', ' ')
                            response += f"{snippet}...\n"
                    else:
                        response += "\n\n### 🧠 MIND PARTITION\nNo relevant notes found (relevance threshold: distance < 0.4)."
            except Exception as e:
                logger.debug(f"Mind search skipped: {e}")
                response += "\n\n### 🧠 MIND PARTITION\nNo relevant notes found."
            
            logger.info("Search completed (large output handled)")
            return response
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"❌ Search failed: {e}\n\nPlease check database."

# Attach search method to ZoteroSearch class (was accidentally de-indented)
ZoteroSearch.search = _zotero_search_method

# Public API for OpenClaw

def zotero_search(query: str, where: dict = None) -> str:
    """
    OpenClaw tool: Search Zotero library with natural language + optional metadata filters
    
    Args:
        query: Natural language question (e.g., "What factors influence digital transformation?")
        where: Optional metadata filter dict. Supports:
            - Exact match: {"field": "value"} or {"field": {"$eq": "value"}}
            - Contains (case-insensitive): {"field": {"$contains": "substring"}}
            - Greater than: {"field": {"$gt": number}} or {"$gte": number}
            - In list: {"field": {"$in": ["val1", "val2"]}}
            
            Common fields: authors, year, title, journal, methodology, topic, tags
            
            Examples:
                where={"authors": {"$contains": "Kathuria"}}
                where={"year": {"$gte": 2020}}
                where={"methodology": "qualitative"}
    
    Returns:
        Formatted response with answer, citations, and sources
    """
    try:
        searcher = ZoteroSearch()
        return searcher.search(query, where=where)
    except Exception as e:
        logger.error(f"zotero_search failed: {e}")
        return f"❌ Search system error: {e}"


def zotero_sync() -> str:
    """
    OpenClaw tool: Sync new Zotero items
    
    Returns:
        Summary of synced items
    """
    try:
        syncer = ZoteroSync()
        return syncer.sync()
    except Exception as e:
        logger.error(f"zotero_sync failed: {e}")
        return f"❌ Sync error: {e}"


def citation_backfill() -> str:
    """
    OpenClaw tool: Backfill citation counts into existing FAISS metadata.
    
    Fetches citation counts from Zotero 'extra' field for all papers and patches
    the existing FAISS literature index metadata. No re-embedding needed.
    
    Run this once after enabling citation-boosted ranking, or periodically
    to refresh citation counts (they change over time).
    
    Returns:
        Summary of patched chunks
    """
    try:
        return backfill_citation_counts()
    except Exception as e:
        logger.error(f"Citation backfill failed: {e}")
        return f"❌ Citation backfill failed: {e}"


def zotero_sync_notes() -> str:
    """
    OpenClaw tool: Sync all Zotero notes (standalone + item notes)
    
    Use this to index notes from existing items that were synced before note support was added.
    - Standalone notes → mind index
    - Item notes → literature index (attached to parent paper)
    
    Returns:
        Summary of synced notes
    """
    try:
        syncer = ZoteroSync()
        embeddings = syncer.embeddings
        
        logger.info("=" * 60)
        logger.info("Syncing all Zotero notes")
        logger.info("=" * 60)
        
        # Get ALL items (not just new ones)
        items = zot.everything(zot.items())
        
        # 1. Process standalone notes
        standalone_notes = [
            item for item in items
            if item['data'].get('itemType') == 'note' 
            and not item['data'].get('parentItem')
        ]
        
        note_docs = []
        for note in standalone_notes:
            docs = syncer.process_standalone_note(note)
            note_docs.extend(docs)
        
        if note_docs:
            logger.info(f"Storing {len(note_docs)} standalone note chunks in mind...")
            mind_path = DB_PATH / "mind"
            if mind_path.exists():
                mind_db = FAISS.load_local(str(mind_path), embeddings, allow_dangerous_deserialization=True)
                mind_db.add_documents(note_docs)
            else:
                mind_db = FAISS.from_documents(note_docs, embeddings)
            mind_db.save_local(str(mind_path))
        
        # 2. Fetch item notes for existing research items
        research_types = ['journalArticle', 'conferencePaper', 'book', 'bookSection', 'report', 'thesis']
        research_items = [item for item in items if item['data'].get('itemType') in research_types]
        
        item_note_docs = []
        items_with_notes = 0
        
        for item in research_items:
            notes = syncer.fetch_item_notes(item['key'])
            if notes:
                items_with_notes += 1
                combined_notes = "\n\n---\n\n".join(notes)
                
                # Create metadata
                data = item['data']
                creators = data.get('creators', [])
                authors = ', '.join([
                    f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()
                    for c in creators if c.get('creatorType') == 'author'
                ])
                
                metadata = {
                    'title': data.get('title', 'Untitled'),
                    'authors': authors or 'Unknown',
                    'year': data.get('date', '')[:4] if data.get('date') else 'N/A',
                    'zotero_key': item['key'],
                    'zotero_url': f"zotero://select/items/{item['key']}",
                    'content_type': 'item_note',
                    'ingested_at': datetime.now().isoformat()
                }
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(combined_notes)
                
                for i, chunk in enumerate(chunks):
                    chunk_meta = metadata.copy()
                    chunk_meta['chunk_id'] = i
                    item_note_docs.append(Document(page_content=chunk, metadata=chunk_meta))
        
        if item_note_docs:
            logger.info(f"Storing {len(item_note_docs)} item note chunks in literature...")
            lit_path = DB_PATH / "literature"
            if lit_path.exists():
                lit_db = FAISS.load_local(str(lit_path), embeddings, allow_dangerous_deserialization=True)
                lit_db.add_documents(item_note_docs)
            else:
                lit_db = FAISS.from_documents(item_note_docs, embeddings)
            lit_db.save_local(str(lit_path))
        
        result = f"✅ **Notes synced**\n\n"
        result += f"**Standalone notes:** {len(standalone_notes)} ({len(note_docs)} chunks) → mind\n"
        result += f"**Item notes:** {items_with_notes} items ({len(item_note_docs)} chunks) → literature\n"
        
        return result
        
    except Exception as e:
        logger.error(f"zotero_sync_notes failed: {e}")
        return f"❌ Notes sync error: {e}"


def add_memory(content: str, metadata: dict = None) -> str:
    """
    OpenClaw tool: Add note to personal knowledge
    
    Args:
        content: Note text
        metadata: Optional metadata dict
    
    Returns:
        Confirmation message
    """
    try:
        embeddings = get_embeddings()
        mind_path = DB_PATH / "mind"
        
        doc = Document(
            page_content=content,
            metadata=metadata or {
                'type': 'manual',
                'added': datetime.now().isoformat()
            }
        )
        
        if mind_path.exists():
            db = FAISS.load_local(str(mind_path), embeddings, allow_dangerous_deserialization=True)
            db.add_documents([doc])
        else:
            db = FAISS.from_documents([doc], embeddings)
        db.save_local(str(mind_path))
        
        logger.info(f"Memory added: {content[:60]}...")
        return f"✅ Added memory: {content[:60]}..."
    except Exception as e:
        logger.error(f"add_memory failed: {e}")
        return f"❌ Memory addition failed: {e}"


# CLI

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Zotero Research Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Initial sync:     python zotero_tools.py --init-sync
  Incremental sync: python zotero_tools.py --sync
  Search:           python zotero_tools.py --search "digital transformation"
  Add memory:       python zotero_tools.py --add-memory "SG prefers IST timestamps"
        """
    )
    parser.add_argument('--init-sync', action='store_true', help="Initial full sync")
    parser.add_argument('--sync', action='store_true', help="Incremental sync")
    parser.add_argument('--search', type=str, help="Search query")
    parser.add_argument('--add-memory', type=str, help="Add personal note")
    
    args = parser.parse_args()
    
    if args.init_sync or args.sync:
        print(zotero_sync())
    elif args.search:
        print(zotero_search(args.search))
    elif args.add_memory:
        print(add_memory(args.add_memory))
    else:
        parser.print_help()

# Import user content functions
import sys
sys.path.insert(0, str(Path(__file__).parent))
from user_content import add_note as _add_note, add_url as _add_url, add_video as _add_video, add_audio as _add_audio

# Export as OpenClaw tools
def add_note(note: str, title: str = None) -> str:
    """
    OpenClaw tool: Add text note to personal knowledge base
    
    Args:
        note: Note content
        title: Optional title
    
    Returns:
        Confirmation message
    """
    return _add_note(note, title)

def add_url_content(url: str, title: str = None) -> str:
    """
    OpenClaw tool: Extract and add URL content to knowledge base
    
    Args:
        url: Webpage URL
        title: Optional title
    
    Returns:
        Confirmation message with extracted content summary
    """
    return _add_url(url, title)

def add_video_transcript(url: str, title: str = None) -> str:
    """
    OpenClaw tool: Extract and add video transcript to knowledge base
    
    Args:
        url: YouTube video URL
        title: Optional title
    
    Returns:
        Confirmation message with transcript summary
    """
    return _add_video(url, title)


def add_audio_content(url: str, title: str = None) -> str:
    """
    OpenClaw tool: Extract and add audio content (podcast, recording) to knowledge base
    
    Grabs full metadata (title, author, date, description, duration) from source.
    If transcript unavailable, asks user whether to paste one manually.
    If critical metadata is missing, prompts user for details.
    
    Args:
        url: Audio URL (podcast, SoundCloud, Spotify, etc.)
        title: Optional title override
    
    Returns:
        Confirmation with metadata summary, or prompt for missing info
    """
    return _add_audio(url, title)


# Second Brain functions (Megabrain expansion)
# Legacy backward compatibility: Import second_brain tools directly
# These are DEPRECATED wrappers - use megabrain_tools instead
from second_brain import add_to_brain, search_brain as brain_search, universal_search
