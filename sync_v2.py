#!/usr/bin/env python3
"""
Megabrain Zotero Sync v2 — Option E (Hybrid)
- Per-item version tracking with PDF content hash
- Metadata-only fast path (no re-embedding for tag/citation changes)
- Batched FAISS writes with checkpointing (every 10 items)
- Deduplication by zotero_key before adding
- Standalone notes → mind index

Author: Percy (OpenClaw Agent)
Date: 2026-02-20
"""

import json
import os
import hashlib
import pickle
import sys
import math
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import torch
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import requests
from pyzotero import zotero
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Paths & Config ──
SKILL_DIR = Path(__file__).parent
CONFIG_PATH = SKILL_DIR / "config.json"
STATE_PATH = SKILL_DIR / "state.json"

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

ZOTERO_API_KEY = get_api_key('zotero')
VOYAGE_API_KEY = get_api_key('voyage')

DB_PATH = Path(CONFIG['chromadb']['persist_directory']).expanduser()
PDF_CACHE = Path(CONFIG['sync']['pdf_cache']).expanduser()
PDF_CACHE.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 10  # Save checkpoint every N items

# ── Zotero client ──
zot = zotero.Zotero(CONFIG['zotero']['user_id'], CONFIG['zotero']['library_type'], ZOTERO_API_KEY)

# ── Embeddings (lazy) ──
_embeddings = None
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        if VOYAGE_API_KEY:
            _embeddings = VoyageAIEmbeddings(
                voyage_api_key=VOYAGE_API_KEY,
                model=CONFIG['voyage']['embedding_model']
            )
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(model_name=CONFIG['chromadb']['embedding_model'])
    return _embeddings


# ── Helpers ──

def parse_citation_count(extra: str) -> int:
    if not extra:
        return 0
    match = re.search(r'(\d+)\s+citations?\s*\(', extra, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def pdf_content_hash(pdf_path: str) -> str:
    """SHA256 of PDF file content."""
    h = hashlib.sha256()
    with open(pdf_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def load_state() -> Dict:
    if STATE_PATH.exists():
        with open(STATE_PATH, 'r') as f:
            state = json.load(f)
        # Ensure item_versions exists
        if 'item_versions' not in state:
            state['item_versions'] = {}
        return state
    return {'last_synced': None, 'item_versions': {}, 'sync_history': []}


def save_state(state: Dict):
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)
    logger.info(f"State saved: last_synced={state['last_synced']}, tracked_items={len(state['item_versions'])}")


def extract_metadata(item: Dict) -> Dict:
    """Extract contextual metadata from Zotero item (who/when/where/what — no file stats)."""
    data = item['data']
    creators = data.get('creators', [])
    authors = ', '.join([
        f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()
        for c in creators if c.get('creatorType') == 'author'
    ])
    return {
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


def download_pdf(item_key: str) -> Optional[str]:
    """Download PDF, return path (uses cache)."""
    cached_pdf = PDF_CACHE / f"{item_key}.pdf"
    if cached_pdf.exists():
        return str(cached_pdf)

    try:
        attachments = zot.children(item_key)
    except Exception as e:
        logger.warning(f"Failed to fetch attachments for {item_key}: {e}")
        return None

    for att in attachments:
        if att['data'].get('contentType') == 'application/pdf':
            try:
                pdf_url = f"https://api.zotero.org/users/{CONFIG['zotero']['user_id']}/items/{att['key']}/file"
                headers = {'Zotero-API-Key': ZOTERO_API_KEY}
                response = requests.get(pdf_url, headers=headers, timeout=30)
                if response.status_code == 200:
                    with open(cached_pdf, 'wb') as f:
                        f.write(response.content)
                    return str(cached_pdf)
                else:
                    logger.warning(f"PDF download failed ({response.status_code}): {item_key}")
            except Exception as e:
                logger.warning(f"PDF download error for {item_key}: {e}")
    return None


def fetch_item_notes(item_key: str) -> List[str]:
    """Fetch notes attached to a specific item."""
    try:
        children = zot.children(item_key)
        notes = []
        for child in children:
            if child['data'].get('itemType') == 'note':
                note_content = child['data'].get('note', '')
                clean = re.sub(r'<[^>]+>', '', note_content).strip()
                if clean:
                    notes.append(clean)
        return notes
    except Exception as e:
        logger.warning(f"Failed to fetch notes for {item_key}: {e}")
        return []


def chunk_text(text: str, metadata: Dict) -> List[Document]:
    """Split text into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(chunks):
        m = metadata.copy()
        m['chunk_id'] = i
        docs.append(Document(page_content=chunk, metadata=m))
    return docs


def full_process_item(item: Dict) -> List[Document]:
    """Full pipeline: extract text from PDF (or abstract), chunk, return docs."""
    metadata = extract_metadata(item)
    text = metadata['abstract'] if metadata['abstract'] else metadata['title']

    if CONFIG['sync']['download_pdfs']:
        pdf_path = download_pdf(item['key'])
        if pdf_path:
            try:
                loader = PyMuPDFLoader(pdf_path)
                pages = loader.load()
                text = '\n\n'.join([p.page_content for p in pages])
                metadata['pdf_path'] = pdf_path
                metadata['pages'] = len(pages)
            except Exception as e:
                logger.warning(f"PDF extraction failed for {item['key']}: {e}")

    docs = chunk_text(text, metadata)

    # Attached notes
    notes = fetch_item_notes(item['key'])
    if notes:
        combined = "\n\n---\n\n".join(notes)
        note_docs = chunk_text(combined, {**metadata, 'content_type': 'item_note'})
        # Renumber chunk_ids
        for j, d in enumerate(note_docs):
            d.metadata['chunk_id'] = len(docs) + j
        docs.extend(note_docs)

    return docs


def patch_faiss_metadata(db_path: Path, zotero_key: str, new_metadata: Dict) -> int:
    """Patch metadata for all chunks matching zotero_key in FAISS index. No re-embedding.
    Returns number of chunks patched."""
    pkl_path = db_path / "index.pkl"
    if not pkl_path.exists():
        return 0

    with open(pkl_path, 'rb') as f:
        docstore, id_map = pickle.load(f)

    patched = 0
    for faiss_id, doc_id in id_map.items():
        doc = docstore.search(doc_id)
        if doc.metadata.get('zotero_key') == zotero_key:
            # Update metadata fields (preserve content-derived fields like chunk_id, pdf_path, pages)
            for key in ['title', 'authors', 'year', 'abstract', 'doi', 'url', 'journal',
                        'tags', 'collections', 'citation_count', 'item_type']:
                if key in new_metadata:
                    doc.metadata[key] = new_metadata[key]
            doc.metadata['metadata_updated'] = datetime.now().isoformat()
            patched += 1

    if patched > 0:
        with open(pkl_path, 'wb') as f:
            pickle.dump((docstore, id_map), f)

    return patched


def remove_chunks_by_key(db_path: Path, zotero_key: str) -> int:
    """Remove all chunks for a zotero_key from FAISS index (for re-add after content change).
    Returns number removed. Note: FAISS doesn't support true deletion, so we rebuild without them."""
    import faiss
    import numpy as np

    faiss_path = db_path / "index.faiss"
    pkl_path = db_path / "index.pkl"
    if not faiss_path.exists() or not pkl_path.exists():
        return 0

    index = faiss.read_index(str(faiss_path))
    with open(pkl_path, 'rb') as f:
        docstore, id_map = pickle.load(f)

    # Find IDs to keep
    keep_faiss_ids = []
    remove_count = 0
    for faiss_id, doc_id in id_map.items():
        doc = docstore.search(doc_id)
        if doc.metadata.get('zotero_key') == zotero_key:
            remove_count += 1
        else:
            keep_faiss_ids.append(faiss_id)

    if remove_count == 0:
        return 0

    # Rebuild index without removed items
    d = index.d  # dimension
    vectors = np.zeros((len(keep_faiss_ids), d), dtype='float32')
    new_id_map = {}
    new_docstore_docs = {}

    for new_idx, old_faiss_id in enumerate(keep_faiss_ids):
        vectors[new_idx] = index.reconstruct(old_faiss_id)
        old_doc_id = id_map[old_faiss_id]
        new_id_map[new_idx] = old_doc_id
        new_docstore_docs[old_doc_id] = docstore.search(old_doc_id)

    new_index = faiss.IndexFlatL2(d)
    if len(vectors) > 0:
        new_index.add(vectors)

    faiss.write_index(new_index, str(faiss_path))

    from langchain_community.docstore.in_memory import InMemoryDocstore
    new_docstore = InMemoryDocstore(new_docstore_docs)
    with open(pkl_path, 'wb') as f:
        pickle.dump((new_docstore, new_id_map), f)

    logger.info(f"Removed {remove_count} chunks for {zotero_key}")
    return remove_count


# ── Main Sync ──

def sync_v2() -> str:
    """Option E: Hybrid incremental sync with per-item tracking."""

    logger.info("=" * 60)
    logger.info("Zotero Sync v2 (Hybrid) starting")
    logger.info("=" * 60)

    state = load_state()
    embeddings = get_embeddings()
    start_time = datetime.now()

    # Stats
    stats = {
        'skipped': 0,
        'metadata_patched': 0,
        'full_processed': 0,
        'notes_synced': 0,
        'chunks_added': 0,
        'chunks_patched': 0,
        'errors': []
    }

    # ── Step 1: Fetch modified items ──
    try:
        if state['last_synced']:
            logger.info(f"Incremental sync from version {state['last_synced']}")
            all_items = zot.everything(zot.items(since=state['last_synced']))
        else:
            logger.info("Initial sync - fetching all items")
            all_items = zot.everything(zot.items())
    except Exception as e:
        return f"❌ Zotero API error: {e}"

    research_types = ['journalArticle', 'conferencePaper', 'book', 'bookSection', 'report', 'thesis']
    items = [i for i in all_items if i['data'].get('itemType') in research_types]
    standalone_notes = [i for i in all_items if i['data'].get('itemType') == 'note' and not i['data'].get('parentItem')]

    logger.info(f"Fetched {len(items)} research items + {len(standalone_notes)} standalone notes")

    if not items and not standalone_notes:
        # Update version even if nothing to process (avoids re-fetching)
        state['last_synced'] = zot.last_modified_version()
        save_state(state)
        return "✅ No new items to sync"

    # ── Step 2: Load FAISS index ──
    lit_path = DB_PATH / "literature"
    if lit_path.exists():
        db = FAISS.load_local(str(lit_path), embeddings, allow_dangerous_deserialization=True)
    else:
        db = None

    # ── Step 3: Process research items ──
    new_docs_buffer = []  # Buffer for batched FAISS writes
    processed_titles = []

    for i, item in enumerate(items, 1):
        zkey = item['key']
        zversion = item['version']
        title = item['data'].get('title', 'Untitled')[:60]

        try:
            prev = state['item_versions'].get(zkey, {})
            prev_version = prev.get('zotero_version')
            prev_pdf_hash = prev.get('pdf_hash')

            # ── Decision: skip / metadata-patch / full-process ──

            if prev_version and prev_version == zversion:
                # Exact same version — skip entirely
                stats['skipped'] += 1
                logger.info(f"[{i}/{len(items)}] SKIP (same version): {title}")
                continue

            # Check PDF: is content actually different?
            cached_pdf = PDF_CACHE / f"{zkey}.pdf"
            current_pdf_hash = None
            if cached_pdf.exists():
                current_pdf_hash = pdf_content_hash(str(cached_pdf))

            metadata_changed_only = (
                prev_pdf_hash is not None
                and current_pdf_hash is not None
                and prev_pdf_hash == current_pdf_hash
            )

            if prev_version and metadata_changed_only:
                # PDF unchanged — metadata-only patch (tags, citations, etc.)
                new_meta = extract_metadata(item)
                patched = patch_faiss_metadata(lit_path, zkey, new_meta)
                stats['metadata_patched'] += 1
                stats['chunks_patched'] += patched
                logger.info(f"[{i}/{len(items)}] METADATA PATCH ({patched} chunks): {title}")

                # Update state for this item
                state['item_versions'][zkey] = {
                    'zotero_version': zversion,
                    'pdf_hash': current_pdf_hash,
                    'chunks': prev.get('chunks', 0),
                    'last_processed': datetime.now().isoformat()
                }

            else:
                # New item OR PDF content changed — full pipeline
                logger.info(f"[{i}/{len(items)}] FULL PROCESS: {title}")

                # Remove old chunks if replacing
                if prev_version and db is not None:
                    remove_chunks_by_key(lit_path, zkey)
                    # Reload db after removal
                    db = FAISS.load_local(str(lit_path), embeddings, allow_dangerous_deserialization=True)

                docs = full_process_item(item)
                new_docs_buffer.extend(docs)
                stats['full_processed'] += 1
                stats['chunks_added'] += len(docs)
                processed_titles.append(title)

                # Compute PDF hash for new/downloaded PDF
                if cached_pdf.exists():
                    current_pdf_hash = pdf_content_hash(str(cached_pdf))

                state['item_versions'][zkey] = {
                    'zotero_version': zversion,
                    'pdf_hash': current_pdf_hash,
                    'chunks': len(docs),
                    'last_processed': datetime.now().isoformat()
                }

            # ── Checkpoint: save every BATCH_SIZE items ──
            if i % BATCH_SIZE == 0:
                if new_docs_buffer:
                    logger.info(f"Checkpoint: saving {len(new_docs_buffer)} chunks to FAISS...")
                    if db is not None:
                        db.add_documents(new_docs_buffer)
                    else:
                        db = FAISS.from_documents(new_docs_buffer, embeddings)
                    db.save_local(str(lit_path))
                    new_docs_buffer = []

                state['last_synced'] = zot.last_modified_version()
                save_state(state)
                logger.info(f"Checkpoint saved at item {i}/{len(items)}")

        except Exception as e:
            logger.error(f"Error processing {zkey}: {e}")
            stats['errors'].append(f"{title}: {e}")

    # ── Step 4: Flush remaining buffer ──
    if new_docs_buffer:
        logger.info(f"Final flush: {len(new_docs_buffer)} chunks to FAISS...")
        if db is not None:
            db.add_documents(new_docs_buffer)
        else:
            db = FAISS.from_documents(new_docs_buffer, embeddings)
        db.save_local(str(lit_path))

    # ── Step 5: Standalone notes → mind ──
    if standalone_notes:
        from zotero_tools import ZoteroSync
        syncer = ZoteroSync()
        note_docs = []
        for note in standalone_notes:
            try:
                docs = syncer.process_standalone_note(note)
                note_docs.extend(docs)
                stats['notes_synced'] += 1
            except Exception as e:
                stats['errors'].append(f"Note: {e}")

        if note_docs:
            mind_path = DB_PATH / "mind"
            if mind_path.exists():
                mem_db = FAISS.load_local(str(mind_path), embeddings, allow_dangerous_deserialization=True)
                mem_db.add_documents(note_docs)
            else:
                mem_db = FAISS.from_documents(note_docs, embeddings)
            mem_db.save_local(str(mind_path))

    # ── Step 6: Final state save ──
    state['last_synced'] = zot.last_modified_version()
    state['sync_history'].append({
        'timestamp': datetime.now().isoformat(),
        'stats': stats
    })
    # Keep only last 20 history entries
    state['sync_history'] = state['sync_history'][-20:]
    save_state(state)

    # ── Result ──
    duration = (datetime.now() - start_time).total_seconds()
    result = f"✅ Sync v2 complete in {duration:.1f}s\n\n"
    result += f"📊 {len(items)} items checked:\n"
    result += f"  • {stats['skipped']} skipped (unchanged)\n"
    result += f"  • {stats['metadata_patched']} metadata-only patches ({stats['chunks_patched']} chunks)\n"
    result += f"  • {stats['full_processed']} full processed ({stats['chunks_added']} chunks added)\n"

    if stats['notes_synced']:
        result += f"  • {stats['notes_synced']} standalone notes synced\n"

    if processed_titles:
        result += f"\nNew/changed papers:\n"
        for t in processed_titles[:5]:
            result += f"  • {t}\n"
        if len(processed_titles) > 5:
            result += f"  • ...and {len(processed_titles) - 5} more\n"

    if stats['errors']:
        result += f"\n⚠️ {len(stats['errors'])} errors:\n"
        for e in stats['errors'][:3]:
            result += f"  • {e}\n"

    logger.info(result)
    return result


if __name__ == "__main__":
    print(sync_v2())
