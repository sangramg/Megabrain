#!/usr/bin/env python3
"""
User Content Handler for Megabrain
Handles URLs, videos, audio, and direct text contributions for the mind/ index.

Key principle: ALWAYS grab full metadata (author, date, topic, description) from
the source. If metadata is unavailable, return a structured prompt asking the user
for details rather than silently storing without context.

Author: Percy (OpenClaw Agent)
Updated: 2026-02-20
"""

import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

# Setup
logger = logging.getLogger(__name__)
SKILL_DIR = Path(__file__).parent
CONFIG_PATH = SKILL_DIR / "config.json"

with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# Lazy imports
def get_embeddings():
    from zotero_tools import get_embeddings as _get_embeddings
    return _get_embeddings()

def get_memory_db():
    """Get personal knowledge FAISS collection"""
    from langchain_community.vectorstores import FAISS

    embeddings = get_embeddings()
    db_path = Path(CONFIG['chromadb']['persist_directory']).expanduser() / "mind"
    db_path.mkdir(parents=True, exist_ok=True)

    faiss_file = db_path / "index.faiss"
    if faiss_file.exists():
        logger.info("Loading existing personal knowledge FAISS index...")
        db = FAISS.load_local(str(db_path), embeddings, allow_dangerous_deserialization=True)
    else:
        logger.info("Creating new personal knowledge FAISS index...")
        dummy_doc = Document(page_content="initialized", metadata={"type": "system"})
        db = FAISS.from_documents([dummy_doc], embeddings)
        db.save_local(str(db_path))
    return db


# ── Metadata Extraction Helpers ──────────────────────────────────

def _extract_opengraph(soup: BeautifulSoup) -> dict:
    """Extract OpenGraph / meta tags from HTML."""
    og = {}
    for tag in soup.find_all('meta'):
        prop = tag.get('property', '') or tag.get('name', '')
        content = tag.get('content', '')
        if not content:
            continue
        if prop in ('og:title', 'twitter:title'):
            og.setdefault('title', content)
        elif prop in ('og:description', 'twitter:description', 'description'):
            og.setdefault('description', content)
        elif prop in ('og:site_name',):
            og['site_name'] = content
        elif prop in ('article:author', 'author'):
            og['author'] = content
        elif prop in ('article:published_time', 'datePublished', 'og:updated_time'):
            og.setdefault('date_published', content)
        elif prop in ('og:type',):
            og['og_type'] = content
        elif prop in ('og:image',):
            og['image'] = content
        elif prop in ('keywords',):
            og['keywords'] = content
    return og


def _extract_jsonld(soup: BeautifulSoup) -> dict:
    """Extract JSON-LD structured data from HTML."""
    meta = {}
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string or '{}')
            if isinstance(data, list):
                data = data[0] if data else {}
            if isinstance(data, dict):
                if data.get('name'):
                    meta.setdefault('title', data['name'])
                if data.get('headline'):
                    meta.setdefault('title', data['headline'])
                if data.get('description'):
                    meta.setdefault('description', data['description'])
                if data.get('datePublished'):
                    meta.setdefault('date_published', data['datePublished'])
                if data.get('dateCreated'):
                    meta.setdefault('date_created', data['dateCreated'])
                # Author can be string or dict
                author = data.get('author')
                if isinstance(author, dict):
                    meta['author'] = author.get('name', str(author))
                elif isinstance(author, list):
                    meta['author'] = ', '.join(
                        a.get('name', str(a)) if isinstance(a, dict) else str(a)
                        for a in author
                    )
                elif isinstance(author, str):
                    meta['author'] = author
                # Creator / publisher
                pub = data.get('publisher')
                if isinstance(pub, dict):
                    meta.setdefault('publisher', pub.get('name', ''))
                elif isinstance(pub, str):
                    meta.setdefault('publisher', pub)
                # Duration (for video/audio)
                if data.get('duration'):
                    meta['duration'] = data['duration']
                # Keywords/tags
                if data.get('keywords'):
                    kw = data['keywords']
                    meta['keywords'] = kw if isinstance(kw, str) else ', '.join(kw)
        except (json.JSONDecodeError, TypeError):
            continue
    return meta


def _get_youtube_metadata(video_id: str, url: str) -> dict:
    """Extract full YouTube video metadata using yt-dlp (no download)."""
    meta = {
        'source_type': 'video',
        'platform': 'youtube',
        'url': url,
        'video_id': video_id,
    }
    try:
        import yt_dlp
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'no_color': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info:
                meta['title'] = info.get('title', '')
                meta['author'] = info.get('uploader', info.get('channel', ''))
                meta['channel'] = info.get('channel', '')
                meta['channel_id'] = info.get('channel_id', '')
                meta['description'] = (info.get('description', '') or '')[:500]
                meta['date_published'] = info.get('upload_date', '')  # YYYYMMDD
                meta['duration_seconds'] = info.get('duration', 0)
                meta['view_count'] = info.get('view_count', 0)
                meta['like_count'] = info.get('like_count', 0)
                meta['categories'] = ', '.join(info.get('categories', []))
                meta['tags'] = ', '.join((info.get('tags', []) or [])[:10])
                meta['thumbnail'] = info.get('thumbnail', '')
                # Format duration
                dur = info.get('duration', 0)
                if dur:
                    meta['duration'] = f"{dur // 3600}h {(dur % 3600) // 60}m {dur % 60}s" if dur >= 3600 else f"{dur // 60}m {dur % 60}s"
        logger.info(f"YouTube metadata: {meta.get('title', 'N/A')} by {meta.get('author', 'N/A')}")
    except ImportError:
        logger.warning("yt-dlp not installed; metadata will be minimal")
    except Exception as e:
        logger.warning(f"yt-dlp metadata extraction failed: {e}")
    return meta


def _get_generic_audio_video_metadata(url: str) -> dict:
    """Extract metadata from non-YouTube audio/video URLs using yt-dlp."""
    meta = {
        'source_type': 'audio_video',
        'url': url,
    }
    try:
        import yt_dlp
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'no_color': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info:
                meta['title'] = info.get('title', '')
                meta['author'] = info.get('uploader', info.get('channel', info.get('creator', '')))
                meta['description'] = (info.get('description', '') or '')[:500]
                meta['date_published'] = info.get('upload_date', '')
                meta['duration_seconds'] = info.get('duration', 0)
                meta['platform'] = info.get('extractor_key', info.get('extractor', 'unknown'))
                dur = info.get('duration', 0)
                if dur:
                    meta['duration'] = f"{dur // 3600}h {(dur % 3600) // 60}m {dur % 60}s" if dur >= 3600 else f"{dur // 60}m {dur % 60}s"
        logger.info(f"A/V metadata: {meta.get('title', 'N/A')} by {meta.get('author', 'N/A')}")
    except ImportError:
        logger.warning("yt-dlp not installed for generic A/V extraction")
    except Exception as e:
        logger.warning(f"A/V metadata extraction failed: {e}")
    return meta


def _identify_missing_metadata(meta: dict) -> list:
    """Return list of critical metadata fields that are missing."""
    missing = []
    if not meta.get('title'):
        missing.append('title')
    if not meta.get('author'):
        missing.append('author/creator')
    if not meta.get('date_published') and not meta.get('date_created'):
        missing.append('date published')
    if not meta.get('description'):
        missing.append('description/abstract')
    return missing


# ── Content Extraction ───────────────────────────────────────────

def extract_url_content(url: str) -> Tuple[str, dict]:
    """Extract readable content + rich metadata from a URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; Megabrain/2.1)'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract metadata layers
        og_meta = _extract_opengraph(soup)
        jsonld_meta = _extract_jsonld(soup)

        # Merge (JSON-LD takes precedence over OG)
        merged = {**og_meta, **jsonld_meta}

        # Fallback title from <title> tag
        if not merged.get('title'):
            title_tag = soup.find('title')
            merged['title'] = title_tag.get_text(strip=True) if title_tag else url

        # Remove non-content elements
        for el in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            el.decompose()

        # Extract main content
        main = soup.find('article') or soup.find('main') or soup.find('body')
        text = main.get_text(separator='\n', strip=True) if main else ""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n\n'.join(lines)

        # Build metadata
        metadata = {
            'source_type': 'url',
            'url': url,
            'title': merged.get('title', url),
            'author': merged.get('author', ''),
            'publisher': merged.get('publisher', merged.get('site_name', '')),
            'date_published': merged.get('date_published', ''),
            'description': merged.get('description', ''),
            'keywords': merged.get('keywords', ''),
            'og_type': merged.get('og_type', ''),
            'added_at': datetime.now().isoformat(),
            'added_by': 'user',
            'char_count': len(text),
        }

        logger.info(f"URL extracted: '{metadata['title']}' by {metadata.get('author', 'N/A')} ({len(text)} chars)")
        return text, metadata

    except Exception as e:
        logger.error(f"URL extraction failed: {e}")
        raise


def extract_youtube_content(url: str) -> Tuple[str, dict, bool]:
    """
    Extract transcript + full metadata from YouTube video.

    Returns:
        (transcript_text, metadata, transcript_available)
    """
    # Extract video ID
    if 'youtu.be/' in url:
        video_id = url.split('youtu.be/')[-1].split('?')[0]
    elif 'v=' in url:
        video_id = url.split('v=')[-1].split('&')[0]
    else:
        raise ValueError(f"Cannot parse YouTube video ID from: {url}")

    # Get full metadata via yt-dlp
    metadata = _get_youtube_metadata(video_id, url)

    # Try to get transcript
    transcript_text = ""
    transcript_available = False
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([entry['text'] for entry in transcript_list])
        transcript_available = True
        metadata['transcript_length'] = len(transcript_text)
        logger.info(f"YouTube transcript: {len(transcript_text)} chars")
    except Exception as e:
        logger.warning(f"YouTube transcript unavailable: {e}")
        metadata['transcript_available'] = False

    # Prepend attribution context to transcript
    if transcript_text:
        author = metadata.get('author', metadata.get('channel', 'Unknown'))
        title = metadata.get('title', 'Untitled')
        date = metadata.get('date_published', 'Unknown date')
        prefix = f"[Transcript of '{title}' by {author}, published {date}]\n\n"
        transcript_text = prefix + transcript_text

    return transcript_text, metadata, transcript_available


def extract_audio_video_content(url: str) -> Tuple[str, dict, bool]:
    """
    Extract metadata from generic audio/video URL.
    Transcript typically not auto-available for non-YouTube.

    Returns:
        (transcript_text, metadata, transcript_available)
    """
    metadata = _get_generic_audio_video_metadata(url)
    # No auto-transcript for generic A/V
    return "", metadata, False


# ── Storage ──────────────────────────────────────────────────────

def _store_content(content: str, metadata: dict) -> str:
    """Chunk, embed, and store content in mind FAISS index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not content or len(content.strip()) < 10:
        return "⚠️ No content to store (empty or too short)."

    metadata['added_at'] = datetime.now().isoformat()
    metadata['added_by'] = 'user'

    # Auto-infer structured tags if missing
    if not metadata.get('topic') or not metadata.get('subtopics'):
        try:
            from tag_inference import infer_tags_mind
            tags = infer_tags_mind(content, metadata)
            for k in ('topic', 'subtopics', 'context', 'sentiment'):
                if tags.get(k) and not metadata.get(k):
                    metadata[k] = tags[k]
            logger.info(f"Auto-inferred tags: topic={metadata.get('topic')}")
        except Exception as e:
            logger.warning(f"Tag inference failed (non-fatal): {e}")

    # Chunk
    if len(content) > 2000:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(content)
    else:
        chunks = [content]

    docs = []
    for i, chunk in enumerate(chunks):
        chunk_meta = metadata.copy()
        chunk_meta['chunk_id'] = i
        chunk_meta['total_chunks'] = len(chunks)
        docs.append(Document(page_content=chunk, metadata=chunk_meta))

    db = get_memory_db()
    db.add_documents(docs)
    db.save_local(str(Path(CONFIG['chromadb']['persist_directory']).expanduser() / "mind"))

    title = metadata.get('title', 'Untitled')
    author = metadata.get('author', '')
    author_str = f" by {author}" if author else ""
    return f"✅ Indexed '{title}'{author_str} ({len(chunks)} chunks, {len(content)} chars)"


# ── Public API (OpenClaw Tools) ──────────────────────────────────

def add_note(note: str, title: Optional[str] = None, metadata: Optional[dict] = None) -> str:
    """Add text note to personal knowledge base with optional structured metadata."""
    base_metadata = {'source_type': 'note', 'title': title or 'User Note'}
    if metadata:
        base_metadata.update(metadata)  # Merge user-provided metadata
    return _store_content(note, base_metadata)


def add_url(url: str, title: Optional[str] = None) -> str:
    """
    Extract and add URL content to personal knowledge base.
    Returns structured prompt if critical metadata is missing.
    """
    text, metadata = extract_url_content(url)
    if title:
        metadata['title'] = title

    missing = _identify_missing_metadata(metadata)
    if missing:
        # Store what we have but flag missing fields
        result = _store_content(text, metadata)
        result += f"\n\n⚠️ **Missing metadata**: {', '.join(missing)}"
        result += "\nPlease provide the missing details so I can update the entry:"
        for field in missing:
            result += f"\n  - {field}: ?"
        return result

    return _store_content(text, metadata)


def add_video(url: str, title: Optional[str] = None) -> str:
    """
    Extract and add video content (YouTube or other) to knowledge base.
    Grabs full metadata. Asks user if transcript unavailable.
    """
    is_youtube = 'youtube.com' in url or 'youtu.be' in url

    if is_youtube:
        transcript, metadata, has_transcript = extract_youtube_content(url)
    else:
        transcript, metadata, has_transcript = extract_audio_video_content(url)

    if title:
        metadata['title'] = title

    # Extract source_author from yt-dlp metadata for structured tags
    if metadata.get('author') or metadata.get('channel'):
        metadata['source_author'] = metadata.get('author') or metadata.get('channel', '')

    # Check missing metadata
    missing = _identify_missing_metadata(metadata)

    # If no transcript, ask user
    if not has_transcript:
        result = f"📹 **Video metadata captured:**\n"
        result += f"  Title: {metadata.get('title', 'N/A')}\n"
        result += f"  Author: {metadata.get('author', 'N/A')}\n"
        result += f"  Date: {metadata.get('date_published', 'N/A')}\n"
        result += f"  Duration: {metadata.get('duration', 'N/A')}\n"
        if metadata.get('description'):
            result += f"  Description: {metadata['description'][:200]}...\n"

        # Store metadata + description even without transcript
        fallback_text = f"[Video: {metadata.get('title', 'Untitled')} by {metadata.get('author', 'Unknown')}]\n"
        fallback_text += f"Date: {metadata.get('date_published', 'Unknown')}\n"
        if metadata.get('description'):
            fallback_text += f"Description: {metadata['description']}\n"
        if metadata.get('tags'):
            fallback_text += f"Tags: {metadata['tags']}\n"

        store_result = _store_content(fallback_text, metadata)
        result += f"\n{store_result}"
        result += f"\n\n⚠️ **No transcript available.** Would you like to:"
        result += f"\n  1. Paste a transcript manually (I'll index it with full attribution)"
        result += f"\n  2. Skip transcript (metadata-only entry saved)"

        if missing:
            result += f"\n\n⚠️ **Also missing**: {', '.join(missing)}"
            for field in missing:
                result += f"\n  - {field}: ?"

        return result

    # Has transcript — store with full context
    result = _store_content(transcript, metadata)

    if missing:
        result += f"\n\n⚠️ **Missing metadata**: {', '.join(missing)}"
        result += "\nPlease provide:"
        for field in missing:
            result += f"\n  - {field}: ?"

    return result


def add_audio(url: str, title: Optional[str] = None) -> str:
    """
    Extract and add audio content (podcast, recording, etc.) to knowledge base.
    Same flow as video — grabs metadata, asks about transcript if unavailable.
    """
    transcript, metadata, has_transcript = extract_audio_video_content(url)
    metadata['source_type'] = 'audio'
    if title:
        metadata['title'] = title

    missing = _identify_missing_metadata(metadata)

    if not has_transcript:
        result = f"🎙️ **Audio metadata captured:**\n"
        result += f"  Title: {metadata.get('title', 'N/A')}\n"
        result += f"  Author: {metadata.get('author', 'N/A')}\n"
        result += f"  Date: {metadata.get('date_published', 'N/A')}\n"
        result += f"  Duration: {metadata.get('duration', 'N/A')}\n"
        if metadata.get('description'):
            result += f"  Description: {metadata['description'][:200]}...\n"

        fallback_text = f"[Audio: {metadata.get('title', 'Untitled')} by {metadata.get('author', 'Unknown')}]\n"
        fallback_text += f"Date: {metadata.get('date_published', 'Unknown')}\n"
        if metadata.get('description'):
            fallback_text += f"Description: {metadata['description']}\n"

        store_result = _store_content(fallback_text, metadata)
        result += f"\n{store_result}"
        result += f"\n\n⚠️ **No transcript available.** Would you like to:"
        result += f"\n  1. Paste a transcript manually (I'll index it with full attribution)"
        result += f"\n  2. Skip transcript (metadata-only entry saved)"

        if missing:
            result += f"\n\n⚠️ **Also missing**: {', '.join(missing)}"
            for field in missing:
                result += f"\n  - {field}: ?"

        return result

    result = _store_content(transcript, metadata)
    if missing:
        result += f"\n\n⚠️ **Missing metadata**: {', '.join(missing)}"
        for field in missing:
            result += f"\n  - {field}: ?"
    return result


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python user_content.py note 'My note text'")
        print("  python user_content.py url https://example.com")
        print("  python user_content.py video https://youtube.com/watch?v=...")
        print("  python user_content.py audio https://podcast.example.com/ep1")
        sys.exit(1)

    cmd = sys.argv[1]
    content_or_url = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else None

    if cmd == 'note':
        print(add_note(content_or_url, title))
    elif cmd == 'url':
        print(add_url(content_or_url, title))
    elif cmd == 'video':
        print(add_video(content_or_url, title))
    elif cmd == 'audio':
        print(add_audio(content_or_url, title))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
