---
name: megabrain
description: Megabrain - Unified knowledge system with Zotero research (FAISS literature/), personal notes (mind/), second brain. Voyage embeddings + rerank, hybrid RAG synthesis (thematic + full source appendix), auto-sync.
homepage: https://www.zotero.org/support/dev/web_api/v3/start
metadata:
  openclaw:
    emoji: "🧠"
    requires: { bins: ["python3"], env: ["OPENAI_API_KEY"] }
---

# Megabrain v3.1.1

Unified knowledge system: Zotero research papers + personal notes + general interests. **FAISS-powered**, Voyage AI embeddings/rerank, **hybrid RAG synthesis** (detailed themes + full source appendix).

## 📋 Recent Updates (v3.1.1 - 2026-02-23)

**Documentation Sync**: All documentation updated to reflect v3.1 unified API
- README.md rewritten to v3.1.1 standards (removed deprecated CLI examples)
- SKILL.md verified against actual codebase
- TOOLS_SUMMARY.md updated with current API
- All version references synchronized
- Zero code changes - documentation-only release
- See `CHANGELOG.md` for complete version history

**v3.1 (2026-02-22)**: Unified API + Mind Partition Rename
- `add_mind()` (renamed from `add_memory` for clarity)
- Partition-specific search: `search_library()`, `search_mind()`, `search_brain()`
- `universal_search()` - cross-partition search
- Mind partition rename: `memories/` → `mind/`

## What It Does
- 🔍 **Unified API**: Clean partition-aware tools (`add_library`, `add_memory`, `add_brain`)
- 📚 **library/**: Research documents (Zotero auto-sync + manual PDFs/reports)
- 🧩 **mind/**: Personal notes, insights, opinions (with rich metadata)
- 🧠 **second_brain/**: General interests, web clippings, hobbies
- 🤖 **Hybrid RAG**: Thematic synthesis + cross-source analysis + full source appendix
- 🚀 **Voyage AI**: voyage-2 embeddings + rerank-2 (top-100 → top-15)

## Quick Start
```
bash /home/san/.openclaw/workspace/zotero-sync.sh
```

## ⚠️ CRITICAL: Execution Rules
**NEVER use inline chaining (`&&`, `||`, `;`) or `python -c` in exec calls.**
OpenClaw's exec layer HTML-escapes `&&` → `;&` → syntax errors.

**ALWAYS wrap commands in a bash script:**
```bash
# Generic megabrain runner (use for ALL megabrain Python calls):
bash /home/san/.openclaw/workspace/megabrain-run.sh "from zotero_tools import zotero_search; print(zotero_search('query'))"

# Or for specific tools:
bash /home/san/.openclaw/workspace/zotero-sync.sh
```

**NEVER do this:**
```bash
# ❌ BROKEN - will fail with syntax error
cd ~/.openclaw/workspace/personal-skills/megabrain && source venv/bin/activate && python -c "..."
```

## Tools (Unified API v3.1)

### 📚 Library Partition (Research: Zotero + Manual)

#### `add_library(content, title, authors, year, metadata) → str`
Manually add research document to library/ (alongside Zotero papers).

**Args:**
- `content` (str): Full text or abstract
- `title` (str): Document title
- `authors` (str): Comma-separated author names
- `year` (int): Publication year
- `metadata` (dict): **REQUIRED fields:**
  - `methodology`: "qualitative|quantitative|mixed|conceptual|review|design-science"
  - `topic`: Primary research topic
  - Optional: `subtopics`, `publication`, `doi`, `source_type` (pdf|report|thesis|preprint)

**Example:**
```python
add_library(
    content="Abstract: This paper examines platform strategies...",
    title="Platform Strategy in Digital Markets",
    authors="Smith, J., Lee, K.",
    year=2024,
    metadata={
        'methodology': 'qualitative',
        'topic': 'platform strategy',
        'subtopics': 'digital markets,network effects',
        'publication': 'Strategic Management Journal',
        'doi': '10.1234/smj.2024.001'
    }
)
```

#### `search_library(query: str, where: dict = None) → str` / `zotero_search(query: str, where: dict = None) → str`
Search library/ partition (Zotero + manual research). Hybrid RAG synthesis with themes + full appendix.

**Optional `where` parameter for metadata filtering:**
```python
# Filter by author (substring match)
zotero_search("knowledge transfer", where={"authors": {"$contains": "Kathuria"}})

# Filter by year (>=)
zotero_search("boundary objects", where={"year": {"$gte": 2020}})

# Filter by methodology (exact)
zotero_search("digital transformation", where={"methodology": "qualitative"})

# Multiple filters (AND)
zotero_search("platforms", where={"authors": {"$contains": "Tiwana"}, "year": {"$gte": 2015}})
```

**Supported operators:**
- `{"field": "value"}` — Exact match (case-insensitive)
- `{"field": {"$eq": "value"}}` — Exact match
- `{"field": {"$contains": "substring"}}` — Substring match (case-insensitive)
- `{"field": {"$gt": number}}` — Greater than
- `{"field": {"$gte": number}}` — Greater than or equal
- `{"field": {"$in": ["v1", "v2"]}}` — Value in list

**Common filterable fields:** `authors`, `year`, `title`, `journal`, `methodology`, `topic`, `tags`

**Output Structure:**
1. **Thematic Analysis** (detailed evidence per theme/source)
2. **Cross-Synthesis** (interactions, gaps)
3. **Implications**
4. **Summary**
5. **Appendix** (all 15 sources: title/authors/year/method/contrib/limits)
6. **Full Biblio** + Zotero links

#### `zotero_sync() → str`
Incremental sync: papers → library/, standalone notes → mind/.

#### `zotero_sync_notes() → str`
Backfill ALL notes (one-time): item notes → library/, standalone → mind/.

---

### 🧩 Mind Partition (Personal Notes)

#### `add_mind(content: str, metadata: dict) → str`
Add personal note to mind/ with structured metadata.

**Args:**
- `content` (str): Note text
- `metadata` (dict): **REQUIRED field:**
  - `topic`: Primary topic
  - Optional: `subtopics`, `context`, `sentiment` (supportive|critical|exploratory|neutral), `supersedes`, `action_items`, `related_to`

**Example:**
```python
add_mind(
    content="Prof Raj mentioned platform strategy connects to PSF value creation...",
    metadata={
        'topic': 'PSF',
        'subtopics': 'value creation,platform strategy',
        'context': 'Class discussion Feb 2026',
        'sentiment': 'exploratory'
    }
)
```

> **Note**: `add_memory()` is deprecated. Use `add_mind()` to avoid confusion with workspace `MEMORY.md`.

#### `add_note(note: str, title: str = None, metadata: dict = None) → str`
Simplified add_mind (auto-infers topic if missing).

#### `search_mind(query: str, top_k: int = 10) → str`
Search mind/ partition with Voyage rerank. Returns formatted results with metadata (topic, context, date).

---

### 🧠 Second Brain Partition (General Interests)

#### `add_brain(content, category, tags, source_author=None, title=None, metadata=None) → str`
Add general interest content to second_brain/.

**Args:**
- `content` (str): Text content
- `category` (str): **REQUIRED** (e.g., 'tech', 'philosophy', 'cooking')
- `tags` (str): **REQUIRED** comma-separated keywords
- `source_author` (str): Original content creator
- `title` (str): Optional title
- `metadata` (dict): Optional (topic, subtopics, related_to)

**Example:**
```python
add_brain(
    content="Interesting blog post on AI agents...",
    category='tech',
    tags='AI,agents,automation',
    source_author='John Doe',
    title='The Future of AI Agents'
)
```

#### `search_brain(query: str, category: str = None, top_k: int = 8) → str`
Search second_brain/ partition with optional category filter and Voyage rerank.

---

### 🔍 Universal Search

#### `universal_search(query: str, top_k: int = 8) → str`
Search all three partitions (library/ + mind/ + second_brain/) simultaneously. Returns top-k results ranked globally across all sources.

---

### 🔧 Legacy Tools (Backward Compatibility)

- `add_url(url, title=None)` → Extract + add URL to mind/
- `add_video(url, title=None)` → YouTube transcript → mind/

## Output Example (Hybrid Format)
```
PART 1: THEMATIC ANALYSIS
Theme 1: Social Influence
  - [Venkatesh & Davis, 2000] — compliance/internalization (longitudinal, n=156)
  - [Aydin & Rice, 1991] — networks (qual case, 12 hospitals)

PART 2: SYNTHESIS
Interactions: social moderates PU...

PART 3: IMPLICATIONS
Leverage champions + TOE fit...

PART 4: SUMMARY

APPENDIX: [15 full source summaries]

SOURCES: [Biblio + Zotero links]
```

## Architecture
```
Zotero API ──> FAISS Triple Index ──> Voyage Rerank ──> gpt-4o-mini Hybrid Prompt ──> Chat
  papers/notes     literature/     top-100→15      Thematic + Appendix
                 mind/                        
                 second_brain/                   
```

## Auto-Sync
**Cron Active**: Hourly Zotero sync (ID: 6925aa7d-...).

## Config (`config.json`)
```
zotero: {user_id: "...", api_key: "..."}
voyage: {api_key: "...", embedding_model: "voyage-2"}
llm: {model: "gpt-4o-mini", api_key_env: "OPENAI_API_KEY", temperature: 0.4}
db_path: "~/.openclaw/workspace/research-vector-db"
```

## Performance
- Query: 5-10s
- Sync: 386 papers indexed
- Storage: FAISS local persistence

## Response Formatting Rules (Mandatory) — LOCKED IN 2026-02-23

### STRUCTURED 4-PART FORMAT (User-Approved)

**ALL search responses use this structure** (500-800 words total):

```
**Core Concept:** One-sentence definition answering the question directly.

**PART 1: THEMATIC ANALYSIS**

**Theme N: [Theme Name]**
• Brief explanation (1-2 sentences)
• Full citation: Author(s) (Year). Title. Journal, Volume(Issue), Pages. DOI: xxx
• Evidence: Key finding with [Author, Year] inline citation

**Library Ties:** Connect to other papers in user's indexed library

**PART 2: SYNTHESIS**
• **Interactions:** How themes connect
• **Gaps:** What's missing from the literature

**PART 3: IMPLICATIONS**
• Bullet points with actionable insights
• How to apply, measure, or avoid pitfalls

**PART 4: SUMMARY**
One punchy sentence capturing the core takeaway.

**APPENDIX (Core Sources)**
N. Author(s) (Year) – Journal. Method: X. Contrib: Y. DOI: xxx

---

📚 ALL REFERENCES FROM LIBRARY (Top 20)
• **Full Title**
  Authors (Year) — Journal [citations count]
  DOI: xxx

🧠 MIND PARTITION
Notes shown only if relevance threshold met (distance < 0.4), otherwise "No relevant notes found."
```

**Key features**: Numbered themes, full citations with DOI, Library Ties section, APPENDIX with method/contrib, punchy summary.

**Full details file**: If >10 sources retrieved, creates `search_results_[timestamp].md` with all 20 sources + full text chunks.

**⚠️ DO NOT CHANGE THIS FORMAT** without explicit user request.

### 1. Enumeration Requests ("Show me papers by X", "Find papers on Y")
- List each paper: **Title**, Authors, Year, Publication/Journal, one-line summary
- Include DOI/Zotero link if available
- Do NOT synthesize themes unless asked

### 2. Citation Rules
- **Every claim** needs [Author, Year] citation
- Never use anonymous sources (e.g., "[Source 5]" alone)
- If metadata incomplete, flag explicitly

### 3. Legacy Large Result Sets
- If enumeration exceeds one message: "Showing X of Y results. Say 'continue' for more."

## Mind Partition Interaction Rules (Mandatory)

**What is "Mind"?**  
The mind partition stores personal notes, insights, and user-contributed content. It's a personal knowledge graph distinct from:
- **Literature** (📚) - Academic papers from Zotero
- **Second Brain** (🧠) - Web clippings, hobbies, casual interests

When a user asks about their mind (e.g., "what's in my mind", "what did I save"):

### 1. Topic Discovery Phase
- Assume the user is the **creator/author** of everything in the mind partition.
- Ask if they have a **specific topic** in mind.
- If not, display a **numbered list of topics** (inferred from note titles, categories, tags) they can choose from.
  - E.g.: `1. AI & Technology  2. Prof Raj Projects  3. Methodology Preferences`

### 2. Knowledge Synthesis Phase (once topic chosen)
- Tell the user what they know / what they should remember about the topic.
- Provide **rich contextual detail**:
  - **Temporal context**: "In February 2026, in this note you said..."
  - **Verbatim quotes**: "Some of the lines you wrote include: '...'"
  - **Evolution of thinking**: "While in note X you wrote A, later in note Y you shifted to B."
  - **Connections**: Link related notes/topics together.
- Always offer: "Do you want me to pull any of this note out for you?"

### 3. Knowledge Graph Approach
- Treat mind as a **knowledge graph**, not a flat list.
- Show connections between notes, evolution over time, and thematic clusters.
- Surface contradictions or shifts in perspective explicitly.

## Structured Tags Schema (All Partitions)

All documents ingested into any Megabrain partition MUST include structured metadata tags for knowledge graph navigation, topic discovery, and cross-partition linking.

### Shared Fields (all partitions)
| Field | Type | Required | Description |
|---|---|---|---|
| `topic` | str | ✅ | Primary topic (e.g., "AI", "methodology", "platforms") |
| `subtopics` | str (comma-sep) | ✅ | Secondary topics/themes |
| `related_to` | str (comma-sep) | Optional | Titles/IDs of related items (cross-partition OK) |

### Literature Partition (additional to existing Zotero metadata)
| Field | Type | Required | Description |
|---|---|---|---|
| `methodology` | str | ✅ | "qualitative", "quantitative", "mixed", "conceptual", "review", "design-science" |
| `topic` | str | ✅ | Inferred from existing Zotero tags (top-level grouping) |
| `subtopics` | str | ✅ | Inferred from existing Zotero tags (secondary themes) |

### Mind Partition
| Field | Type | Required | Description |
|---|---|---|---|
| `topic` | str | ✅ | Primary topic |
| `subtopics` | str | ✅ | Secondary themes |
| `context` | str | ✅ | When/why written (e.g., "Prof Raj class", "reading session") |
| `sentiment` | str | Optional | Author's stance: "supportive", "critical", "exploratory", "neutral" |
| `supersedes` | str | Optional | Title/ID of note this updates or contradicts |
| `action_items` | str | Optional | Embedded todos (comma-separated) |
| `related_to` | str | Optional | Cross-links to other notes/papers |

### Second Brain Partition
| Field | Type | Required | Description |
|---|---|---|---|
| `topic` | str | ✅ | Primary topic |
| `subtopics` | str | ✅ | Secondary themes |
| `category` | str | ✅ | Enforced (e.g., "tech", "philosophy", "cooking") |
| `tags` | str | ✅ | Enforced (comma-separated keywords) |
| `source_author` | str | ✅ | Original content creator |
| `related_to` | str | Optional | Cross-links |

### Auto-Inference Rules
- On ingest: use LLM (gpt-4o-mini) to infer `topic`, `subtopics`, `methodology` from content when not provided by user.
- On literature sync: derive `topic`/`subtopics` from existing Zotero `tags` field.
- Prompt user for missing required fields only when auto-inference confidence is low.
- Backfill script: `tag_enrichment.py` for existing documents.

## Content Extraction Rules (Mandatory)
When a user provides a URL or file to add to any partition:

### 1. Authentication-Gated Content (iCloud, Google Drive private, Notion, etc.)
- **IMMEDIATELY inform the user** that the content cannot be extracted due to login requirements.
- **Do NOT silently save a placeholder.** The user must know nothing was captured.
- **Offer alternatives:**
  - Paste the content directly in chat
  - Export as PDF/text and share as attachment
  - Use browser tool (if Chrome relay is attached) to access authenticated pages

### 2. Extraction Fails for Other Reasons (format, encoding, paywall, etc.)
- **Diagnose the cause** before giving up.
- **Find solutions**, including:
  - Alternative extraction tools (pdftotext, browser, web_fetch with different modes)
  - Skills that might help (mcporter, browser relay, etc.)
  - Manual workarounds (screenshot + OCR, copy-paste)
- **Present choices to the user** with clear pros/cons.
- **Never silently store an empty placeholder and move on.**

### 3. General Rule
- A document is only "added" when **actual content** is indexed and searchable.
- Link-only placeholders are **not** valid additions — they must be flagged as incomplete.

## Troubleshooting
- **No results**: Run `zotero_sync()`
- **Chroma error**: Fixed (FAISS migration complete)
- **LLM fail**: Check API keys (OpenAI/Voyage)

## Future
- Obsidian/Notion sync
- Citation graphs
- Multi-LLM support

**Status:** ✅ Production | **v3.0 Hybrid** (2026-02-19)
**Maintainer:** Percy
