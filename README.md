# Megabrain v3.2.0

A unified knowledge management system: research papers, personal notes, and general interests — all searchable with natural language.

**Version:** 3.2.0 (2026-02-23)  
**Engine:** FAISS + Voyage AI embeddings + reranking  
**Synthesis:** Venice AI (grok-41-fast)

---

## 🚀 Quick Start (New Users)

### Step 1: Get Your API Keys

You'll need three API keys:

| Service | Purpose | Get it at |
|---------|---------|-----------|
| **Zotero** | Sync your research library | https://www.zotero.org/settings/keys |
| **Voyage AI** | Embeddings + reranking | https://www.voyageai.com/ |
| **Venice AI** | LLM synthesis | https://venice.ai/ |

### Step 2: Install

```bash
# Navigate to skill directory
cd ~/.openclaw/workspace/skills/megabrain

# Run setup (creates venv, installs dependencies)
bash setup.sh
```

### Step 3: Configure

```bash
# Copy template to config
cp config.json.template config.json

# Edit config.json with your Zotero user ID
# Find your user ID at: https://www.zotero.org/settings/keys
```

**config.json** — Edit these fields:
```json
{
  "zotero": {
    "user_id": "YOUR_ZOTERO_USER_ID",  ← Replace this
    "api_key_env": "ZOTERO_API_KEY",
    "library_type": "user"
  }
}
```

### Step 4: Set Environment Variables

```bash
# Add to ~/.bashrc (or ~/.zshrc)
export ZOTERO_API_KEY="your-zotero-api-key"
export VOYAGE_API_KEY="your-voyage-api-key"
export VENICE_API_KEY="your-venice-api-key"

# Reload shell
source ~/.bashrc
```

### Step 5: Initial Sync (Build Your Index)

This downloads your Zotero papers, extracts text, and creates the FAISS index:

```bash
# First-time sync (takes 5-15 min depending on library size)
bash run.sh "zotero_sync()"
```

**What happens:**
1. Connects to Zotero API
2. Downloads PDF attachments
3. Extracts text from PDFs
4. Creates Voyage AI embeddings (1536-dim vectors)
5. Builds FAISS index at `~/.openclaw/workspace/research-vector-db/literature/`

### Step 6: Test Your Setup

```bash
# Search your library
bash run.sh "print(zotero_search('digital transformation'))"
```

If you see a structured response with themes and citations, you're ready! 🎉

---

## 📖 How It Works

Megabrain has **three partitions** (separate FAISS indexes):

| Partition | Content | Use Case |
|-----------|---------|----------|
| 📚 **library/** | Research papers (Zotero sync) | Academic research, lit reviews |
| 🧩 **mind/** | Personal notes, insights | Class notes, meeting notes, ideas |
| 🧠 **second_brain/** | General interests | Web clippings, articles, hobbies |

**Universal search** queries all three simultaneously.

---

## 🔧 Core Commands

### Wrapper Scripts

The package includes two wrapper scripts for easy execution:

**run.sh** — Execute any Python code:
```bash
bash run.sh "print(zotero_search('supply chain'))"
bash run.sh "print(universal_search('productivity'))"
bash run.sh "zotero_sync()"
```

**sync.sh** — Quick Zotero sync:
```bash
bash sync.sh
```

### API Functions

#### Library (Research Papers)

```python
# Search with RAG synthesis
zotero_search("digital transformation frameworks")
search_library("supply chain resilience")  # alias

# Search with author filter
zotero_search("IT investment", where={"authors": {"$contains": "Kathuria"}})

# Search with year filter
zotero_search("platforms", where={"year": {"$gte": 2020}})

# Sync new papers from Zotero
zotero_sync()

# Sync Zotero annotations/notes
zotero_sync_notes()
```

#### Mind (Personal Notes)

```python
# Add a personal note
add_mind(
    content="Prof mentioned PSF value creation links to platform strategy",
    metadata={
        'topic': 'PSF',
        'subtopics': 'value creation,platforms',
        'context': 'Class discussion Feb 2026'
    }
)

# Search notes
search_mind("platform strategy")
```

#### Second Brain (General Interests)

```python
# Add content
add_brain(
    content="Interesting blog on Stoic philosophy...",
    category='philosophy',
    tags='stoicism,wisdom'
)

# Search
search_brain("stoicism", category='philosophy')
```

#### Universal Search

```python
# Search all three partitions at once
universal_search("productivity techniques")
```

---

## 📋 Response Format (Library)

Library searches return a **structured 4-part analysis**:

```
Core Concept: [One-sentence definition]

PART 1: THEMATIC ANALYSIS
Theme 1: [Theme Name]
• Explanation
• Full citation: Author(s) (Year). Title. Journal. DOI: xxx
• Evidence: [Author, Year] finding...

Library Ties: Connects to [Other Paper] on [topic]...

PART 2: SYNTHESIS
• Interactions: How themes connect
• Gaps: What's missing

PART 3: IMPLICATIONS
• Actionable insights
• How to apply/measure

PART 4: SUMMARY
[One punchy sentence]

APPENDIX (Core Sources)
1. Author (Year) – Journal. Method: X. Contrib: Y. DOI: xxx

📚 ALL REFERENCES FROM LIBRARY
• Full Title — Authors (Year) — Journal [citations]
```

---

## 🔍 Metadata Filtering

Filter search results by metadata:

```python
# By author (substring match)
zotero_search("knowledge", where={"authors": {"$contains": "Argote"}})

# By year
zotero_search("AI", where={"year": {"$gte": 2023}})

# By year range
zotero_search("platforms", where={"year": {"$gte": 2018, "$lte": 2022}})

# Multiple filters (AND)
zotero_search("digital", where={
    "authors": {"$contains": "Tiwana"},
    "year": {"$gte": 2015}
})
```

**Supported operators:**
- `$contains` — Substring match (case-insensitive)
- `$eq` — Exact match
- `$gt` / `$gte` — Greater than (for numbers)
- `$in` — Value in list

**Filterable fields:** `authors`, `year`, `title`, `journal`, `methodology`, `tags`

---

## 📁 File Structure

After installation, your directory looks like:

```
skills/megabrain/
├── run.sh                   # ⭐ Main execution wrapper
├── sync.sh                  # Quick sync wrapper
├── setup.sh                 # Installation script
├── config.json.template     # Configuration template
├── config.json              # Your config (git-ignored)
├── requirements.txt         # Python dependencies
├── venv/                    # Virtual environment (created by setup.sh)
│
├── __openclaw__.py          # OpenClaw integration
├── megabrain_tools.py       # Unified API
├── zotero_tools.py          # Core search + sync
├── second_brain.py          # Second brain partition
├── user_content.py          # URL/video extraction
├── sync_v2.py               # Sync utilities
├── tag_inference.py         # Tag inference
│
├── SKILL.md                 # OpenClaw skill documentation
├── README.md                # This file
└── CHANGELOG.md             # Version history

~/.openclaw/workspace/research-vector-db/   # Created on first sync
├── literature/              # Research papers index
├── mind/                    # Personal notes index
└── second_brain/            # General interests index
```

---

## ⚙️ Configuration Reference

**config.json.template:**
```json
{
  "zotero": {
    "user_id": "YOUR_ZOTERO_USER_ID",
    "api_key_env": "ZOTERO_API_KEY",
    "library_type": "user"
  },
  "llm": {
    "model": "grok-41-fast",
    "api_key_env": "VENICE_API_KEY",
    "base_url": "https://api.venice.ai/v1",
    "temperature": 0.4
  },
  "voyage": {
    "api_key_env": "VOYAGE_API_KEY",
    "embedding_model": "voyage-2"
  },
  "sync": {
    "batch_size": 50,
    "download_pdfs": true,
    "pdf_cache": "~/.openclaw/workspace/zotero-pdfs"
  },
  "chromadb": {
    "persist_directory": "~/.openclaw/workspace/research-vector-db"
  }
}
```

**Environment variables:**
```bash
export ZOTERO_API_KEY="..."      # Zotero API key
export VOYAGE_API_KEY="..."      # Voyage AI key
export VENICE_API_KEY="..."      # Venice AI key (for LLM)
```

---

## 🔄 Keeping Your Index Updated

### Manual Sync
```bash
bash sync.sh
# or
bash run.sh "zotero_sync()"
```

### Automated Sync (Cron)

Set up hourly sync via OpenClaw cron:
```
Schedule a cron job: every hour, run zotero_sync()
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
cd ~/.openclaw/workspace/skills/megabrain
bash setup.sh  # Reinstall dependencies
```

### "No results found"
```bash
bash sync.sh  # Sync latest papers
```

### "API key not found"
```bash
# Check env vars are set
echo $ZOTERO_API_KEY
echo $VOYAGE_API_KEY
echo $VENICE_API_KEY

# If empty, add to ~/.bashrc and reload
source ~/.bashrc
```

### "Database not initialized"
```bash
# Run initial sync to create FAISS indexes
bash run.sh "zotero_sync()"
```

### "config.json not found"
```bash
cp config.json.template config.json
# Then edit with your Zotero user ID
```

---

## 📊 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Single search | 300-500ms | FAISS + rerank |
| RAG synthesis | 5-10s | LLM bottleneck |
| Incremental sync | ~45s | When no new papers |
| Full sync (100 papers) | 5-10 min | First-time only |

---

## 📚 Documentation

- **SKILL.md** — Full API reference for OpenClaw
- **CHANGELOG.md** — Version history
- **config.json.template** — Configuration reference

---

## 🆕 What's New in v3.2.0

- **4-PART structured output** — Core Concept → Themes → Synthesis → Implications → Summary → Appendix
- **Metadata filtering** — Filter by author, year, journal with `where` parameter
- **Full citations** — Journal, DOI, method, contribution in output
- **Library Ties** — Connections to other papers in your index
- **Mind relevance threshold** — Only shows notes if highly relevant (distance < 0.4)

---

## 📧 Support



---

*Megabrain v3.2.0 — Your research, notes, and interests in one searchable brain.*
