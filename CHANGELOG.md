# Megabrain Changelog

## v3.1.1 (2026-02-23)

### Security: API Keys → Environment Variables

**Breaking Change (Security Improvement)**:
- ✅ **API keys moved to environment variables**
  - `ZOTERO_API_KEY` - Zotero API key
  - `VOYAGE_API_KEY` - Voyage AI API key
  - `VENICE_API_KEY` - Venice.ai LLM key (already using env)
- ✅ `config.json` now references env var names via `*_env` fields
- ✅ Backward compatible: still checks for direct `api_key` with deprecation warning
- ✅ Added `config.json.template` for easy setup

**Migration**:
```bash
# Set environment variables
export ZOTERO_API_KEY="your-key"
export VOYAGE_API_KEY="your-key"

# Update config.json to use *_env fields (see config.json.template)
```

**Files Updated**:
- `zotero_tools.py` - Added `get_api_key()` helper, uses env vars
- `megabrain_tools.py` - Uses `VOYAGE_API_KEY` from env
- `second_brain.py` - Uses `VOYAGE_API_KEY` from env
- `sync_v2.py` - Uses both `ZOTERO_API_KEY` and `VOYAGE_API_KEY` from env
- `config.json` - Refactored to reference env vars
- `config.json.template` - Template for new installations
- `README.md` - Updated configuration section

### Documentation & Cleanup (Feb 23, 09:25)

**Changes**:
- Updated all core files with latest improvements
- Cleaned up zip package (removed search results and log files)
- All documentation synchronized with current implementation
- `SKILL.md` - Updated API reference
- `SEARCH-ARCHITECTURE.md` - Complete technical docs
- `MIGRATION-GUIDE.md` - Migration instructions

---

## v3.1 (2026-02-22)

### Major Refactor: Unified API

**Breaking Changes**: None (backward compatible)

**New Features**:
- **`megabrain_tools.py`** — Unified API with clean partition-aware naming
  - `add_library(...)` — Manually add research docs to library/
  - **`add_mind(...)`** — Add personal notes (renamed from `add_memory` to avoid MEMORY.md confusion)
  - `add_brain(...)` — Add general interests
  - `search_library(...)`, `search_mind(...)`, `search_brain(...)` — Partition-specific search
  - `universal_search(...)` — Cross-partition search

**Architecture Cleanup**:
- Removed redundant wrapper functions in `zotero_tools.py`
- Direct imports from `second_brain.py` (no more `_universal_search` aliasing)
- All search logic consolidated in `second_brain.py` and `megabrain_tools.py`

**Deprecated**:
- `add_memory()` → Use `add_mind()` instead (logs warning)
- `zotero_tools.brain_search()` → Use `megabrain_tools.search_brain()` (still works via import)

**Documentation**:
- Added `SEARCH-ARCHITECTURE.md` — Full technical documentation
- Added `MIGRATION-GUIDE.md` — Migration from v3.0
- Updated `SKILL.md` — New API reference

---

## v3.0 (2026-02-19)

### Mind Partition Rename

**Breaking Changes**: None (backward compatible)

**Changes**:
- Renamed `memories/` → `mind/` (FAISS index directory)
- Emoji: 💡 → 🧩
- Updated all metadata references
- Migration script: `migrate_mind_metadata.py`

**Rationale**: Avoid confusion with workspace `MEMORY.md` / `memory/` logs

---

## v2.x (2026-02-18)

- FAISS migration (from Chroma)
- Citation-boosted ranking
- Voyage rerank-2 integration
- Hybrid RAG output format

---

## v1.x (2026-01-XX)

- Initial Zotero integration
- Chroma vector DB
- Basic semantic search

---

**Status**: ✅ Production (v3.1)
