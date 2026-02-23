# Data Flows — Megabrain v3.1

This document visualizes the end-to-end data flows for all major operations in Megabrain.

## Table of Contents

1. [Zotero Sync Flow](#zotero-sync-flow)
2. [Literature Search Flow](#literature-search-flow)
3. [Add Content Flows](#add-content-flows)
4. [Universal Search Flow](#universal-search-flow)
5. [Citation Backfill Flow](#citation-backfill-flow)
6. [Error Handling Flows](#error-handling-flows)

---

## Zotero Sync Flow

### High-Level Overview

```mermaid
graph TD
    A[User/Cron: zotero_sync] --> B[Load state.json]
    B --> C{Has last_synced?}
    C -->|Yes| D[Incremental Sync]
    C -->|No| E[Full Sync]
    D --> F[GET /items?since=version]
    E --> F
    F --> G[Zotero API Response]
    G --> H[Process Items Batch]
    H --> I{Has Attachments?}
    I -->|Yes| J[Download PDFs]
    I -->|No| K[Extract Metadata]
    J --> L[Extract PDF Text]
    L --> K
    K --> M{Has Notes?}
    M -->|Yes| N[Extract Note Text]
    M -->|No| O[Chunk Content]
    N --> O
    O --> P[Generate Embeddings]
    P --> Q{Content Type?}
    Q -->|Paper/PDF| R[Add to 📚 Literature]
    Q -->|Standalone Note| S[Add to 🧩 Mind]
    R --> T[Update state.json]
    S --> T
    T --> U[Return Sync Stats]
```

### Detailed Sync Process

```mermaid
sequenceDiagram
    participant U as User/Cron
    participant ZS as ZoteroSync
    participant ZA as Zotero API
    participant PE as PDF Extractor
    participant VE as Voyage API
    participant LI as Literature Index
    participant MI as Mind Index
    participant ST as state.json

    U->>ZS: zotero_sync()
    ZS->>ST: Read last_synced version
    ST-->>ZS: version = 12345
    
    ZS->>ZA: GET /items?since=12345
    ZA-->>ZS: [items: 10 new/changed]
    
    loop For each item
        ZS->>ZS: Parse metadata
        
        alt Has PDF attachment
            ZS->>ZA: Download PDF
            ZA-->>ZS: PDF bytes
            ZS->>PE: Extract text
            PE-->>ZS: Full text
        end
        
        alt Has notes
            ZS->>ZS: Extract note text
        end
        
        ZS->>ZS: Chunk text (1000 chars, 200 overlap)
        
        ZS->>VE: Embed chunks (voyage-2)
        VE-->>ZS: Embeddings [1024-dim]
        
        alt Paper/PDF content
            ZS->>LI: Add chunks to literature
            LI-->>ZS: Success
        end
        
        alt Standalone note
            ZS->>MI: Add chunks to mind
            MI-->>ZS: Success
        end
    end
    
    ZS->>ZA: GET /users/me
    ZA-->>ZS: Current library version = 12355
    
    ZS->>ST: Update last_synced = 12355
    ST-->>ZS: Saved
    
    ZS-->>U: {papers_added: 8, notes_added: 15, chunks: 847}
```

### State Management

```mermaid
stateDiagram-v2
    [*] --> NoState: First run
    NoState --> FullSync: last_synced = None
    FullSync --> Synced: Save version
    Synced --> CheckVersion: Next sync
    CheckVersion --> IncrementalSync: Changes found
    CheckVersion --> NoOp: No changes
    IncrementalSync --> Synced: Update version
    NoOp --> Synced: No update
    Synced --> [*]
```

---

## Literature Search Flow

### End-to-End RAG Pipeline

```mermaid
graph TD
    A[User: zotero_search query] --> B[Embed Query]
    B --> C[Voyage-2 Embedding]
    C --> D[FAISS Search]
    D --> E[Top-100 Candidates]
    E --> F[Extract Full Texts]
    F --> G[Voyage Rerank-2]
    G --> H[Top-15 Reranked]
    H --> I{Has Citations?}
    I -->|Yes| J[Apply Citation Boost]
    I -->|No| K[Use Rerank Score]
    J --> L[Sort by Final Score]
    K --> L
    L --> M[Format Context]
    M --> N[Build LLM Prompt]
    N --> O[Grok-41-fast Synthesis]
    O --> P[Parse Response]
    P --> Q[Generate Bibliography]
    Q --> R[Return Formatted Result]
```

### Detailed Search Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant ZS as ZoteroSearch
    participant VE as Voyage API
    participant FI as FAISS Index
    participant LLM as Venice LLM

    U->>ZS: search("transformer efficiency")
    
    ZS->>VE: Embed query (voyage-2)
    VE-->>ZS: query_vec [1024-dim]
    
    ZS->>FI: similarity_search(query_vec, k=100)
    FI-->>ZS: [100 candidate chunks]
    
    ZS->>ZS: Deduplicate by paper
    Note over ZS: Merge chunks from same paper
    
    ZS->>VE: Rerank(query, [75 papers], top_k=15)
    VE-->>ZS: [15 papers with rerank_scores]
    
    loop For each paper
        ZS->>ZS: Get citation count
        ZS->>ZS: final_score = rerank * (1 + 0.15*log(1+cites))
    end
    
    ZS->>ZS: Sort by final_score
    
    ZS->>LLM: Synthesize(query, [top 15 papers])
    Note over ZS,LLM: Prompt includes:<br/>- Query<br/>- Paper abstracts<br/>- Key chunks<br/>- Metadata
    
    LLM-->>ZS: Synthesis text
    
    ZS->>ZS: Format response:<br/>- Thematic analysis<br/>- Cross-synthesis<br/>- Appendix<br/>- Bibliography
    
    ZS-->>U: Formatted result
```

### Citation Boost Calculation

```mermaid
graph LR
    A[Rerank Score: 0.85] --> B{Citation Count}
    B -->|0| C[Boost: 1.0]
    B -->|10| D[Boost: 1.36]
    B -->|100| E[Boost: 1.69]
    B -->|1000| F[Boost: 2.04]
    
    C --> G[Final: 0.85 × 1.0 = 0.85]
    D --> H[Final: 0.85 × 1.36 = 1.16]
    E --> I[Final: 0.85 × 1.69 = 1.44]
    F --> J[Final: 0.85 × 2.04 = 1.73]
    
    style G fill:#ffcccc
    style H fill:#ffeecc
    style I fill:#ffffcc
    style J fill:#ccffcc
```

**Formula**: `final_score = rerank_score × (1 + 0.15 × log(1 + citations))`

---

## Add Content Flows

### Add Note Flow (🧩)

```mermaid
graph TD
    A[User: add_note] --> B[UserContent.add_note]
    B --> C[Build Metadata]
    C --> D[TagInference.infer_tags]
    D --> E[LLM Prompt: Extract Topics]
    E --> F[Parse LLM Response]
    F --> G[Merge with User Metadata]
    G --> H[Chunk Text]
    H --> I[Generate Embeddings]
    I --> J[Add to 🧩 Mind Index]
    J --> K[Save Index]
    K --> L[Return ID + Stats]
```

### Add URL Flow (🧩)

```mermaid
sequenceDiagram
    participant U as User
    participant UC as UserContent
    participant WEB as Web
    participant TI as TagInference
    participant VE as Voyage API
    participant MI as Mind Index

    U->>UC: add_url_content(url, title)
    
    UC->>WEB: Fetch webpage
    WEB-->>UC: HTML content
    
    UC->>UC: Extract main text<br/>(BeautifulSoup + Readability)
    
    UC->>UC: Generate title from <title> tag
    
    UC->>TI: Infer tags from text
    TI->>TI: Build LLM prompt
    TI-->>UC: {topics, subtopics, related_to}
    
    UC->>UC: Chunk text (1000 chars)
    
    UC->>VE: Embed chunks
    VE-->>UC: Embeddings
    
    UC->>MI: Add chunks + metadata
    MI-->>UC: Success
    
    UC-->>U: {id, chunks_added, topics}
```

### Add Video Transcript Flow (🧩)

```mermaid
graph TD
    A[User: add_video_transcript] --> B[Extract Video ID]
    B --> C[YouTube API / yt-dlp]
    C --> D{Transcript Available?}
    D -->|Yes| E[Download Transcript]
    D -->|No| F[Error: No Captions]
    E --> G[Combine Segments]
    G --> H[Get Video Metadata]
    H --> I[TagInference]
    I --> J[Chunk + Embed]
    J --> K[Add to 🧩 Mind]
    K --> L[Return Stats]
    F --> M[Return Error]
```

### Add Audio Content Flow (🧩)

```mermaid
graph TD
    A[User: add_audio_content] --> B[Download Audio File]
    B --> C[Whisper Transcription]
    C --> D[Full Transcript Text]
    D --> E[TagInference]
    E --> F[Chunk + Embed]
    F --> G[Add to 🧩 Mind]
    G --> H[Return Stats]
```

### Add to Second Brain Flow (🧠)

```mermaid
graph TD
    A[User: add_to_brain] --> B[Validate Category/Tags]
    B --> C[TagInference Enhancement]
    C --> D[Merge Manual + Auto Tags]
    D --> E[Chunk Content]
    E --> F[Generate Embeddings]
    F --> G[Add to 🧠 Second Brain]
    G --> H[Save Index]
    H --> I[Return ID + Stats]
```

---

## Universal Search Flow

### Cross-Partition Search

```mermaid
graph TD
    A[User: universal_search] --> B[Embed Query]
    B --> C[Parallel Search]
    C --> D[Search 📚 Literature k=10]
    C --> E[Search 🧩 Mind k=10]
    C --> F[Search 🧠 Second Brain k=10]
    D --> G[Merge Results Pool: 30]
    E --> G
    F --> G
    G --> H[Voyage Rerank-2]
    H --> I[Top-15 Cross-Partition]
    I --> J[Group by Partition]
    J --> K[Format with Emojis]
    K --> L[Return Structured Results]
```

### Detailed Universal Search Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant SB as SecondBrain
    participant VE as Voyage API
    participant LI as Lit Index
    participant MI as Mind Index
    participant BI as Brain Index

    U->>SB: universal_search(query)
    
    SB->>VE: Embed query
    VE-->>SB: query_vec
    
    par Search all partitions
        SB->>LI: search(query_vec, k=10)
        LI-->>SB: [10 lit results]
    and
        SB->>MI: search(query_vec, k=10)
        MI-->>SB: [10 mind results]
    and
        SB->>BI: search(query_vec, k=10)
        BI-->>SB: [10 brain results]
    end
    
    SB->>SB: Merge into pool [30 results]
    
    SB->>VE: Rerank(query, [30 docs], top_k=15)
    VE-->>SB: [15 reranked results]
    
    SB->>SB: Group by partition:<br/>📚 Literature: 5<br/>🧩 Mind: 6<br/>🧠 Second Brain: 4
    
    SB->>SB: Format with partition labels
    
    SB-->>U: Structured results
```

---

## Citation Backfill Flow

### Update Citation Counts

```mermaid
graph TD
    A[User/Cron: citation_backfill] --> B[Load Literature Index]
    B --> C[Extract All Paper IDs]
    C --> D[Batch Papers: 50 at a time]
    D --> E{More Batches?}
    E -->|Yes| F[Query Citation API]
    E -->|No| M[Complete]
    F --> G[Semantic Scholar / Crossref]
    G --> H[Parse Citation Counts]
    H --> I[Update Metadata]
    I --> J[Rebuild Index Entry]
    J --> K[Increment Batch]
    K --> E
    M --> N[Save Updated Index]
    N --> O[Return Stats]
```

### Citation API Integration

```mermaid
sequenceDiagram
    participant CB as CitationBackfill
    participant LI as Literature Index
    participant SS as Semantic Scholar
    participant CR as Crossref

    CB->>LI: Get all paper DOIs
    LI-->>CB: [DOI list: 386 papers]
    
    loop Batch of 50 DOIs
        CB->>SS: GET /paper/batch {DOIs}
        SS-->>CB: [citation counts for found papers]
        
        CB->>SS: Check missing DOIs
        SS-->>CB: Not found
        
        CB->>CR: GET /works {missing DOIs}
        CR-->>CB: [citation counts for remaining]
        
        CB->>LI: Update metadata[citations]
        LI-->>CB: Updated
    end
    
    CB->>LI: Save index
    LI-->>CB: Saved
    
    CB-->>CB: {updated: 380, missing: 6}
```

---

## Error Handling Flows

### Sync Error Recovery

```mermaid
stateDiagram-v2
    [*] --> Syncing
    Syncing --> Success: API responds
    Syncing --> NetworkError: Timeout/Connection
    Syncing --> AuthError: Invalid API key
    Syncing --> RateLimited: 429 response
    
    NetworkError --> Retry: Wait 5s
    RateLimited --> Retry: Wait 60s
    AuthError --> Abort: User action required
    
    Retry --> Syncing: Attempt < 3
    Retry --> Abort: Attempt >= 3
    
    Success --> UpdateState
    Abort --> Rollback
    
    UpdateState --> [*]
    Rollback --> [*]: state.json unchanged
```

### Search Error Handling

```mermaid
graph TD
    A[Search Request] --> B{Index Exists?}
    B -->|No| C[Error: Index Not Found]
    B -->|Yes| D[Embed Query]
    D --> E{Embedding Fails?}
    E -->|Yes| F[Error: Voyage API]
    E -->|No| G[FAISS Search]
    G --> H{Results Found?}
    H -->|No| I[Return Empty]
    H -->|Yes| J[Rerank]
    J --> K{Rerank Fails?}
    K -->|Yes| L[Use FAISS Scores]
    K -->|No| M[LLM Synthesis]
    M --> N{LLM Fails?}
    N -->|Yes| O[Return Raw Results]
    N -->|No| P[Return Synthesis]
    
    C --> Q[User Action: Run zotero_sync]
    F --> R[User Action: Check API key]
    I --> S[Suggest: Broaden query]
    L --> P
    O --> P
```

### Add Content Error Handling

```mermaid
graph TD
    A[Add Content Request] --> B{Valid URL?}
    B -->|No| C[Error: Invalid URL]
    B -->|Yes| D[Fetch Content]
    D --> E{Fetch Success?}
    E -->|No| F[Error: HTTP/Network]
    E -->|Yes| G[Extract Text]
    G --> H{Text Length > 0?}
    H -->|No| I[Error: No Content]
    H -->|Yes| J[Tag Inference]
    J --> K{LLM Responds?}
    K -->|No| L[Use Default Tags]
    K -->|Yes| M[Embed Chunks]
    M --> N{Embedding Success?}
    N -->|No| O[Error: Voyage API]
    N -->|Yes| P[Add to Index]
    L --> M
    P --> Q[Success]
    
    C --> R[Return Error]
    F --> R
    I --> R
    O --> R
```

---

## Data Structure Transformations

### Paper → FAISS Entry

```mermaid
graph LR
    A[Zotero Item] --> B{Parse}
    B --> C[Metadata Dict]
    C --> D[Extract Text]
    D --> E[Chunk: 1000 chars]
    E --> F[Multiple Chunks]
    F --> G[Embed Each Chunk]
    G --> H[Vector Array]
    H --> I[FAISS Entry]
    
    style A fill:#e1f5ff
    style C fill:#fff4e1
    style F fill:#ffe1f5
    style H fill:#e1ffe1
    style I fill:#f5e1ff
```

**Example Transformation**:
```
Input (Zotero):
{
  "key": "ABC123",
  "title": "Attention Is All You Need",
  "creators": [{"firstName": "Ashish", "lastName": "Vaswani"}],
  "date": "2017"
}

↓ Process

Output (FAISS Metadata):
[
  {
    "item_key": "ABC123",
    "title": "Attention Is All You Need",
    "authors": ["Vaswani, Ashish"],
    "year": 2017,
    "chunk_index": 0,
    "text": "The dominant sequence transduction models..."
  },
  {
    "item_key": "ABC123",
    "title": "Attention Is All You Need",
    "authors": ["Vaswani, Ashish"],
    "year": 2017,
    "chunk_index": 1,
    "text": "...based on complex recurrent or convolutional..."
  }
]
```

---

## Performance Flows

### Latency Breakdown (Search)

```mermaid
gantt
    title Search Operation Timeline (Typical Query)
    dateFormat X
    axisFormat %L ms

    section Embedding
    Voyage API Call: 0, 50
    
    section FAISS
    Vector Search (k=100): 50, 55
    
    section Reranking
    Voyage Rerank API: 55, 255
    
    section Citation
    Boost Calculation: 255, 260
    
    section Synthesis
    LLM API Call: 260, 760
    Format Response: 760, 765
```

**Total Latency**: ~765ms per search

---

## Cron Job Flow

### Auto-Sync Schedule

```mermaid
gantt
    title Megabrain Auto-Sync (Daily Schedule)
    dateFormat HH:mm
    axisFormat %H:%M

    section Sync Events
    Sync Run 1: 00:00, 00:02
    Sync Run 2: 02:00, 02:02
    Sync Run 3: 04:00, 04:02
    Sync Run 4: 06:00, 06:02
    Sync Run 5: 08:00, 08:02
    Sync Run 6: 10:00, 10:02
    Sync Run 7: 12:00, 12:02
    Sync Run 8: 14:00, 14:02
    Sync Run 9: 16:00, 16:02
    Sync Run 10: 18:00, 18:02
    Sync Run 11: 20:00, 20:02
    Sync Run 12: 22:00, 22:02
```

### Cron Job Execution

```mermaid
sequenceDiagram
    participant CR as Cron Daemon
    participant SH as Shell Script
    participant ZS as ZoteroSync
    participant LOG as Log File

    CR->>SH: Execute zotero-sync-v2.sh
    SH->>LOG: Timestamp: Start
    SH->>ZS: python -m megabrain.zotero_tools sync
    
    alt Sync Successful
        ZS-->>SH: Exit 0 + Stats
        SH->>LOG: Success + Stats
        SH-->>CR: Exit 0
    else Sync Failed
        ZS-->>SH: Exit 1 + Error
        SH->>LOG: Error + Traceback
        SH-->>CR: Exit 1
    end
```

---

## Summary

### Key Data Flow Patterns

1. **Incremental Sync**: State-based versioning minimizes API calls
2. **Multi-Stage Retrieval**: FAISS → Rerank → Boost for precision
3. **Parallel Search**: Universal search queries all partitions concurrently
4. **Error Recovery**: Graceful degradation (skip rerank/LLM if needed)
5. **Async Processing**: Background cron job keeps data fresh

### Bottlenecks

| Operation | Bottleneck | Mitigation |
|-----------|------------|------------|
| Sync | Zotero API rate limits | Batch requests, incremental sync |
| Search | Voyage rerank API latency | Cache frequent queries (future) |
| Embedding | Voyage API cost | Efficient chunking, batch operations |
| LLM Synthesis | Venice.ai latency | Stream responses (future) |

---

**Version**: 3.1  
**Last Updated**: February 2026
