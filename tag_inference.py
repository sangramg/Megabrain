#!/usr/bin/env python3
"""
Shared LLM-based tag inference for Megabrain structured metadata enrichment.
Used by user_content.py, second_brain.py, zotero_tools.py, and tag_enrichment.py.

Author: Percy (OpenClaw Agent)
Date: 2026-02-20
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_client = None

def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.environ.get("VENICE_API_KEY")
        if not api_key:
            raise EnvironmentError("VENICE_API_KEY not set")
        _client = OpenAI(base_url="https://api.venice.ai/v1", api_key=api_key)
    return _client


def infer_tags_mind(content: str, existing_meta: dict = None) -> dict:
    """Infer topic, subtopics, context, sentiment for mind partition."""
    meta = existing_meta or {}
    snippet = content[:1500]
    title = meta.get('title', '')

    prompt = f"""Analyze this personal note and return JSON with exactly these fields:
- "topic": primary topic (1-3 words, e.g. "AI", "methodology", "platforms")
- "subtopics": comma-separated secondary themes (2-5 items)
- "context": when/why this was written (e.g. "reading session", "class notes", "personal reflection")
- "sentiment": author's stance - one of: "supportive", "critical", "exploratory", "neutral"

Title: {title}
Content: {snippet}

Return ONLY valid JSON, no markdown fences."""

    return _call_llm(prompt)


def infer_tags_second_brain(content: str, existing_meta: dict = None) -> dict:
    """Infer topic, subtopics for second_brain partition."""
    meta = existing_meta or {}
    snippet = content[:1500]
    title = meta.get('title', '')
    category = meta.get('category', '')

    prompt = f"""Analyze this content and return JSON with exactly these fields:
- "topic": primary topic (1-3 words)
- "subtopics": comma-separated secondary themes (2-5 items)
{"" if category else '- "category": content category (e.g. "tech", "philosophy", "cooking", "business")'}

Title: {title}
Content: {snippet}

Return ONLY valid JSON, no markdown fences."""

    return _call_llm(prompt)


def infer_tags_literature(content: str, existing_meta: dict = None) -> dict:
    """Infer topic, subtopics, methodology for literature partition."""
    meta = existing_meta or {}
    snippet = content[:1500]
    title = meta.get('title', '')
    tags = meta.get('tags', '')

    # Try to derive from existing Zotero tags first
    if tags:
        tag_list = [t.strip() for t in tags.split(',') if t.strip()]
        if tag_list:
            result = {}
            result['topic'] = tag_list[0]
            if len(tag_list) > 1:
                result['subtopics'] = ', '.join(tag_list[1:])
            else:
                result['subtopics'] = tag_list[0]
            # Still need methodology from LLM
            meth = infer_methodology_heuristic(content)
            if meth:
                result['methodology'] = meth
            else:
                meth_result = _call_llm(
                    f"""What research methodology does this paper use? Return JSON with one field:
- "methodology": one of "qualitative", "quantitative", "mixed", "conceptual", "review", "design-science"

Title: {title}
Content: {snippet}

Return ONLY valid JSON, no markdown fences."""
                )
                result['methodology'] = meth_result.get('methodology', 'conceptual')
            return result

    # Full LLM inference
    prompt = f"""Analyze this research paper and return JSON with exactly these fields:
- "topic": primary research topic (1-3 words)
- "subtopics": comma-separated secondary themes (2-5 items)
- "methodology": one of "qualitative", "quantitative", "mixed", "conceptual", "review", "design-science"

Title: {title}
Tags: {tags}
Content: {snippet}

Return ONLY valid JSON, no markdown fences."""

    return _call_llm(prompt)


def infer_methodology_heuristic(content: str) -> Optional[str]:
    """Try to infer methodology from keywords before falling back to LLM."""
    text = content.lower()
    mapping = {
        'literature review': 'review',
        'systematic review': 'review',
        'meta-analysis': 'review',
        'survey': 'quantitative',
        'regression': 'quantitative',
        'sem': 'quantitative',
        'structural equation': 'quantitative',
        'experiment': 'quantitative',
        'interview': 'qualitative',
        'case study': 'qualitative',
        'ethnograph': 'qualitative',
        'grounded theory': 'qualitative',
        'design science': 'design-science',
        'design-science': 'design-science',
        'mixed method': 'mixed',
    }
    for keyword, method in mapping.items():
        if keyword in text:
            return method
    return None


def _call_llm(prompt: str) -> dict:
    """Call LLM and parse JSON response."""
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        text = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return json.loads(text)
    except Exception as e:
        logger.warning(f"LLM tag inference failed: {e}")
        return {}
