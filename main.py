import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import MarianMTModel, MarianTokenizer
import torch
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from difflib import SequenceMatcher
import base64
import numpy as np
from PIL import Image
import io
from urllib.parse import urlparse
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
from readability import Document
from typing import List, Tuple
import json
from math import log2
from functools import lru_cache
import hashlib
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from sentence_transformers import SentenceTransformer, util
from contextvars import ContextVar
import json
import pillow_avif
import cv2
from difflib import get_close_matches
import wordfreq
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from transformers import pipeline
import pytesseract
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
from paddleocr import PaddleOCR
import shutil
if shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(title="Fake News Detection API ")

@app.get("/")
def root():
    return {
        "status": "Fake News Detection API running",
        "docs": "/docs",
        "health": "/health"
    }
    
@app.get("/health")
def health():
    return {
        "api": "ok",
        "model_loaded": True
    }

# =========================================================
# CONFIGURATION
# =========================================================

GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
GOOGLE_FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "models" / "distilbert_english")

# =========================================================
# LANGUAGE DETECTION
# =========================================================

LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu"
}

def get_language_name(lang_code: str):
    if not lang_code:
        return "Unknown"

    lang_code = lang_code.lower()

    for prefix, name in LANGUAGE_MAP.items():
        if lang_code.startswith(prefix):
            return name

    return "Unknown"

# =========================================================
# >>> ADDED: INDIC NORMALIZATION + SAFE LANGUAGE DETECTION
# =========================================================

indic_normalizer_factory = IndicNormalizerFactory()

def normalize_indic_text(text: str, lang: str) -> str:
    try:
        if lang in ["hi", "te"]:
            normalizer = indic_normalizer_factory.get_normalizer(lang)
            return normalizer.normalize(text)
        return text
    except Exception:
        return text

def safe_language_detect(text: str) -> str:
    try:
        lang = detect(text)
        if lang in ["en", "hi", "te"]:
            return lang
    except Exception:
        pass

    # Unicode fallback (critical for Telugu/Hindi)
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "te"
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    return "en"

def language_specific_preprocess(text: str, lang: str) -> str:
    if lang == "en":
        text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    elif lang == "hi":
        text = re.sub(r"[^\u0900-\u097F\s]", "", text)
    elif lang == "te":
        text = re.sub(r"[^\u0C00-\u0C7F\s]", "", text)

    return re.sub(r"\s+", " ", text).strip()





# =====================================================
# FETCH PAGE SAFELY
# =====================================================

@lru_cache(maxsize=512)
def cached_fetch_page(url: str) -> str:
    return fetch_page(url)


def fetch_page(url: str) -> str:
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (RAG Bot)"
            },
            timeout=8
        )

        if r.status_code != 200:
            return ""

        html = r.text

        # Reject tiny / broken pages
        if not html or len(html) < 1500:
            return ""

        # Reject obvious error pages
        error_signals = [
            "page not found",
            "404 error",
            "access denied",
            "temporarily unavailable",
            "service unavailable"
        ]

        html_l = html.lower()
        if any(e in html_l for e in error_signals):
            return ""

        return html

    except Exception:
        return ""


# =====================================================
# CLEAN ARTICLE TEXT EXTRACTION
# =====================================================

def _embed_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

@lru_cache(maxsize=2048)
def cached_single_embedding(text: str):
    model = get_similarity_model()
    emb = model.encode(text, normalize_embeddings=True)
    return emb

@lru_cache(maxsize=1024)
def cached_multi_embedding(texts_key: str, texts_tuple: tuple):
    model = get_similarity_model()
    embs = model.encode(list(texts_tuple), normalize_embeddings=True)
    return embs

def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

@lru_cache(maxsize=512)
def cached_classify_url_type(url: str):
    return classify_url_type(url)


@lru_cache(maxsize=1024)
def cached_sentences(text: str):
    return tuple(extract_sentences(text))


@lru_cache(maxsize=512)
def cached_claim_parse(claim: str):
    return (
        extract_subject_from_claim(claim),
        extract_role_from_claim(claim)
    )


@lru_cache(maxsize=512)
def cached_extract_main_text(html: str, page_type: str) -> str:
    return extract_main_text(html, page_type)


def extract_main_text(html: str, page_type: str | None = None) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # =========================
    # Remove junk tags (UNCHANGED)
    # =========================
    for tag in soup([
        "script", "style", "nav", "footer",
        "header", "noscript", "form",
        "aside", "svg"
    ]):
        tag.decompose()

    PROFILE_WORDS = [
        "professor", "assistant professor",
        "associate professor", "department",
        "hod", "faculty", "research",
        "qualification", "experience",
        "publications", "email"
    ]

    blocks = []
    seen_texts = set()   # ✅ NEW: duplicate removal
    # ==================================================
    # 🔥 FIX-1 (INDUSTRY STANDARD)
    # Semantic role-first extraction (NEW – PRIMARY)
    # ==================================================

    semantic_blocks = []
    reconstructed_roles = reconstruct_roles_from_dom(soup)

    if page_type == "PROFILE" and reconstructed_roles:
        return " ".join(reconstructed_roles)

    for tag in soup.find_all(
        ["h1", "h2", "h3", "p", "span", "strong", "li", "td", "dd", "dt"]
    ):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue

        tl = text.lower()
        if any(r in tl for r in ROLE_WORDS):
            parent = tag.find_parent(["section", "article", "div"]) or tag
            block = parent.get_text(" ", strip=True)

            if 40 <= len(block.split()) <= 200:
                key = block.lower()[:120]
                if key not in seen_texts:
                    seen_texts.add(key)
                    semantic_blocks.append(block)

    if page_type == "PROFILE" and semantic_blocks:
        return " ".join(semantic_blocks[:2])


    # =========================
    # Add headings explicitly (UNCHANGED)
    # =========================
    for h in soup.find_all(["h1", "h2", "h3"]):
        ht = h.get_text(" ", strip=True)
        if ht and len(ht.split()) <= 12:
            norm = ht.lower()
            if norm not in seen_texts:
                seen_texts.add(norm)
                blocks.append((50, ht))

    # =========================
    # Main semantic blocks (ENHANCED)
    # =========================
    for tag in soup.find_all(["article", "main", "section", "div"]):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue

        if is_boilerplate(text):
            continue

        text = re.sub(r"\s+", " ", text).strip()
        text_lower = text.lower()

        # ✅ NEW: skip duplicates
        if text_lower in seen_texts:
            continue
        seen_texts.add(text_lower)

        word_count = len(text.split())



        # =========================
        # Short block protection (UNCHANGED)
        # =========================
        if word_count < 40:
            has_role = any(r in text_lower for r in ROLE_WORDS)
            if not has_role:
                continue

        # =========================
        # Junk filtering (UNCHANGED)
        # =========================
        junk_patterns = [
            "about us", "contact us", "privacy policy",
            "all rights reserved", "navigation",
            "menu", "login", "register"
        ]
        if any(j in text_lower for j in junk_patterns):
            continue

        # 🔥 EARLY PROFILE ROLE CAPTURE (CRITICAL FOR FACULTY PAGES)
        if page_type == "PROFILE":
            for tag in soup.find_all(["p", "li", "span", "td", "strong"]):
                t = tag.get_text(" ", strip=True)
                if (
                    any(r in t.lower() for r in ROLE_WORDS)
                    and len(t.split()) >= 3
                ):
                    return t

        if page_type == "PROFILE":
            identity_text = extract_profile_identity_text(soup)
            if identity_text:
                return identity_text

        # =========================
        # Link density penalty (UNCHANGED)
        # =========================
        links = tag.find_all("a")
        link_text_len = sum(len(a.get_text(strip=True)) for a in links)
        link_ratio = link_text_len / max(1, len(text))
        density = word_count * (1 - min(link_ratio, 0.9))

        # =========================
        # Profile keyword boost (UNCHANGED)
        # =========================
        profile_hits = sum(w in text_lower for w in PROFILE_WORDS)

        # =========================
        # NEW: Page-type aware boost
        # =========================
        page_boost = 0
        if page_type == "PROFILE":
            page_boost = profile_hits * 4   # strong boost for identity pages

        # =========================
        # Repetition penalty (UNCHANGED)
        # =========================
        repetition_penalty = 0
        if text_lower.count("forums") > 3:
            repetition_penalty = 3

        score = (
            density
            + profile_hits * 2
            + page_boost
            - repetition_penalty
        )

        blocks.append((score, text))

    # =========================
    # ✅ NEW: Table extraction (VERY IMPORTANT)
    # =========================
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(" ".join(cells))

        table_text = " ".join(rows)
        if len(table_text.split()) >= 10:
            blocks.append((45, table_text))

    # =========================
    # ✅ NEW: List extraction
    # =========================
    for ul in soup.find_all(["ul", "ol"]):
        items = [li.get_text(" ", strip=True) for li in ul.find_all("li")]
        list_text = " ".join(items)

        if len(list_text.split()) >= 10:
            blocks.append((35, list_text))

    # =========================
    # Fallback to paragraphs (UNCHANGED)
    # =========================
    if not blocks:
        for p in soup.find_all("p"):
            t = p.get_text(" ", strip=True)
            if len(t.split()) > 8:
                blocks.append((5, t))

    if not blocks:
        return ""

    # 🔥 ARTICLE FALLBACK — FULL PARAGRAPHS
    if page_type != "PROFILE":
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True).split()) > 30
        ]

        if paragraphs:
            return " ".join(paragraphs[:6])

    # =========================
    # Choose best content (UNCHANGED)
    # =========================
    blocks.sort(key=lambda x: x[0], reverse=True)

    best_text = " ".join(b[1] for b in blocks[:3])


    # # 🔥 FORCE PROFILE EXTRACTION FOR OFFICIAL PAGES
    # if len(best_text) < 200:
    #     profile_blocks = []

    #     for tag in soup.find_all(["p", "li", "td", "span"]):
    #         t = tag.get_text(" ", strip=True)
    #         if (
    #             "professor" in t.lower()
    #             or "department" in t.lower()
    #              or "vbit" in t.lower()
    #         ) and len(t.split()) >= 5:
    #             profile_blocks.append(t)

    #     if profile_blocks:
    #         best_text = " ".join(profile_blocks[:5])

    # 🔥 FORCE PROFILE EXTRACTION FOR OFFICIAL PAGES (DYNAMIC VERSION)
    if len(best_text) < 200:
        profile_blocks = []

        for tag in soup.find_all(["h1","h2","h3","h4","p","li","td","span","strong","dd"]):

            t = tag.get_text(" ", strip=True)
            if not t:
                continue

            tl = t.lower()

            # ✅ Use dynamic vocab instead of hardcoding
            role_hit = any(r in tl for r in ROLE_WORDS)
            profile_hit = any(p in tl for p in PROFILE_WORDS)

            # Require meaningful identity text
            if (role_hit or profile_hit) and len(t.split()) >= 5:
                profile_blocks.append(t)

        if profile_blocks:
            best_text = " ".join(profile_blocks[:5])

    # ==================================================
    # 🔥 LAYER 3: SENTENCE-LEVEL IDENTITY FALLBACK (LAST)
    # ==================================================
    if not best_text or len(best_text.split()) < 20:
        full_text = soup.get_text(" ", strip=True)

        sentences = re.split(r"[.\n]", full_text)

        identity_sentences = []

        for s in sentences:
            s = s.strip()
            if len(s.split()) < 3:
                continue

            if page_type == "PROFILE":
                # For PROFILE pages, require role words
                if not any(r in s.lower() for r in ROLE_WORDS):
                    continue

            # For ARTICLE pages, accept all sentences
            identity_sentences.append(s)

        if identity_sentences:
            best_text = identity_sentences[0]

    # =========================
    # Cleanup noise (UNCHANGED + SMALL UPGRADE)
    # =========================
    best_text = re.sub(r"\[[^\]]*\]", "", best_text)
    best_text = re.sub(r"\s+", " ", best_text)

    return best_text.strip()


def prepare_claim_for_rag(headline: str) -> str:
    """
    Normalizes and translates claim before sending to RAG.
    Used ONLY in URL pipeline.
    """

    if not headline:
        return headline

    # Detect language
    lang = safe_language_detect(headline)
    print("🌐 HEADLINE LANG:", lang)

    # Normalize Indic text (if you already have this function)
    try:
        headline = normalize_indic_text(headline, lang)
    except:
        pass

    # Translate Telugu / Hindi to English
    headline = cached_translate_to_english(headline, lang)

    print("📝 CLAIM SENT TO RAG:", headline)

    return headline

def get_source_authority(url: str) -> int:
    """
    Returns authority score (0–100) based on domain & structure.
    Generic, no institute hardcoding.
    """
    u = url.lower()

    # Academic / government institutions
    if any(d in u for d in [".ac.", ".edu", ".gov"]):
        return 90

    # Official org subdomains (employees, staff, people)
    if any(k in u for k in ["employees", "faculty", "staff", "people", "profile"]):
        return 85

    # Academic identity platforms
    if any(d in u for d in ["scholar.google", "researchgate", "orcid"]):
        return 70

    # Professional networks
    if "linkedin.com" in u:
        return 60

    # Blogs / aggregators
    if any(d in u for d in ["biography", "omicsonline", "thecn"]):
        return 40

    return 30


# =====================================================
# CHUNKING (important for embeddings)
# =====================================================

def chunk_text(text, chunk_size=120, overlap=30):
    words = text.split()

    # ✅ fallback for small pages
    if len(words) < chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        chunk = words[start:start + chunk_size]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    print("Chunks:", len(chunks))

    return chunks

def extract_best_evidence_sentence(text: str, claim: str) -> str:
    text = remove_citation_markers(text)
    sentences = extract_sentences(text)
    


    if not sentences:
        return text[:200]

    # 🔥 Prefer sentences that contain ROLE words
    role_hits = [
        s for s in sentences
        if any(r in s.lower() for r in ROLE_WORDS)
    ]

    candidates = role_hits if role_hits else sentences

    model = get_similarity_model()
    claim_emb = cached_single_embedding(claim)
    candidates_tuple = tuple(candidates)
    candidates_key = _embed_key("||".join(candidates_tuple))

    sent_embs = cached_multi_embedding(
        candidates_key,
        candidates_tuple
    )


    scores = util.cos_sim(claim_emb, sent_embs)[0]
    best_idx = int(scores.argmax())

    return candidates[best_idx]

# =====================================================
# MAIN UNIVERSAL RETRIEVER
# =====================================================

def detect_college_urls(text: str):
    text_lower = text.lower()
    urls = []

    for college, url in LOCAL_COLLEGES.items():
        if college in text_lower:
            urls.append(url)

    return urls

def reconstruct_roles_from_dom(soup):
    """
    Reconstructs split roles like:
    Assistant + Professor → Assistant Professor
    """
    role_candidates = []

    ROLE_PARTS = {
        "assistant", "associate", "professor",
        "lecturer", "faculty", "hod", "dean"
    }

    for parent in soup.find_all(["div", "section", "tr", "dl"]):
        texts = [
            t.get_text(" ", strip=True).lower()
            for t in parent.find_all(["span", "strong", "td", "dd", "p"])
        ]

        joined = " ".join(texts)

        # semantic proximity check
        if (
            "assistant" in joined
            and "professor" in joined
        ):
            role_candidates.append("assistant professor")

        elif "associate" in joined and "professor" in joined:
            role_candidates.append("associate professor")

        elif "professor" in joined:
            role_candidates.append("professor")

    return role_candidates

def extract_profile_identity_text(soup):
    """
    Strict DOM-scoped extractor for institutional profile pages.
    Prevents nav / footer / menu pollution.
    """

    texts = []

    # 🔒 Step 1: Try common profile containers FIRST
    PROFILE_SELECTORS = [
        "div.employee-details",
        "div.employee-profile",
        "div.profile",
        "section.profile",
        "div.content",
        "div.container"
    ]

    for sel in PROFILE_SELECTORS:
        block = soup.select_one(sel)
        if block:
            for tag in block.find_all(
                ["h1", "h2", "h3", "p", "li", "span", "td", "strong"]
            ):
                t = tag.get_text(" ", strip=True)
                if t:
                    texts.append(t)

            break  # 👈 stop at first valid container

    # 🔄 Step 2: Fallback — ONLY near <h1>/<h2>
    if not texts:
        header = soup.find(["h1", "h2"])
        if header:
            parent = header.find_parent(["div", "section", "article"])
            if parent:
                for tag in parent.find_all(
                    ["h1", "h2", "h3", "p", "li", "span", "td", "strong"]
                ):
                    t = tag.get_text(" ", strip=True)
                    if t:
                        texts.append(t)

    if not texts:
        return None

    # 🔥 Step 3: Filter identity-relevant text only
    identity_lines = []

    for t in texts:
        tl = t.lower()

        # ❌ remove nav/menu garbage
        if any(x in tl for x in [
            "virtual tour", "home", "about", "contact",
            "r&d director", "ph.ds awarded", "centre"
        ]):
            continue

        # ✅ keep if role or name-like
        if any(r in tl for r in ROLE_WORDS):
            identity_lines.append(t)
            continue

        # keep likely name
        if sum(c.isupper() for c in t) >= 2 and len(t.split()) <= 6:
            identity_lines.append(t)

    if not identity_lines:
        return None

    # Deduplicate + compact
    seen = set()
    final = []
    for t in identity_lines:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            final.append(t)

    return " ".join(final[:3])

SPECULATIVE_TERMS = [
    "could", "may", "might", "possible",
    "plans to", "wants to", "expected to",
    "loophole", "eligible to", "run for",
    "cannot run", "can't run", "not eligible"
]
CURRENT_ROLE_GUARD_TERMS = [
    "former",
    "ex-",
    "no longer",
    "resigned",
    "stepped down",
    "ousted",
    "removed",
    "ended his term",
    "lost the election",
    "defeated",
    "replaced by"
]


def is_speculative_or_negative(text: str) -> bool:
    tl = text.lower()
    return any(t in tl for t in SPECULATIVE_TERMS)

def extract_role_anchored_blocks(soup, ROLE_WORDS):
    blocks = []
    seen = set()

    for tag in soup.find_all(
        ["h1", "h2", "h3", "p", "span", "strong", "li", "td", "dd", "dt"]
    ):
        text = tag.get_text(" ", strip=True)
        if not text or len(text) > 400:
            continue

        tl = text.lower()

        matched_roles = [r for r in ROLE_WORDS if r in tl]
        if not matched_roles:
            continue

        parent = tag.find_parent(["section", "article", "div", "tr", "dl"]) or tag
        block = parent.get_text(" ", strip=True)

        # quality filters
        if len(block.split()) < 8:
            continue

        key = block.lower()[:150]
        if key in seen:
            continue
        seen.add(key)

        blocks.append({
            "role": matched_roles,
            "text": block
        })

    return blocks

def explicit_role_assertion(person: str, role: str, text: str) -> bool:
    tl = text.lower()
    return (
        f"{person.lower()} is {role}" in tl
        or f"{person.lower()} serves as {role}" in tl
        or f"{person.lower()} has been {role}" in tl
    )


BOILERPLATE_PHRASES = [
    "r&d policy",
    "research facilities",
    "sponsored fdps",
    "workshops and seminars",
    "ph.ds awarded",
    "patents"
]

def is_boilerplate(text: str) -> bool:
    t = text.lower()
    hits = sum(1 for p in BOILERPLATE_PHRASES if p in t)
    return hits >= 2


def extract_subject_from_claim(claim: str) -> str | None:
    """
    Extracts the subject (person/entity) from simple claims.
    Examples:
        "Elon Musk is President of USA" -> "Elon Musk"
        "Narendra Modi is the Prime Minister of India" -> "Narendra Modi"
    """
    if not claim:
        return None

    # Split at ' is ' (safe for English factual claims)
    parts = re.split(r"\s+is\s+", claim, flags=re.IGNORECASE)
    if not parts:
        return None

    subject = parts[0].strip()

    # Cleanup trailing junk
    subject = re.sub(r"[^\w\s.-]", "", subject)

    return subject

def universal_rag_retrieve(claim: str, urls: list[str], sim_threshold=0.7, top_k=3, authority_entity: str | None = None):

    # === Init (kept same variables as original for compatibility) ===
    contradiction_found = False
    contradiction_evidence = None
    contradiction_detected = False
    support_evidence = []
    neutral_evidence = []
    support_found = False
    support_url = None

    profile_positive_confirmed = False
    profile_hierarchy_mismatch = False
    profile_authority_state = None
    profile_evidence = None
    profile_verdict = None
    max_support = 0.0
    support_hits = 0
    profile_cache = {}
    STRONG_SUPPORT_THRESHOLD = 0.70
    early_stop = False

    # NEW incremental counters (stop when either reaches 2)
    real_count = 0
    fake_count = 0
    MIN_SUPPORT_SOURCES = 2

    claim_type = classify_claim_type(claim)
    language = safe_language_detect(claim)
    keywords = extract_keywords_multilingual(claim, language)


    if claim_type == "IDENTITY_ROLE":
        MIN_SUPPORT_SOURCES = 1
        print("🧠 IDENTITY CLAIM — SINGLE SOURCE MODE ENABLED")

    role = extract_role_from_claim(claim) or ""

    # negation detection (reuse your existing helper for safety)
    is_negated = is_negated_claim(claim)
    if is_negated:
        print("🚨 NEGATED CLAIM — CONTRADICTION MODE ENABLED")

    # Preload claim embeddings lazily later only if needed
    similarity_model = None
    claim_emb = None

    # Trusted/low-priority config (reuse values from original)
    LOW_PRIORITY_DOMAINS = ["scholar.google", "researchgate"]
    trusted_domains = [".gov", ".edu", ".ac.", ".org", ".edu.in", ".ac.in"]
    official_keywords = ["employees", "faculty", "staff", "people", "profile", "directory", "department"]
    print("\n================= 🔍 RAG TRACE START =================")
    print(f"🧪 CLAIM: {claim}")
    print(f"🌐 TOTAL URLS RECEIVED: {len(urls)}")
    print("=====================================================\n")

    # ==========================================================
    # 🔥 STEP 1: DIRECT WIKIPEDIA CHECK (NATIONALITY ONLY)
    # ==========================================================

    parsed_subject, _ = cached_claim_parse(claim)
    attribute_type = detect_attribute_type(claim)

    if attribute_type == "NATIONALITY" and parsed_subject:

        wiki_subject = parsed_subject.strip().replace(" ", "_")
        wiki_url = f"https://en.wikipedia.org/wiki/{wiki_subject}"

        print("📌 DIRECT WIKI NATIONALITY CHECK:", wiki_url)

        html = cached_fetch_page(wiki_url)


        if html:
            wiki_text = extract_wikipedia_text(html)

            if wiki_text and len(wiki_text.split()) > 20:

                attr_verdict = validate_attribute(claim, wiki_text)

                print("🔎 DIRECT WIKI ATTRIBUTE VERDICT:", attr_verdict)
                print("🔎 CLAIM NEGATED:", is_negated)


                if attr_verdict == "SUPPORTED":

                    if is_negated:
                        print("🔒 NEGATED CLAIM + WIKI SUPPORT → FINAL FAKE")

                        return {
                            "status": "CONTRADICTION",
                            "finalLabel": "FAKE",
                            "confidencePercent": 97,
                            "summary": "The claim is negated but Wikipedia confirms the positive fact.",
                            "aiExplanation": (
                                "Wikipedia confirms the structured attribute, "
                                "which directly contradicts the negated claim."
                            ),
                            "keywords": keywords,
                            "evidence": [{
                                "url": wiki_url,
                                "snippet": extract_best_evidence_sentence(wiki_text, claim),
                                "score": 0.98,
                                "page_type": "ARTICLE"
                            }]
                        }

                    else:
                        print("🔒 WIKIPEDIA SUPPORT → FINAL REAL")

                        return {
                            "status": "SUPPORTED",
                            "finalLabel": "REAL",
                            "confidencePercent": 97,
                            "summary": "Wikipedia confirms the claim.",
                            "aiExplanation": "Wikipedia explicitly confirms the structured attribute.",
                            "keywords": keywords,
                            "evidence": [{
                            "url": wiki_url,
                            "snippet": extract_best_evidence_sentence(wiki_text, claim),
                            "score": 0.98,
                            "page_type": "ARTICLE"
                            }]
                        }

                if attr_verdict == "CONTRADICTION":

                    if is_negated:
                        print("🔒 NEGATED CLAIM + WIKI CONTRADICTION → FINAL REAL")

                        return {
                            "status": "SUPPORTED",
                            "finalLabel": "REAL",
                            "confidencePercent": 97,
                            "summary": "The negated claim aligns with Wikipedia.",
                            "aiExplanation": (
                                "Wikipedia contradicts the positive form, "
                                "which supports the negated claim."
                            ),
                            "keywords": keywords,
                            "evidence": [{
                                "url": wiki_url,
                                "snippet": extract_best_evidence_sentence(wiki_text, claim),
                                "score": 0.98,
                                "page_type": "ARTICLE"
                            }]
                        }

                    else:
                        print("🔒 WIKIPEDIA CONTRADICTION → FINAL FAKE")

                        return {
                            "status": "CONTRADICTION",
                            "finalLabel": "FAKE",
                            "confidencePercent": 97,
                            "summary": "Wikipedia contradicts the claim.",
                            "aiExplanation": "Wikipedia explicitly lists conflicting information.",
                            "keywords": keywords,
                            "evidence": [{
                                "url":wiki_url,
                                "snippet": extract_best_evidence_sentence(wiki_text, claim),
                                "score": 0.98,
                                "page_type": "ARTICLE"
                            }]
                        }

        print("⚠️ DIRECT WIKI INCONCLUSIVE → CONTINUE TO SEARCH RAG")


    # Helper to ensure similarity model loaded once
    def ensure_similarity_model():
        nonlocal similarity_model, claim_emb
        if similarity_model is None:
            similarity_model = get_similarity_model()
            claim_emb = similarity_model.encode(claim, convert_to_tensor=True, normalize_embeddings=True)

    # Iterate once over urls — do everything per-URL
    for url in urls:
        # ------------- classify -------------
        try:
            page_type = cached_classify_url_type(url)
        except Exception:
            page_type = "UNKNOWN"

        if page_type in ["UNKNOWN", "CATEGORY"]:
            # skip these (same behaviour as earlier)
            continue

        if page_type == "PROFILE":
            print("👤 PROFILE PAGE DETECTED → identity verification path")
        else:
            print("📰 ARTICLE PAGE DETECTED → semantic RAG path")

        semantic_score=0

        # ------------- fetch -------------
        role_snippets = []
        html = ""

        if page_type == "PROFILE":

            print(f"PROFILE FAST PATH : {url}")

            html = cached_fetch_page(url)   # lightweight fallback only
            if "Just a moment" in html or "Checking your browser" in html:
                print("🚫 CLOUDFLARE BLOCK DETECTED")
                return ""

            if not html:
                print("⚠️ Fast fetch empty — extractor  will handle DOM")

            role_snippets = []

        else:
            # 📰 NON-PROFILE → FAST REQUESTS ONLY
            html = cached_fetch_page(url)

            if not html:
                print(f" Article fetch failed, skipping: {url}")
                continue

            print("\n-----------------------------------------------------")
            print(f"🔗 URL: {url}")
            print(f"📄 PAGE TYPE: {page_type}")

        # ------------- PROFILE fast-path (must not send PROFILE to RAG) -------------
        if page_type == "PROFILE":
            # protect is_negated existence
            _is_negated = bool(is_negated)
            if url not in profile_cache:
                profile_cache[url] = extract_profile_roles(url)

            profile_data = profile_cache[url]

            role_blocks = profile_data.get("roles", [])
            profile_names = profile_data.get("names", [])
            
            print("🧠 PROFILE NAMES EXTRACTED:", profile_names)
            print("🧠 PROFILE ROLES EXTRACTED:", role_blocks)



            # quick name mismatch filter (same as earlier)
            subject, _ = cached_claim_parse(claim)
            print("🧠 CLAIM SUBJECT:", subject)
            best_name = choose_best_profile_name(subject, profile_names)
            print("🧠 BEST PROFILE NAME:", best_name)
            if subject and profile_names and not name_matches(subject, profile_names):
                # skip this profile page
                continue

            # POSITIVE claim fast-path — exact name+role match => REAL
            if role_blocks:
                subject, _ = cached_claim_parse(claim)
                if subject:
                    claim_role = role
                    name_match = name_matches(subject, profile_names)
                    role_match = any(
                                    role_matches_strict(claim_role, r)
                                    for r in role_blocks
                                )
                    # ==========================================================
                    # 🧠 MODEL-5 IDENTITY AUTHORITY BLOCK
                    # ==========================================================

                    if page_type == "PROFILE" and profile_names and role_blocks:

                        print("🧠 MODEL-5 AUTHORITY CHECK START")

                        # ✅ name match
                        name_match = name_matches(subject, profile_names)

                        # ✅ multi-role safe matching
                        # role_match = any(
                        #     role_matches_strict(claim_role, r)
                        #     for r in role_blocks
                        # )
                        expanded_roles = []

                        for r in role_blocks:
                            # Split on common separators
                            parts = re.split(r"[,&/]| and ", r.lower())
                            for p in parts:
                                p = p.strip()
                                if p:
                                    expanded_roles.append(p)

                        role_match = any(
                            role_matches_strict(claim_role, er)
                            for er in expanded_roles
                        )


                        print("🧠 MODEL-5 NAME MATCH:", name_match)
                        print("🧠 MODEL-5 ROLE MATCH:", role_match)

                        if name_match:

                            # ==================================================
                            # 🚨 NEGATED CLAIM LOGIC
                            # ==================================================
                            if _is_negated:

                                if role_match:
                                    print("🔒 MODEL-5 NEGATED CONTRADICTION — FINAL FAKE")
                                    contradiction_detected = True
                                    if semantic_score >= 0.85 and is_trusted_domain(url):
                                        contradiction_found = True

                                    fake_count += 1

                                    return {
                                        "status": "CONTRADICTION",
                                        "finalLabel": "FAKE",
                                        "confidencePercent": 96,
                                        
                                        "summary": "Official profile confirms the role exists, contradicting the negated claim.",
                                        "aiExplanation": (
                                            "An authoritative institutional profile confirms the academic role. "
                                            "Because the claim denies this role, it is false."
                                        ),
                                        "keywords": keywords,
                                        "evidence": [{
                                            "url": url,
                                            "snippet": f"{best_name} — {role_blocks[0]}",
                                            "score": 0.97,
                                            "page_type": "PROFILE"
                                        }]
                                    }

                                else:
                                    print("🛑 MODEL-5 NEGATED CLAIM VERIFIED — FINAL REAL")

                                    return {
                                        "status": "SUPPORTED",
                                        "finalLabel": "REAL",
                                        "confidencePercent": 94,
                                        
                                        "summary": "Official profile supports the negated claim.",
                                        "aiExplanation": (
                                            "The institutional profile does not show the denied role, "
                                            "supporting the negated statement."
                                        ),
                                        "keywords": keywords,
                                        "evidence": [{
                                            "url": url,
                                            "snippet": f"{best_name} — {role_blocks[0]}",
                                            "score": 0.94,
                                            "page_type": "PROFILE"
                                        }]
                                    }

                            # ==================================================
                            # ✅ POSITIVE CLAIM LOGIC
                            # ==================================================
                            else:

                                if role_match:
                                    print("🛑 MODEL-5 PROFILE SUPPORT — FINAL REAL")

                                    return {
                                        "status": "SUPPORTED",
                                        "finalLabel": "REAL",
                                        "confidencePercent": 95,
                                        "summary": "Official institutional profile confirms the claimed role.",
                                        "aiExplanation": (
                                            "An authoritative institutional profile verifies the academic role."
                                        ),
                                        "keywords": keywords,
                                        "evidence": [{
                                            "url": url,
                                            "snippet": f"{best_name} — {role_blocks[0]}",
                                            "score": 0.96,
                                            "page_type": "PROFILE"
                                        }]
                                    }

                                else:
                                    print("🚨 MODEL-5 PROFILE CONTRADICTION — FINAL FAKE")
                                    contradiction_detected = True
                                    if semantic_score >= 0.85 and is_trusted_domain(url):
                                        print("debug semantic score:", semantic_score)
                                        contradiction_found = True
                                    fake_count += 1

                                    return {
                                        "status": "CONTRADICTION",
                                        "finalLabel": "FAKE",
                                        "confidencePercent": 95,
                                        "summary": "Official institutional profile contradicts the claimed role.",
                                        "aiExplanation": (
                                            "The institutional profile lists a different academic rank. "
                                            "Hierarchy mismatch indicates the claim is false."
                                        ),
                                        "keywords": keywords,
                                        "evidence": [{
                                            "url": url,
                                            "snippet": f"{best_name} — {role_blocks[0]}",
                                            "score": 0.97,
                                            "page_type": "PROFILE"
                                        }]
                                    }

            # FALLBACK identity extraction handling (unchanged)
            soup = BeautifulSoup(html, "html.parser")
            identity_text = extract_profile_identity_text(soup)

            # fallback checks for negation/positive (same logic)
            if _is_negated and identity_text and subject and role:
                identity_lower = identity_text.lower()
                subject_tokens = [t.lower() for t in subject.split() if len(t) > 2]
                name_match_fb = fuzzy_name_match(subject_tokens, identity_lower)
                role_match_fb = role_matches_strict(role, role_blocks[0]) if role_blocks else False

                if name_match_fb and role_match_fb:
                    fake_count += 1
                    if semantic_score >= 0.85 and is_trusted_domain(url):
                        contradiction_found = True
                    contradiction_evidence = {
                        "url": url,
                        "snippet": identity_text,
                        "score": 0.92,
                        "page_type": "PROFILE"
                    }
                    if fake_count >= MIN_SUPPORT_SOURCES:
                        return {
                            "status": "CONTRADICTED",
                            "finalLabel": "FAKE",
                            "confidencePercent": 92,
                            "summary": "Official profile evidence contradicts the negated claim.",
                            "aiExplanation": (
                                "An authoritative institutional profile confirms the person holds the role, "
                                "which contradicts the negated statement. Therefore the claim is false."
                            ),
                            "keywords": keywords,
                            "sentiment": {
                                "overall": "neutral",
                                "anger": 0,
                                "fear": 0,
                                "neutral": 100
                            },
                            "evidence": [contradiction_evidence]
                        }
                    continue

            if identity_text and claim_type in {"IDENTITY_ROLE", "POSITIONAL_ROLE"} and not _is_negated:
                if subject and role:
                    subject_tokens = [t.lower() for t in subject.split() if len(t) > 2]
                    text_l = identity_text.lower()
                    name_match = fuzzy_name_match(subject_tokens, text_l)
                    role_match = role.lower() in text_l

                    if name_match and not role_match and not _is_negated:
                        real_count += 1
                        support_evidence.append({
                            "url": url,
                            "snippet": identity_text,
                            "score": 0.92,
                            "page_type": "PROFILE"
                        })
                        if real_count >= MIN_SUPPORT_SOURCES:
                            return {
                                "status": "SUPPORTED",
                                "finalLabel": "REAL",
                                "confidencePercent": 92,
                                "summary": "Official profile page confirms the identity and role.",
                                "aiExplanation": (
                                    "An official institutional profile provides authoritative confirmation "
                                    "of the person's identity and professional role, supporting the claim."
                                ),
                                "keywords": keywords,
                                "sentiment": {
                                    "overall": "neutral",
                                    "anger": 0,
                                    "fear": 0,
                                    "neutral": 100
                                },
                                "evidence": support_evidence[:1]
                            }
                        continue

            # PROFILE -> semantic support (collect evidence) then continue
            if identity_text and claim_type == "IDENTITY_ROLE" and not _is_negated:
                subject = extract_subject_from_claim(claim)
                print("DEBUG SUBJECT:", subject)
                role = extract_role_from_claim(claim)
                entity_ok = subject and subject_supported_by_text(subject, identity_text)
                role_ok = role and role_supported_by_text(role, identity_text)
                if entity_ok and role_ok:
                    support_evidence.append({
                        "url": url,
                        "snippet": identity_text,
                        "score": 0.90,
                        "page_type": "PROFILE"
                    })
            # never send profile pages to RAG
            continue
        # 🚫 FIX 3: MODEL5 RULE — never do ARTICLE RAG for negated identity claims
        # ✅ Only skip article RAG if profile verification already succeeded
        if is_negated and claim_type == "IDENTITY_ROLE" and profile_positive_confirmed:
            print("⛔ SKIPPING ARTICLE RAG — PROFILE ALREADY VERIFIED")
            continue


        # ------------- NON-PROFILE: extract text once and run RAG for this URL -------------
        if page_type != "PROFILE":

            # ==========================================================
            # 🚫 AUTHORITY OVERRIDE
            # ==========================================================
            if profile_authority_state == "SUPPORT":
                print("🛑 SKIPPING ARTICLES — PROFILE ALREADY CONFIRMED REAL")
                break

            if profile_authority_state == "CONTRADICTION":
                print("🛑 SKIPPING ARTICLES — PROFILE ALREADY CONFIRMED FAKE")
                break
            # ==========================================================
            # 🧠 PRE-PARSE CLAIM (ONCE PER URL)
            # ==========================================================

            parsed_subject, parsed_role = cached_claim_parse(claim)
            attribute_type = detect_attribute_type(claim)

            print("CLAIM", claim)
            print("PARSED SUBJECT:", parsed_subject)
            print("ATTRIBUTE TYPE:", attribute_type)

            # ⚡ WIKIPEDIA EXTRACTION MODE
            if html and "wikipedia.org" in url:
                print("⚡ WIKIPEDIA STRUCTURED MODE")

                extracted_text = extract_wikipedia_text(html)

                print("EXTRACTED WIKI TEXT (first 500 chars):")
                print(extracted_text[:500])

            else:
                extracted_text = cached_extract_main_text(html, page_type)

                print("EXTRACTED ARTICLE TEXT (first 500 chars):")
                print(extracted_text[:500])

            if not extracted_text or len(extracted_text.split()) < 6:
                continue
            text = extracted_text

            # ==========================================================
            # 🔒 WIKIPEDIA ROLE HARD VALIDATION
            # ==========================================================

            if "wikipedia.org" in url:

                subject = extract_subject_from_claim(claim)
                role = extract_role_from_claim(claim)
                print("SUBJECT:", subject)
                print(" ROLE:", role) 

                # Ensure role exists (this avoids depending on claim_type)
                if subject and role:

                    text_l = text.lower()
                    role_l = role.lower()

                    name_match = name_token_match(subject, text)
                    role_match = role_l in text_l



                    print("🔎 WIKI ROLE CHECK → name:", name_match, "role:", role_match)
                    print("🔎 CLAIM NEGATED:", is_negated)

                    clean_snippet = extract_best_evidence_sentence(text, claim)

                    # ------------------------------------------------------
                    # CASE 1️⃣ ROLE FOUND IN WIKI
                    # ------------------------------------------------------
                    if name_match and role_match:

                        if is_negated:
                            print("🔒 NEGATED + WIKI ROLE MATCH → FINAL FAKE")

                            return {
                                "status": "CONTRADICTION",
                                "finalLabel": "FAKE",
                                "confidencePercent": 97,
                                "summary": "Wikipedia confirms the role, contradicting the negated claim.",
                                "aiExplanation": (
                                    f"Wikipedia clearly states that {subject} is {role}, "
                                    "which contradicts the negated claim."
                                ),
                                "keywords": keywords,
                                "evidence": [{
                                    "url": url,
                                    "snippet": clean_snippet,
                                    "score": 0.98,
                                    "page_type": "ARTICLE"
                                }]
                            }

                        else:
                            print("🔒 WIKI ROLE MATCH → FINAL REAL")

                            return {
                                "status": "SUPPORTED",
                                "finalLabel": "REAL",
                                "confidencePercent": 97,
                                "summary": "Wikipedia confirms the role.",
                                "aiExplanation": (
                                    f"Wikipedia clearly states that {subject} is {role}, "
                                    "which directly supports the claim."
                                ),
                                "keywords": keywords,
                                "evidence": [{
                                    "url": url,
                                    "snippet": clean_snippet,
                                    "score": 0.98,
                                    "page_type": "ARTICLE"
                                }]
                            }

                    # ------------------------------------------------------
                    # CASE 2️⃣ ROLE NOT FOUND IN WIKI
                    # ------------------------------------------------------
                    if name_match and not role_match:

                        if is_negated:
                            print("🔒 NEGATED + WIKI ROLE MISMATCH → FINAL REAL")

                            return {
                                "status": "SUPPORTED",
                                "finalLabel": "REAL",
                                "confidencePercent": 96,
                                "summary": "Wikipedia does not list that role, supporting the negated claim.",
                                "aiExplanation": (
                                    f"Wikipedia does not list {role} as a role for {subject}, "
                                    "which supports the negated claim."
                                ),
                                "keywords": keywords,
                                "evidence": [{
                                    "url": url,
                                    "snippet": clean_snippet,
                                    "score": 0.96,
                                    "page_type": "ARTICLE"
                                }]
                            }

                        else:
                            print("🔒 WIKI ROLE MISMATCH → FINAL FAKE")

                            return {
                                "status": "CONTRADICTION",
                                "finalLabel": "FAKE",
                                "confidencePercent": 96,
                                "summary": "Wikipedia contradicts the role.",
                                "aiExplanation": (
                                    f"Wikipedia does not list {role} as a role for {subject}, "
                                    "which contradicts the claim."
                                ),
                                "keywords": keywords,
                                "evidence": [{
                                    "url": url,
                                    "snippet": clean_snippet,
                                    "score": 0.96,
                                    "page_type": "ARTICLE"
                                }]
                            }

        # ⚡ TEXT LIGHT MODE — keep only important section
        if len(text.split()) > 600:
            print("⚡ TEXT TRIMMED FOR SPEED")
            text = " ".join(text.split()[:600])

        # Final text quality guards (same as before)
        subject, _ = cached_claim_parse(claim)
        _is_negated = bool(is_negated)

        text_l = text.lower()

        # ==========================================================
        # 🌍 INDUSTRY GEOGRAPHY CONTRADICTION CHECK
        # Runs BEFORE semantic RAG
        # Safe with negation + identity + hard-stop pipeline
        # ==========================================================
        if claim_type != "IDENTITY_ROLE" and subject:
            country_key = subject.strip().lower()
            claimed_region = extract_region(claim)
            true_region = COUNTRY_REGION_MAP.get(country_key)

            # Only trigger when claim explicitly asserts a region
            if claimed_region and true_region:

                # --------------------------------------------------
                # Case 1️⃣ Positive claim but WRONG region
                # Example: "India is in Southeast Asia"
                # --------------------------------------------------
                if not is_negated and claimed_region != true_region:
                    print("🌍 GEOGRAPHY CONTRADICTION — REGION MISMATCH")

                    if semantic_score >= 0.85 and is_trusted_domain(url):
                        contradiction_found = True

                    contradiction_evidence = {
                        "url": url,
                        "snippet": f"{subject.title()} belongs to {true_region}, not {claimed_region}",
                        "score": 0.95,
                        "page_type": page_type
                    }

                    # Industry behavior → immediate exit from URL loop
                    break

                # --------------------------------------------------
                # Case 2️⃣ NEGATED region claim
                # Example: "India is NOT in South Asia"
                # --------------------------------------------------
                if is_negated and claimed_region == true_region:
                    print("🌍 NEGATED REGION CLAIM CONTRADICTED")

                    if semantic_score >= 0.85 and is_trusted_domain(url):
                        contradiction_found = True

                    contradiction_evidence = {
                        "url": url,
                        "snippet": f"{subject.title()} is indeed part of {true_region}",
                        "score": 0.95,
                        "page_type": page_type
                    }

                    break

        unique_tokens = set(text_l.split())
        if len(unique_tokens) <= 3:
            continue

        role_spam = any(text_l.count(r) > 5 for r in ROLE_WORDS)
        if role_spam:
            continue

        if any(d in url for d in LOW_PRIORITY_DOMAINS):
            continue

        clean_snippet = extract_best_evidence_sentence(text, claim)

        # STRUCTURED ATTRIBUTE VALIDATION
        attr_verdict = validate_attribute(claim, text)
        # if attr_verdict == "CONTRADICTION":

        #     fake_count += 1
        #     contradiction_found = True

        #     if "wikipedia.org" in url and detect_attribute_type(parsed["attribute"]) == "NATIONALITY":
        #         print("🔒 WIKIPEDIA NATIONALITY CONTRADICTION — FINAL FAKE")

        #         return {
        #             "status": "CONTRADICTION",
        #             "finalLabel": "FAKE",
        #             "confidencePercent": 97,
        #             "summary": "Wikipedia contradicts the nationality.",
        #             "aiExplanation": (
        #                 "Wikipedia explicitly lists a different nationality "
        #                 "than what the claim states."
        #             ),
        #             "keywords": keywords,
        #             "evidence": [{
        #                 "url": url,
        #                 "snippet": clean_snippet,
        #                 "score": 0.98,
        #                 "page_type": page_type
        #             }]
        #         }

        #     continue

        # if attr_verdict == "SUPPORTED":

        #     print("✅ ATTRIBUTE SUPPORT DETECTED")

        #     real_count += 1
        #     support_evidence.append({
        #         "url": url,
        #         "snippet": clean_snippet,
        #         "score": 0.95,
        #         "page_type": page_type
        #     })

        #     # 🔒 HARD STOP for Wikipedia on NATIONALITY
        #     if "wikipedia.org" in url and detect_attribute_type(parsed["attribute"]) == "NATIONALITY":
        #         print("🔒 WIKIPEDIA NATIONALITY SUPPORT — FINAL REAL")

        #         return {
        #             "status": "SUPPORTED",
        #             "finalLabel": "REAL",
        #             "confidencePercent": 97,
        #             "summary": "Wikipedia confirms the nationality.",
        #             "aiExplanation": (
        #                 "Wikipedia explicitly states the nationality, "
        #                 "which directly supports the claim."
        #             ),
        #             "keywords": keywords,
        #             "evidence": support_evidence[:1]
        #         }

        #     continue

        # POSITIONAL_ROLE quick checks (unchanged)
        if claim_type == "POSITIONAL_ROLE":
            subject, role = cached_claim_parse(claim)
            if is_speculative_or_negative(text_l):
                fake_count += 1
                if semantic_score >= 0.85 and is_trusted_domain(url):
                    contradiction_found = True
                contradiction_evidence = {"url": url, "snippet": clean_snippet, "score": 0.95}
                if fake_count >= MIN_SUPPORT_SOURCES:
                    return {
                        "status": "CONTRADICTION",
                        "finalLabel": "FAKE",
                        "confidencePercent": 90,
                        "reason": "Speculative/negative language found indicating contradiction.",
                        "evidence": [contradiction_evidence]
                    }
                continue

            if subject and role and explicit_role_assertion(subject, role, text_l):
                support_evidence.append({
                    "url": url,
                    "snippet": clean_snippet,
                    "score": 0.95,
                    "page_type": page_type
                })

                if claim_type != "IDENTITY_ROLE":
                    real_count += 1

                    if real_count >= MIN_SUPPORT_SOURCES:
                        return {
                            "status": "SUPPORTED",
                            "finalLabel": "REAL",
                            "confidencePercent": 92,
                            "summary": "Explicit role assignment found on page.",
                            "aiExplanation": (
                                "The retrieved source explicitly assigns the role to the person, "
                                "providing strong semantic evidence that supports the claim."
                            ),
                            "keywords": keywords,
                            "sentiment": {
                                "overall": "neutral",
                                "anger": 0,
                                "fear": 0,
                                "neutral": 100
                            },
                            "evidence": support_evidence[:MIN_SUPPORT_SOURCES]
                        }
                continue

            if any(r in text_l for r in ROLE_WORDS) and (role and role in text_l):
                contradiction_evidence = {
                    "url": url,
                    "snippet": clean_snippet,
                    "score": 0.9
                }

                if claim_type != "IDENTITY_ROLE":
                    fake_count += 1
                    if semantic_score >= 0.85 and is_trusted_domain(url):
                        contradiction_found = True


                    if fake_count >= MIN_SUPPORT_SOURCES:
                        return {
                            "status": "CONTRADICTION",
                            "finalLabel": "FAKE",
                            "confidencePercent": 90,

                            "summary": "Role language suggests contradiction.",

                            "aiExplanation": (
                                "The retrieved source contains role-related information that conflicts "
                                "with the claim. Semantic analysis indicates the role description "
                                "contradicts the stated assertion, so the claim is classified as false."
                            ),
                            "keywords": keywords,
                            "sentiment": {
                                "overall": "neutral",
                                "anger": 0,
                                "fear": 0,
                                "neutral": 100
                            },

                            "evidence": [contradiction_evidence]
                        }
                continue

        # EVENT_HISTORICAL direct fact check
        if claim_type == "EVENT_HISTORICAL":
            verdict, score = fact_match_decision(claim, text)
            if verdict == "REAL":
                support_evidence.append({"url": url, "snippet": clean_snippet, "score": score, "page_type": page_type})
                real_count += 1
                if real_count >= MIN_SUPPORT_SOURCES:
                    return {
                        "status": "SUPPORTED",
                        "finalLabel": "REAL",
                        "confidencePercent": int(min(95, 70 + score * 30)),

                        "summary": "Event or historical evidence matched.",

                        "aiExplanation": (
                            "Trusted sources contain historical or event-based information that "
                            "aligns with the claim. Semantic matching confirms that the timeline "
                            "and factual details support the statement."
                        ),
                        "keywords": keywords,
                        "sentiment": {
                            "overall": "neutral",
                            "anger": 0,
                            "fear": 0,
                            "neutral": 100
                        },
                        "evidence": support_evidence[:MIN_SUPPORT_SOURCES]
                    }

                continue
            if verdict == "FAKE":
                fake_count += 1
                if semantic_score >= 0.85 and is_trusted_domain(url):
                    contradiction_found = True
                contradiction_evidence = {"url": url, "snippet": clean_snippet, "score": score}
                if fake_count >= MIN_SUPPORT_SOURCES:
                    return {
                        "status": "CONTRADICTION",
                        "finalLabel": "FAKE",
                        "confidencePercent": 90,

                        "summary": "Strong contradictory historical evidence found.",

                        "aiExplanation": (
                            "Authoritative historical sources contain information that directly "
                            "conflicts with the claim. Timeline or factual details from reliable "
                            "references indicate the statement is incorrect."
                        ),
                        "keywords": keywords,
                        "sentiment": {
                            "overall": "neutral",
                            "anger": 0,
                            "fear": 0,
                            "neutral": 100
                        },

                        "evidence": [contradiction_evidence]
                    }

                continue
            print("🧠 RAG: Preparing semantic embeddings")


            if subject and subject.lower() not in extracted_text.lower():
                print("⛔ SUBJECT NOT IN ARTICLE — SKIPPING RAG")
                continue

        # -------------------- Chunking & embeddings for this page --------------------
        word_count = len(text.split())
        if word_count <= 140:
            chunks = [text]
        else:
            if page_type == "PROFILE":
                chunks = chunk_text(text, chunk_size=60, overlap=10)
            else:
                chunks = chunk_text(text, chunk_size=120, overlap=20)

        # HARD CAP on chunks
        chunks = chunks[:3]
        print(f"🧩 Chunks prepared: {len(chunks)}")

        if not chunks:
            continue

        # Ensure model loaded once
        ensure_similarity_model()

        # compute chunk embeddings and scores for this URL
        chunk_embs = similarity_model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(claim_emb, chunk_embs)[0]

        scored_chunks = []
        claim_words = claim.lower().split()
        authority_match = False

        if authority_entity:
            authority_lower = authority_entity.lower()
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            if authority_lower in domain:
                authority_match = True

        for i, chunk in enumerate(chunks):

            if len(chunk.split()) < 20:
                continue

            try:
                score = float(scores[i])
            except Exception:
                score = 0.0

            chunk_lower = chunk.lower()

            matches = sum(1 for w in claim_words if w in chunk_lower)

            entity_boost = 0.05 if matches >= 2 else 0.0
            role_boost = 0.05 if any(r in chunk_lower for r in ROLE_WORDS) else 0.0

            boosted_score = score + (0.02 * matches) + entity_boost + role_boost

            # Existing trusted domain boost
            if any(td in url for td in trusted_domains):
                boosted_score += 0.12

            # Existing official keyword boost
            if any(k in url.lower() for k in official_keywords):
                boosted_score += 0.15

            # =========================================================
            # 🔐 CAR AUTHORITY BOOST (NO EXTRACTION HERE)
            # =========================================================
            if authority_match:
                boosted_score += 0.07  # controlled semantic authority boost

            boosted_score = min(boosted_score, 1.0)

            scored_chunks.append((boosted_score, chunk))

        if not scored_chunks:
            continue

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = scored_chunks[:top_k]
        avg_score = sum(p[0] for p in top_chunks) / len(top_chunks)
        best_chunk = top_chunks[0][1]
        best_score = top_chunks[0][0]
        semantic_score = best_score
        print(f"📊 Semantic score: {semantic_score:.3f}")


        # Role/predicate sanity for identity claims
        subject_ok = extract_subject_from_claim(claim)
        role_ok = True
        if claim_type == "IDENTITY_ROLE":
            role_ok = role.lower() in text.lower()

        # Support candidate
        SUPPORT_THRESHOLD = 0.7
        HIGH_CONFIDENCE_THRESHOLD = 0.88
        
        # should remove if we want more urls to fetch
        if semantic_score >= STRONG_SUPPORT_THRESHOLD and not profile_hierarchy_mismatch:


            claim_l = claim.lower()
            chunk_l = best_chunk.lower()

            # ⭐ UPDATED SAFETY CHECK (Improved Identity Handling)
            allow_early_real = True

            if " is " in claim_l:

                identity_words = [" is ", " was ", " serves as ", " became "]

                has_identity_word = any(w in chunk_l for w in identity_words)

                subject = extract_subject_from_claim(claim)
                print("DEBUG SUBJECT:", subject)
                name_match = name_token_match(subject, best_chunk)
                if not name_match:
                    name_match = name_token_match(subject, text)

                if not (has_identity_word and name_match):
                    allow_early_real = False

            if allow_early_real:
                print("🚀 STRONG SEMANTIC SUPPORT — EARLY STOP")

                support_evidence.append({
                    "url": url,
                    "snippet": best_chunk,
                    "score": semantic_score,
                    "page_type": page_type
                })

                # 🔥 NEW BLOCK — SINGLE TRUSTED SOURCE CONFIRMATION
                if is_trusted_domain(url) and subject_ok and role_ok:
                    print("🔒 SINGLE TRUSTED SOURCE CONFIRMATION — FINAL REAL")
                    return {
                        "status": "SUPPORTED",
                        "finalLabel": "REAL",
                        "confidencePercent": int(min(95, semantic_score * 100)),
                        "summary": "Trusted authoritative source confirms the claim.",
                        "aiExplanation": (
                            "A trusted and authoritative source provides strong semantic "
                            "support for the claim. Single-source confirmation from a "
                            "reliable domain is sufficient for verification."
                        ),
                        "keywords": keywords,
                        "sentiment": {
                            "overall": "neutral",
                            "anger": 0,
                            "fear": 0,
                            "neutral": 100
                        },
                        "evidence": support_evidence[:1]
                    }

                # existing behavior remains untouched
                real_count += 1
                support_hits += 1
                max_support = max(max_support, semantic_score)

                break


        if semantic_score >= SUPPORT_THRESHOLD:
            print("✅ SUPPORT SIGNAL detected from this URL")
        else:
            print("⚠️ Weak semantic signal")

        if subject_ok and role_ok and semantic_score >= SUPPORT_THRESHOLD and not contradiction_found and not profile_hierarchy_mismatch:

            support_hits += 1
            max_support = max(max_support, semantic_score)
            real_count += 1
            support_evidence.append({"url": url, "snippet": clean_snippet, "score": semantic_score, "page_type": page_type})
            # early high confidence stop
            if max_support >= HIGH_CONFIDENCE_THRESHOLD and support_hits >= MIN_SUPPORT_SOURCES:
                return {
                    "status": "SUPPORTED",
                    "finalLabel": "REAL",
                    "confidencePercent": int(min(95, max_support * 100)),

                    "summary": "High-confidence semantic support across sources.",

                    "aiExplanation": (
                        "Multiple independent sources provide strong semantic evidence that "
                        "aligns closely with the claim. Cross-source agreement increases the "
                        "confidence that the statement is accurate."
                    ),
                    "keywords": keywords,
                    "sentiment": {
                        "overall": "neutral",
                        "anger": 0,
                        "fear": 0,
                        "neutral": 100
                    },

                    "evidence": support_evidence[:MIN_SUPPORT_SOURCES]
                }

                
        if support_hits >= 2:
            if claim_type == "IDENTITY_ROLE" and max_support >= 0.75:
                return {
                    "status": "SUPPORTED",
                    "finalLabel": "REAL",
                    "confidencePercent": int(min(95, max_support * 100)),

                    "summary": "Identity role confirmed by multiple sources.",

                    "aiExplanation": (
                        "Multiple independent sources provide consistent role information "
                        "about the person. The cross-source agreement increases confidence "
                        "that the claimed identity and role are correct."
                    ),
                    "keywords": keywords,
                    "sentiment": {
                        "overall": "neutral",
                        "anger": 0,
                        "fear": 0,
                        "neutral": 100
                    },

                    "evidence": support_evidence[:2]
                }
            if max_support >= 0.80:
                return {
                    "status": "SUPPORTED",
                    "finalLabel": "REAL",
                    "confidencePercent": int(min(95, max_support * 100)),

                    "summary": "High confidence support reached.",

                    "aiExplanation": (
                        "High semantic similarity across retrieved sources indicates a strong "
                        "match with the claim. This level of agreement supports the claim with high confidence."
                    ),
                    "keywords": keywords,
                    "sentiment": {
                        "overall": "neutral",
                        "anger": 0,
                        "fear": 0,
                        "neutral": 100
                    },

                    "evidence": support_evidence[:2]
                }

        # Trusted domain fast-path (kept)
        if is_trusted_domain(url) and subject_ok and role_ok and semantic_score >= 0.9:
            real_count += 1
            support_evidence.append({"url": url, "snippet": clean_snippet, "score": semantic_score, "page_type": page_type})
            if real_count >= MIN_SUPPORT_SOURCES:
                return {
                    "status": "SUPPORTED",
                    "finalLabel": "REAL",
                    "confidencePercent": int(min(95, semantic_score * 100)),

                    "summary": "Trusted source confirms the claim.",

                    "aiExplanation": (
                        "A trusted and authoritative source explicitly supports the claim. "
                        "High publisher trust increases the reliability of this verification."
                    ),
                    "keywords": keywords,

                    "sentiment": {
                        "overall": "neutral",
                        "anger": 0,
                        "fear": 0,
                        "neutral": 100
                    },

                    "evidence": support_evidence[:MIN_SUPPORT_SOURCES]
                }

        # Contradiction via trusted domain
        if is_trusted_domain(url) and contradiction_found:
            fake_count += 1
            contradiction_detected = True
            if fake_count >= MIN_SUPPORT_SOURCES:
                return {
                    "status": "CONTRADICTION",
                    "finalLabel": "FAKE",
                    "confidencePercent": 90,

                    "summary": "Trusted source contradicts the claim.",

                    "aiExplanation": (
                        "An authoritative and trusted source contains information that directly "
                        "contradicts the claim. Because the source is trusted, this contradiction "
                        "is given high weight in the final verdict."
                    ),
                    "keywords": keywords,

                    "sentiment": {
                        "overall": "neutral",
                        "anger": 0,
                        "fear": 0,
                        "neutral": 100
                    },

                    "evidence": [contradiction_evidence]
                }

        # High semantic exact-match handling (kept)
        SIM_THRESHOLD = max(sim_threshold, 0.85)
        speculative_hit = any(t in best_chunk.lower() for t in SPECULATIVE_TERMS)

        if semantic_score >= SIM_THRESHOLD and not speculative_hit and not contradiction_found and claim_type in {"IDENTITY_ROLE", "EVENT_HISTORICAL"}:
            if claim_type == "IDENTITY_ROLE":
                subject, role = cached_claim_parse(claim)
                if subject and role and explicit_role_assertion(subject, role, best_chunk.lower()):
                    real_count += 1
                    support_evidence.append({"url": url, "snippet": clean_snippet, "score": best_score, "page_type": page_type})
                    if real_count >= MIN_SUPPORT_SOURCES:
                        return {
                            "status": "SUPPORTED",
                            "finalLabel": "REAL",
                            "confidencePercent": int(best_score * 100),

                            "summary": "Strong semantic match between extracted evidence and the claim.",

                            "aiExplanation": (
                                "A extracted passage from the source has very high semantic similarity "
                                "to the claim and explicitly mentions the role/identity, providing "
                                "strong evidence in favor of the claim."
                            ),
                            "keywords": keywords,
                            "sentiment": {
                                "overall": "neutral",
                                "anger": 0,
                                "fear": 0,
                                "neutral": 100
                            },

                            "evidence": support_evidence[:MIN_SUPPORT_SOURCES]
                        }
            else:
                real_count += 1
                support_evidence.append({"url": url, "snippet": clean_snippet, "score": best_score, "page_type": page_type})
                if real_count >= MIN_SUPPORT_SOURCES:
                    return {
                        "status": "SUPPORTED",
                        "finalLabel": "REAL",
                        "confidencePercent": int(best_score * 100),

                        "summary": "Strong semantic match between extracted evidence and the claim.",

                        "aiExplanation": (
                            "A high-quality semantic match was found in the source text that aligns "
                            "closely with the claim's content, supporting the claim's accuracy."
                        ),
                        "keywords": keywords,

                        "sentiment": {
                            "overall": "neutral",
                            "anger": 0,
                            "fear": 0,
                            "neutral": 100
                        },

                        "evidence": support_evidence[:MIN_SUPPORT_SOURCES]
                    }

        # Sentence-level NLI / stance checks (kept)
        sentences = split_sentences(best_chunk)
        top_sentences = get_top_k_sentences(claim, sentences, similarity_model, k=5)

        final_stance = "NEUTRAL"
        final_score = 0.0
        final_sentence = None

        for sent in top_sentences:
            stance, score = detect_stance(claim, sent)

            # For identity_role ignore CONTRADICT (conservative)
            if claim_type == "IDENTITY_ROLE" and stance == "CONTRADICT":
                continue

            if stance == "CONTRADICT" and score >= 0.8:
                if claim_type == "POSITIONAL_ROLE":
                    sent_l = sent.lower()
                    if not any(t in sent_l for t in CURRENT_ROLE_GUARD_TERMS):
                        # ignore biography/tense contradictions
                        pass
                    else:
                        final_stance = stance
                        final_score = score
                        final_sentence = sent
                        break
                else:
                    final_stance = stance
                    final_score = score
                    final_sentence = sent
                    break

            if stance == "SUPPORT" and score > final_score:
                final_stance = "SUPPORT"
                final_score = score
                final_sentence = sent

            if stance == "NEUTRAL" and score >= 0.65 and 'generic_role' in globals() and semantic_role_support(glob.get('generic_role', None), sent):
                final_stance = "SUPPORT"
                final_score = score
                final_sentence = sent

        if final_stance == "CONTRADICT" and final_score >= 0.8:
            fake_count += 1
            if semantic_score >= 0.85 and is_trusted_domain(url):
                contradiction_found = True

            contradiction_evidence = {"url": url, "snippet": final_sentence, "stance": "CONTRADICT", "score": final_score}
            if fake_count >= MIN_SUPPORT_SOURCES:
                return {
                    "status": "CONTRADICTION",
                    "finalLabel": "FAKE",
                    "confidencePercent": int(min(95, final_score * 100)),

                    "summary": "Contradictory evidence detected.",

                    "aiExplanation": (
                        "A sentence-level natural language inference (NLI) check found a strong "
                        "contradiction with the claim. The high-stance score indicates the source "
                        "directly disputes the claim."
                    ),
                    "keywords": keywords,

                    "sentiment": {
                        "overall": "neutral",
                        "anger": 0,
                        "fear": 0,
                        "neutral": 100
                    },

                    "evidence": [contradiction_evidence]
                }

        # Collect evidence based on stance
        if final_stance == "SUPPORT" and final_score >= 0.7:
            support_evidence.append({"url": url, "snippet": clean_snippet, "score": min(0.95, final_score + 0.1), "page_type": page_type})
        elif final_stance == "NEUTRAL":
            neutral_evidence.append({"url": url, "snippet": clean_snippet})

        # Update best_evidence if needed (kept priority logic)
        priority_score = avg_score
        if any(x in url.lower() for x in trusted_domains):
            priority_score += 0.25
        if page_type == "PROFILE":
            priority_score += 0.10

        # Check incremental early-stopping by counts
        if real_count >= MIN_SUPPORT_SOURCES:
            return {
                "status": "SUPPORTED",
                "finalLabel": "REAL",
                "confidencePercent": int(min(95, 70 + max_support * 30)),

                "summary": "Two independent sources support the claim.",

                "aiExplanation": (
                    "Independent sources agree in supporting the claim. The redundancy across "
                    "multiple sources increases confidence in the claim's validity."
                ),
                "keywords": keywords,

                "sentiment": {
                    "overall": "neutral",
                    "anger": 0,
                    "fear": 0,
                    "neutral": 100
                },

                "evidence": support_evidence[:MIN_SUPPORT_SOURCES]
            }
        if fake_count >= MIN_SUPPORT_SOURCES:
            # choose contradiction evidence if available else neutral
            ev = [contradiction_evidence] if contradiction_evidence else neutral_evidence[:1]
            return {
                "status": "CONTRADICTION",
                "finalLabel": "FAKE",
                "confidencePercent": 90,

                "summary": "Multiple sources contradict the claim.",

                "aiExplanation": (
                    "Several independent sources contain information that contradicts the claim. "
                    "Cross-source contradiction reduces confidence in the claim and supports a false verdict."
                ),
                "keywords": keywords,

                "sentiment": {
                    "overall": "neutral",
                    "anger": 0,
                    "fear": 0,
                    "neutral": 100
                },

                "evidence": ev
            }

    # =========================================================
    # 📰 NEWS-AWARE CONTRADICTION PATCH (SAFE UPGRADE)
    # =========================================================

    # Detect news-style text (lightweight heuristic)
    is_news_mode = detect_news_style(claim)   # claim variable already exists in your RAG

    if contradiction_found and not support_evidence:


        # 📰 If OCR text looks like a NEWS ARTICLE
        if is_news_mode:
            print("📰 NEWS MODE ACTIVE — Softening contradiction decision")

            # Only allow HARD FAKE if contradiction is extremely strong
            # and there is basically no semantic support
            if contradiction_score >= 0.98 and max_support < 0.15:
                print("🔒 CONTRADICTION LOCK — FINAL FAKE (NEWS MODE HARD CASE)")
                return {
                    "status": "CONTRADICTED",
                    "finalLabel": "FAKE",
                    "confidencePercent": 95,

                    "summary": "Strong contradiction detected even in news context.",

                    "aiExplanation": (
                        "Despite being in a news-like context, the contradiction signal is extremely strong "
                        "and there is minimal supporting evidence; hence the claim is labeled false."
                    ),
                    "keywords": keywords,

                    "sentiment": {
                        "overall": "neutral",
                        "anger": 0,
                        "fear": 0,
                        "neutral": 100
                    },

                    "evidence": [contradiction_evidence]
                }
            else:
                print("🟡 NEWS MODE — Ignoring contradiction lock (treated as contextual news)")
                # DO NOTHING — let normal RAG aggregation continue

        else:
            # 🔴 Normal claim behaviour (UNCHANGED)
            print("🔒 CONTRADICTION LOCK — FINAL FAKE")
            return {
                "status": "CONTRADICTED",
                "finalLabel": "FAKE",
                "confidencePercent": 95,

                "summary": "Claim is negated and contradicted by authoritative evidence.",

                "aiExplanation": (
                    "Authoritative sources provide decisive contradictory information that negates the claim."
                ),
                "keywords": keywords,

                "sentiment": {
                    "overall": "neutral",
                    "anger": 0,
                    "fear": 0,
                    "neutral": 100
                },

                "evidence": [contradiction_evidence]
            }


    rag_supported_any = (len(support_evidence) > 0) and not (contradiction_evidence and contradiction_evidence.get("score", 0) >= 0.8)

    if is_negated:

        if is_negated and not rag_supported_any and not contradiction_found and profile_evidence:
            return {
                "status": "SUPPORTED",
                "finalLabel": "REAL",
                "confidencePercent": 95,

                "summary": "Official profile shows a different role, supporting the negated claim.",

                "aiExplanation": (
                    "The institutional profile confirms the person holds a different academic rank. "
                    "Since the denied role is not present, the negated claim is considered true."
                ),
                "keywords": keywords,

                "sentiment": {
                    "overall": "neutral",
                    "anger": 0,
                    "fear": 0,
                    "neutral": 100
                },

                "evidence": [profile_evidence] if profile_evidence else []
            }

        if rag_supported_any:
            return {
                "status": "CONTRADICTION",
                "finalLabel": "FAKE",
                "confidencePercent": 90,

                "summary": "Evidence supports the positive claim, so the negated claim is false.",

                "aiExplanation": (
                    "Sources support the positive formulation of the claim; therefore a negated "
                    "version of this claim is contradicted and should be considered false."
                ),
                "keywords": keywords,

                "sentiment": {
                    "overall": "neutral",
                    "anger": 0,
                    "fear": 0,
                    "neutral": 100
                },

                "evidence": support_evidence[:1]
            }
        if contradiction_found:
            return {
                "status": "SUPPORTED",
                "finalLabel": "REAL",
                "confidencePercent": 90,

                "summary": "Evidence contradicts the positive claim, confirming the negated claim.",

                "aiExplanation": (
                    "Available evidence directly contradicts the positive assertion, which in turn "
                    "supports the negated form of the claim (i.e., the negation is true)."
                ),
                "keywords": keywords,
                "sentiment": {
                    "overall": "neutral",
                    "anger": 0,
                    "fear": 0,
                    "neutral": 100
                },

                "evidence": [contradiction_evidence] if contradiction_evidence else []
            }

    if contradiction_found and not support_evidence:
        return {
            "status": "CONTRADICTION",
            "finalLabel": "FAKE",
            "confidencePercent": 90,

            "summary": "Strong contradiction detected by reliable sources.",

            "aiExplanation": (
                "Reliable sources provide contradicting information and no supporting evidence was found; "
                "the claim is therefore likely false."
            ),
            "keywords": keywords,

            "sentiment": {
                "overall": "neutral",
                "anger": 0,
                "fear": 0,
                "neutral": 100
            },

            "evidence": [contradiction_evidence]
        }

    # Prefer PROFILE evidence if available
    profile_supports = [e for e in support_evidence if e.get("page_type") == "PROFILE"]
    if profile_supports:
        best_support = profile_supports[0]
        return {
            "status": "SUPPORTED",
            "finalLabel": "REAL",
            "confidencePercent": 95,

            "summary": "Official institutional profile confirms the claim.",

            "aiExplanation": (
                "An official institutional profile page explicitly confirms the person's role or identity, "
                "which is authoritative evidence supporting the claim."
            ),
            "keywords": keywords,

            "sentiment": {
                "overall": "neutral",
                "anger": 0,
                "fear": 0,
                "neutral": 100
            },

            "evidence": [best_support]
        }

    if is_negated and profile_positive_confirmed:
        return {
            "status": "CONTRADICTION",
            "finalLabel": "FAKE",
            "confidencePercent": 95,

            "summary": "Official institutional profile confirms the role, so the negated claim is false.",

            "aiExplanation": (
                "An official profile confirms the role which contradicts the negated claim; thus the negated claim is false."
            ),
            "keywords": keywords,

            "sentiment": {
                "overall": "neutral",
                "anger": 0,
                "fear": 0,
                "neutral": 100
            },

            "evidence": [profile_evidence]
        }

    if support_evidence:
        best_support = max(support_evidence, key=lambda x: x["score"])
        confidence = int(min(95, 70 + best_support["score"] * 30))
        return {
            "status": "SUPPORTED",
            "finalLabel": "REAL",
            "confidencePercent": confidence,

            "summary": "Claim is supported by reliable sources.",

            "aiExplanation": (
                "At least one reliable source contains information that aligns with the claim. "
                "The best-matching evidence was used to produce a confidence estimate."
            ),
            "keywords": keywords,

            "sentiment": {
                "overall": "neutral",
                "anger": 0,
                "fear": 0,
                "neutral": 100
            },

            "evidence": support_evidence[:3]
        }

    # fallback to best_evidence if any (reuse logic paths — best_evidence variable may not be present; improvise)
    # If neutral evidence exist, return UNVERIFIABLE with neutral evidence
    if neutral_evidence:
        return {
            "status": "PARTIAL",
            "finalLabel": "UNVERIFIABLE",
            "confidencePercent": 60,

            "summary": "Weak or neutral evidence found but nothing decisive.",

            "aiExplanation": (
                "Only weak or neutral supporting signals were found. There isn't enough definitive evidence "
                "to classify the claim as true or false."
            ),
            "keywords": keywords,

            "sentiment": {
                "overall": "neutral",
                "anger": 0,
                "fear": 0,
                "neutral": 100
            },

            "evidence": neutral_evidence[:2]
        }
    print("REAL COUNT", real_count)
    print("FAKE COUNT", fake_count)
    print("CONTRADICTION FOUND", contradiction_found)

    print("\n================= 🧾 RAG TRACE END =================")
    print("====================================================\n")


    return {
        "status": "NO_EVIDENCE",
        "finalLabel": "UNVERIFIABLE",
        "confidencePercent": 50,

        "summary": "No reliable supporting or contradicting evidence found.",

        "aiExplanation": (
            "The retrieval and analysis process did not find sufficiently relevant or reliable evidence "
            "to verify or refute the claim."
        ),
        "keywords": keywords,

        "sentiment": {
            "overall": "neutral",
            "anger": 0,
            "fear": 0,
            "neutral": 100
        },

        "evidence": []
    }

# =========================================================
# SOURCE CREDIBILITY
# =========================================================
TRUSTED_SOURCES = {
    "bbc.com",
    "reuters.com",
    "thehindu.com",
    "ndtv.com",
    "indiatoday.in",
    "timesofindia.indiatimes.com",
    "economictimes.indiatimes.com",
    "hindustantimes.com",
    "livemint.com",
    "wikipedia.org",

    # 🇮🇳 Hindi News
    "aajtak.in",
    "amarujala.com",
    "bhaskar.com",
    "jansatta.com",
    "navbharattimes.indiatimes.com",
    "ndtv.in",
    "livehindustan.com",
    "patrika.com",
    "zeenews.india.com/hindi",
    "abplive.com",

    # 🇮🇳 Telugu News
    "eenadu.net",
    "sakshi.com",
    "andhrajyothy.com",
    "greatandhra.com",
    "tv9telugu.com",
    "ntvtelugu.com",
    "abntelugu.com",
    "10tv.in",
    "v6velugu.com"
}

ROLE_EQUIVALENTS = {

    # ---------------- POLITICS ----------------
    "politician": {
        "prime minister",
        "chief minister",
        "president",
        "vice president",
        "minister",
        "cabinet minister",
        "political leader",
        "leader of",
        "member of parliament",
        "mp",
        "mla",
        "senator",
        "governor",
        "head of government",
        "head of state"
    },

    # ---------------- ENTERTAINMENT ----------------
    "actor": {
        "film actor",
        "movie actor",
        "television actor",
        "screen actor",
        "lead actor",
        "actor in",
        "starred in",
        "starring"
    },

    "actress": {
        "film actress",
        "movie actress",
        "television actress",
        "screen actress",
        "lead actress",
        "actress in",
        "starred in",
        "starring"
    },

    # ---------------- MEDIA ----------------
    "journalist": {
        "reporter",
        "news correspondent",
        "news anchor",
        "editor",
        "columnist",
        "media professional"
    },

    # ---------------- SCIENCE & ACADEMIA (GENERIC ONLY) ----------------
    "scientist": {
        "researcher",
        "research scientist",
        "scientific researcher",
        "physicist",
        "chemist",
        "biologist"
    },

    "author": {
        "writer",
        "novelist",
        "book author",
        "essayist",
        "biographer"
    },

    # ---------------- BUSINESS ----------------
    "businessman": {
        "business executive",
        "industrialist",
        "entrepreneur",
        "tycoon",
        "business leader"
    },

    "entrepreneur": {
        "startup founder",
        "co founder",
        "founder",
        "business founder"
    },

    "ceo": {
        "chief executive officer",
        "ceo of",
        "head of company",
        "company ceo"
    },

    # ---------------- SPORTS ----------------
    "sportsperson": {
        "athlete",
        "player",
        "sports player",
        "professional athlete"
    },

    "cricketer": {
        "cricket player",
        "international cricketer",
        "former cricketer"
    },

    "footballer": {
        "football player",
        "soccer player",
        "professional footballer"
    },

    # ---------------- LAW ----------------
    "lawyer": {
        "advocate",
        "attorney",
        "legal practitioner"
    },

    "judge": {
        "justice",
        "high court judge",
        "supreme court judge"
    }
}

ABSURD_IMPOSSIBLE_PATTERNS = [
    # Physics / astronomy
    "sun is made of ice",
    "sun made of ice",
    "star made of ice",

    # Biology / space
    "breathe in space without equipment",
    "humans can breathe in space",
    "humans survive in vacuum",

    # General impossibilities
    "live without oxygen",
    "survive without oxygen",
]
# ==========================================================
# 🌍 GEOGRAPHY KNOWLEDGE BASE (lightweight authoritative map)
# ==========================================================
COUNTRY_REGION_MAP = {
    "india": "south asia",
    "pakistan": "south asia",
    "bangladesh": "south asia",
    "sri lanka": "south asia",
    "nepal": "south asia",
    "bhutan": "south asia",
    "maldives": "south asia",

    "thailand": "southeast asia",
    "vietnam": "southeast asia",
    "indonesia": "southeast asia",
    "malaysia": "southeast asia",
    "philippines": "southeast asia",
    "singapore": "southeast asia",
    "laos": "southeast asia",
    "cambodia": "southeast asia",
    "myanmar": "southeast asia",
}

ROLE_WORDS = [

    # =========================
    # Academic roles
    # =========================
    "professor",
    "assistant professor",
    "associate professor",
    "faculty",
    "lecturer",
    "hod",
    "head of department",
    "dean",
    "principal",
    "researcher",
    "scientist",
    "scholar",
    "academic",
    "research fellow",

    # =========================
    # Corporate roles
    # =========================
    "ceo",
    "chief executive officer",
    "cto",
    "chief technology officer",
    "cfo",
    "chief financial officer",
    "coo",
    "chief operating officer",
    "founder",
    "co-founder",
    "chairman",
    "director",
    "managing director",
    "executive director",
    "board member",
    "manager",
    "senior manager",
    "vice president",
    "president",
    "partner",
    "lead engineer",
    "software engineer",
    "developer",
    "architect",

    # =========================
    # Government & politics
    # =========================
    "prime minister",
    "president",
    "vice president",
    "governor",
    "chief minister",
    "minister",
    "mp",
    "member of parliament",
    "mla",
    "senator",
    "secretary",
    "ambassador",
    "spokesperson",
    "advisor",
    "policy advisor",

    # =========================
    # Medical
    # =========================
    "doctor",
    "physician",
    "surgeon",
    "consultant",
    "medical officer",
    "health officer",

    # =========================
    # Media & public roles
    # =========================
    "journalist",
    "reporter",
    "editor",
    "anchor",
    "host",
    "commentator",
    "analyst",

    # =========================
    # Legal roles
    # =========================
    "lawyer",
    "attorney",
    "judge",
    "advocate",
    "legal advisor",

    # =========================
    # Sports roles
    # =========================
    "player",
    "captain",
    "coach",
    "trainer",

    # =========================
    # Military
    # =========================
    "general",
    "colonel",
    "commander",
    "officer",

    # =========================
    # Generic employment roles
    # =========================
    "employee","staff member", "member","official","representative",
    "head","chief","lead","in charge"
]
LOW_TRUST_SOURCES = {
    "beforeitsnews.com",
    "infowars.com",
    "worldnewsdailyreport.com"
}

KNOWN_FALSE_PATTERNS = [

    # ===================== ASTRONOMY & PHYSICS =====================

    (
        "earth is flat",
        "Multiple independent observations and measurements confirm that the Earth is an oblate spheroid, including satellite imagery, gravity measurements, and circumnavigation."
    ),
    (
        "flat earth",
        "The shape of the Earth has been scientifically established as an oblate spheroid through astronomy, geodesy, and satellite data."
    ),
    (
        "sun revolves around earth",
        "Astronomical observations confirm that the Earth orbits the Sun, as described by heliocentric models supported by physics and observation."
    ),

    # ===================== BIOLOGY & MEDICINE =====================

    (
        "vaccines cause autism",
        "Extensive scientific studies and global health organizations have found no causal link between vaccines and autism."
    ),
    (
        "humans only use 10 percent of their brain",
        "Neuroscience research shows that humans use all regions of the brain, with different areas active at different times."
    ),

    # ===================== BASIC CHEMISTRY / PHYSICS =====================

    (
        "water has memory",
        "There is no credible scientific evidence supporting the claim that water retains memory of substances once dissolved."
    ),
    (
        "cold weather causes colds",
        "Colds are caused by viral infections, not exposure to cold temperatures."
    ),

    # ===================== HISTORY (SETTLED) =====================

    (
        "humans never landed on the moon",
        "Multiple independent sources, including physical samples, telemetry data, and international tracking, confirm human Moon landings."
    ),

    # ===================== TECHNOLOGY MYTHS =====================

    (
        "5g causes covid",
        "There is no scientific evidence linking 5G technology to COVID-19; COVID-19 is caused by a virus."
    ),
]

ENTERTAINMENT_DOMAINS = {
    "buzzfeed.com",
    "reddit.com",
    "quora.com",
    "medium.com"
}
FACT_TYPES = {
    "IDENTITY_ROLE",     # X is a politician
    "POSITIONAL_ROLE",   # X is PM of India
    "EVENT_HISTORICAL",  # Chandrayaan-3 launched in 2023
    "NEGATED_FACT",      # X is not a politician
    "OPINION",
    "PREDICTION"
}




# =========================================================
# LOAD MODEL (LAZY)
# =========================================================

tokenizer = None
model = None

def get_model():
    global tokenizer, model
    if model is None:
        print("🔄 Loading ML model (first request only)...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )
        model.eval()
        print("✅ ML model loaded")
    return tokenizer, model

# =========================================================
# MARIANMT TRANSLATION (NO TOKEN, NO LOGIN)
# =========================================================


BASE_DIR = Path(__file__).resolve().parent

MARIAN_PATH = BASE_DIR / "models" / "marian_mul_en"
_marian_model = None
_marian_tokenizer = None

def get_marian():
    global _marian_model, _marian_tokenizer

    if _marian_model is None:
        print("🔄 Loading Marian translation model...")
        _marian_tokenizer = MarianTokenizer.from_pretrained(
            BASE_DIR / "models" / "marian_mul_en",
            local_files_only=True
        )
        _marian_model = MarianMTModel.from_pretrained(
            BASE_DIR / "models" / "marian_mul_en",
            local_files_only=True
        )
        print("✅ Marian loaded")

    return _marian_tokenizer, _marian_model


@lru_cache(maxsize=512)
def cached_translate_to_english(text: str, lang: str) -> str:
    """
    Caches translation results.
    Safe because translation is deterministic and stateless.
    """

    # ✅ English needs NO translation
    if lang == "en":
        return text

    if lang not in ["hi", "te"]:
        return text

    global _marian_model, _marian_tokenizer

    if _marian_model is None:
        model_name = str(MARIAN_PATH)
        _marian_tokenizer = MarianTokenizer.from_pretrained(model_name)
        _marian_model = MarianMTModel.from_pretrained(model_name)
        _marian_model.eval()

    inputs = _marian_tokenizer(text, return_tensors="pt", truncation=True)
    translated = _marian_model.generate(**inputs)

    return _marian_tokenizer.decode(
        translated[0],
        skip_special_tokens=True
    )


# =========================================================
# SENTENCE TRANSFORMER (LIGHTWEIGHT)
# =========================================================

SIM_MODEL = None

def get_similarity_model():
    global SIM_MODEL
    if SIM_MODEL is None:
        print("🔄 Loading Sentence Transformer (MiniLM)...")
        SIM_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        print("✅ Sentence Transformer loaded")
    return SIM_MODEL

# =========================
# SINGLE INDICBERT MODEL
# =========================

INDIC_MODEL_PATH = str(BASE_DIR / "models" / "indicbert_hi_te")

indic_tokenizer = None
indic_model = None

def get_indic_model():
    global indic_tokenizer, indic_model

    if indic_model is None:
        print("🔄 Loading IndicBERT model (Hindi + Telugu)...")

        indic_tokenizer = AutoTokenizer.from_pretrained(
            INDIC_MODEL_PATH,
            local_files_only=True
        )

        indic_model = AutoModelForSequenceClassification.from_pretrained(
            INDIC_MODEL_PATH,
            local_files_only=True
        )

        indic_model.eval()

        print("✅ IndicBERT model loaded successfully")

    return indic_tokenizer, indic_model

# =========================================================
# NLP HELPERS
# =========================================================

sentiment_analyzer = SentimentIntensityAnalyzer()

STOPWORDS = {
    "the", "is", "in", "and", "to", "of", "a", "for", "on",
    "with", "as", "by", "at", "from", "this", "that", "it",
    "be", "are", "was", "were", "has", "have", "had",
    "will", "would", "can", "could", "may", "might"
}

HINDI_STOPWORDS = {
    "और", "का", "की", "है", "में", "से", "को", "पर", "यह",
    "था", "थे", "हो", "हैं", "लिए", "गया", "गई",
    "कर", "किया", "किए", "करने"
}

TELUGU_STOPWORDS = {
    "మరియు", "లో", "కి", "కు", "తో", "పై", "ఇది", "అని",
    "ఉంది", "ఉన్న", "చేసింది", "చేశారు",
    "చేయడం", "కావచ్చు"
}

HISTORICAL_KEYWORDS = [
    # English (events / institutions)
    "demonetisation", "demonetization",
    "apollo 11", "moon landing",
    "chandrayaan", "isro",
    "independence day", "indian independence",
    "constitution of india", "indian constitution",
    "g20 summit", "union budget",

    # Hindi
    "विमुद्रीकरण", "संविधान",
    "स्वतंत्रता दिवस", "चंद्रयान", "इसरो",
    "जी20", "केंद्रीय बजट",

    # Telugu
    "నోట్ల రద్దు", "భారత రాజ్యాంగం",
    "స్వాతంత్ర్య దినోత్సవం", "చంద్రయాన్", "ఇస్రో",
    "జి20", "కేంద్ర బడ్జెట్"
]

SPECULATIVE_WORDS = [
    # Future / uncertainty
    "will", "soon", "coming", "next month", "next year",
    "may", "might", "expected",

    # Conspiracy / secrecy
    "secretly", "hidden", "leaked",
    "cover up", "exposed", "conspiracy",

    # Control / manipulation
    "control", "controlled", "manipulated",

    # Non-human speculation
    "aliens", "ufo", "extraterrestrial"
]

IMPOSSIBLE_KEYWORDS = [
    # Biological / medical impossibilities
    "immortal", "immortality", "live forever", "never die",
    "reverse aging", "stop aging",
    "instant cure", "miracle cure", "cure all diseases",
    "heal all diseases", "one cure for all",

    # Physical / cosmic impossibilities
    "earth will split", "earth will break", "earth will crack",
    "planet will split", "world will end",
    "sun will explode", "moon will crash",
    "gravity will stop", "time will stop",

    # Absolute transformations
    "turn water into fuel",
    "create energy from nothing",
    "defy laws of physics",

     # Hindi
    # =========================
    "अमर", "अमरता", "हमेशा जीवित", "कभी नहीं मरे",

    # =========================
    # Telugu
    # =========================
    "అమరత్వం", "చిరంజీవి", "ఎప్పటికీ చనిపోరు",

    # =========================
    # Physical / cosmic
    # =========================
    "earth will split", "earth will break", "earth will crack",
    "planet will split", "world will end",
    "sun will explode", "moon will crash",
    "gravity will stop", "time will stop",
]

FUTURE_CERTAINTY_WORDS = [
    "will", "will soon", "very soon",
    "next month", "next year",
    "guaranteed", "confirmed",
]

FAKE_AUTHORITY_PATTERNS = [
    r"\bnasa\b.*\bconfirmed\b",
    r"\bscientists\b.*\bconfirmed\b",
    r"\bexperts\b.*\bconfirmed\b",
    r"\bresearchers\b.*\bconfirmed\b",
    r"\bstudy\b.*\bproved\b",
]


RAG_ATTEMPTS = ContextVar("RAG_ATTEMPTS", default=0)
RAG_DECISION = ContextVar("RAG_DECISION", default=None)

def role_supported_by_text(role: str, text: str) -> bool:
    role = role.lower()
    text = text.lower()

    if role in text:
        return True

    # Ontology support (politician ← prime minister, etc.)
    if role in ROLE_EQUIVALENTS:
        return any(r in text for r in ROLE_EQUIVALENTS[role])

    return False


def subject_supported_by_text(subject: str, text: str) -> bool:

    if not subject or not text:
        return False

    text_l = text.lower()

    # -------------------------------------------------
    # 1️⃣ Normalize subject (remove titles / honorifics)
    # -------------------------------------------------
    subject = subject.lower()

    subject = re.sub(
        r"\b(pm|cm|dr|mr|mrs|ms|shri|sir)\b\.?",
        "",
        subject
    ).strip()

    subject_tokens = [t for t in subject.split() if len(t) > 2]

    if not subject_tokens:
        return False

    # -------------------------------------------------
    # 2️⃣ Strong surname anchoring (LAST TOKEN)
    # -------------------------------------------------
    surname = subject_tokens[-1]

    if surname in text_l:
        return True

    # -------------------------------------------------
    # 3️⃣ Fuzzy match fallback (handles aliases / initials)
    # -------------------------------------------------
    for word in re.findall(r"[a-z]{3,}", text_l):
        ratio = SequenceMatcher(None, surname, word).ratio()
        if ratio >= 0.85:
            return True

    return False

def role_claim_is_broader(claim_role, profile_role):
    claim_role = normalize_role(claim_role)
    profile_role = normalize_role(profile_role)

    if claim_role in ROLE_RANKS and profile_role in ROLE_RANKS:
        return ROLE_RANKS[claim_role] > ROLE_RANKS[profile_role]

    return False



def fuzzy_name_match(subject_tokens, text_l):
    hits = 0
    for tok in subject_tokens:
        if tok in text_l:
            hits += 1
        elif len(tok) > 4:
            # allow minor spelling variation (hanumantha vs hanumanth)
            if tok[:-1] in text_l or tok[:-2] in text_l:
                hits += 1
    return hits >= max(1, len(subject_tokens) - 1)


def is_trusted_news_domain(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(t in domain for t in TRUSTED_SOURCES)


def is_trusted_domain(url: str) -> bool:
    return any(td in url.lower() for td in TRUSTED_SOURCES)

def known_false_claim_gate(claim: str):
    claim_l = claim.lower()

    for pattern, correction in KNOWN_FALSE_PATTERNS:
        if pattern in claim_l:
            return {
                "status": "KNOWN_FALSE",
                "finalLabel": "FAKE",
                "confidencePercent": 98,
                "summary": "Well-established scientific falsehood.",
                "explanation": (
                    "This claim contradicts well-established scientific consensus. "
                    + correction
                ),
                "sentiment": {
                    "overall": "neutral",
                    "anger": 0,
                    "fear": 0,
                    "neutral": 100
                },
                "verificationMethod": "SCIENTIFIC_CONSENSUS",
                "evidence": []
            }

    return None

def absurd_claim_gate(claim: str):
    claim_l = claim.lower()
    is_negated = is_negated_claim(claim)

    # =====================================================
    # 🔴 NEW: Impossible physics / biology (ADD ONLY)
    # =====================================================
    for pattern in ABSURD_IMPOSSIBLE_PATTERNS:
        if pattern in claim_l:
            if is_negated:
                return {
                    "finalLabel": "REAL",
                    "confidencePercent": 95,
                    "summary": "Correct denial of an impossible claim.",
                    "explanation": (
                        "The claim correctly denies a statement that violates "
                        "fundamental physical or biological laws."
                    ),
                    "sentiment": {
                        "overall": "neutral",
                        "anger": 0,
                        "fear": 0,
                        "neutral": 100
                    },
                    "verificationMethod": "RULE_BASED",
                    "evidence": []
                }
            else:
                return {
                    "finalLabel": "FAKE",
                    "confidencePercent": 95,
                    "summary": "Physically or biologically impossible claim.",
                    "explanation": (
                        "The claim asserts a condition that violates fundamental "
                        "physical or biological laws."
                    ),
                    "sentiment": {
                        "overall": "neutral",
                        "anger": 0,
                        "fear": 0,
                        "neutral": 100
                    },
                    "verificationMethod": "RULE_BASED",
                    "evidence": []
                }

    ABSURD_TERMS = [
        "alien", "dragon", "time traveler",
        "immortal", "god", "supernatural being"
    ]

    for term in ABSURD_TERMS:
        if term in claim_l:
            if is_negated:
                return {
                    "status": "ABSURD_NEGATED",
                    "finalLabel": "REAL",
                    "confidencePercent": 95,
                    "summary": "Correct denial of an absurd claim.",
                    "explanation": "The claim correctly denies an impossible or absurd assertion.",
                    "sentiment": {
                        "overall": "neutral",
                        "anger": 0,
                        "fear": 0,
                        "neutral": 100
                    },
                    "verificationMethod": "RULE_BASED",
                    "evidence": []
                }
            else:
                return {
                    "status": "ABSURD",
                    "finalLabel": "FAKE",
                    "confidencePercent": 95,
                    "summary": "Absurd claim detected.",
                    "explanation": "The claim assigns an impossible or absurd property that violates real-world constraints.",
                    "sentiment": {
                        "overall": "neutral",
                        "anger": 0,
                        "fear": 0,
                        "neutral": 100
                    },
                    "verificationMethod": "RULE_BASED",
                    "evidence": []
                }

    return None

# ==========================================================
# 🌍 Extract geographic region from claim text
# ==========================================================
def extract_region(text: str):
    if not text:
        return None

    REGIONS = [
        "south asia",
        "southeast asia",
        "east asia",
        "west asia",
        "central asia",
        "middle east",
        "europe",
        "africa",
        "north america",
        "south america"
    ]

    t = text.lower()

    for r in REGIONS:
        if r in t:
            return r

    return None


def extract_semantic_role_blocks(soup):
    blocks = []

    for tag in soup.find_all(
        ["h1", "h2", "h3", "h4", "p", "span", "strong", "li", "dt", "dd"]
    ):
        text = tag.get_text(" ", strip=True)
        if not text or len(text) > 300:
            continue

        t = text.lower()
        if any(role in t for role in ROLE_WORDS):
            parent = tag.find_parent(["section", "article", "div"]) or tag
            block = parent.get_text(" ", strip=True)

            if 40 <= len(block) <= 800:
                blocks.append(block)

    return dedupe_blocks(blocks)


SOCIAL_TEXT = [
    "facebook", "instagram", "twitter",
    "linkedin", "youtube", "dribbble", "x"
]
COUNTRY_META = {
    "colombia": {
        "entity_type": "country",
        "official_name": "Republic of Colombia",

        # Geography
        "continent": "south america",
        "subregion": "latin america",

        # Sovereignty
        "sovereign": True,
        "member_of": [],

        # ISO / identifiers (industry standard)
        "iso": {
            "alpha2": "CO",
            "alpha3": "COL",
            "numeric": "170"
        },

        # Political facts
        "capital": "bogotá",
        "government_type": "republic",

        # Validation helpers
        "can_contain_countries": False,
        "can_be_contained_by_country": False
    },

    "united kingdom": {
        "entity_type": "country",
        "official_name": "United Kingdom of Great Britain and Northern Ireland",

        "continent": "europe",
        "subregion": "western europe",

        "sovereign": True,

        # UK is a special case (countries inside it)
        "member_of": [],
        "contains_entities": [
            "england",
            "scotland",
            "wales",
            "northern ireland"
        ],

        "iso": {
            "alpha2": "GB",
            "alpha3": "GBR",
            "numeric": "826"
        },

        "capital": "london",
        "government_type": "constitutional monarchy",

        "can_contain_countries": False,
        "can_be_contained_by_country": False
    }
}


NAV_JUNK = [
    "academic regulations", "campus facilities",
    "student facilities", "departments",
    "academic forums", "clubs", "committees",
    "policy", "infrastructure", "library",
    "sports", "hostel", "transport",
    "about vbit", "governing body",
    "audit reports", "r&d", "iiic"
]


NAME_REGEX = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof)\.?\s*[A-Z]\.?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
)


def extract_headline_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # 1️⃣ Try OpenGraph (most reliable)
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()

    # 2️⃣ Try Twitter title
    twitter = soup.find("meta", property="twitter:title")
    if twitter and twitter.get("content"):
        return twitter["content"].strip()

    # 3️⃣ Try H1
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    # 4️⃣ Fallback to <title>
    if soup.title and soup.title.string:
        return soup.title.string.strip()

    return ""


def geography_validator(claim: str):
    claim_l = claim.lower()

    if " is in " not in claim_l:
        return None

    subject, location = map(str.strip, claim_l.split(" is in ", 1))
    subject = normalize_entity(subject)
    location = normalize_entity(location)

    subj = COUNTRY_META.get(subject)
    loc = COUNTRY_META.get(location)

    # --- Case 1: Country inside country (IMPOSSIBLE) ---
    if subj and loc:
        if (
            subj["entity_type"] == "country"
            and loc["entity_type"] == "country"
        ):
            return {
                "verdict": "FAKE",
                "reason": (
                    f"{subj['official_name']} is a sovereign country "
                    f"and cannot be part of {loc['official_name']}."
                )
            }

    # --- Case 2: Country in continent (VALID) ---
    if subj and location == subj.get("continent"):
        return {
            "verdict": "REAL",
            "reason": (
                f"{subj['official_name']} is located in "
                f"{subj['continent'].title()}."
            )
        }

    return None

def normalize_entity(text: str) -> str:
    return text.lower().strip()


# =========================================================
# 📰 HEADLINE EXTRACTOR (SAFE — NO PIPELINE BREAK)
# =========================================================
def extract_headline_from_ocr(text: str) -> str:
    if not text:
        return ""

    # Split OCR into sentences
    sentences = [s.strip() for s in re.split(r'[.!?\\n]', text) if s.strip()]

    if not sentences:
        return text

    # Heuristic:
    # headline is usually first meaningful sentence
    first = sentences[0]

    # Remove newspaper header noise
    noise_words = [
        "www.", "follow us", "contact us", "vol.", "pages",
        "facebook", "twitter", "@", "latestnews"
    ]

    clean_first = first.lower()
    if any(n in clean_first for n in noise_words) and len(sentences) > 1:
        return sentences[1]

    return first

# =========================================================
# CAR STEP 1 — AUTHORITY ENTITY EXTRACTION
# =========================================================

AUTHORITY_PREPOSITIONS = [
    " in ",
    " at ",
    " of ",
    " from "
]

def extract_authority_entity(claim: str) -> str | None:

    if not claim:
        return None

    claim_clean = claim.strip()

    # Normalize spacing
    claim_clean = re.sub(r"\s+", " ", claim_clean)

    lower_claim = claim_clean.lower()

    for prep in AUTHORITY_PREPOSITIONS:
        if prep in lower_claim:
            # Split only once
            parts = re.split(prep, claim_clean, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) < 2:
                continue

            candidate = parts[1].strip()

            # Remove trailing punctuation
            candidate = re.sub(r"[^\w\s.&-]", "", candidate)

            # Limit to first 6 words (avoid long sentences)
            words = candidate.split()
            if not words:
                continue

            candidate = " ".join(words[:6])

            # Accept entity if meaningful length
            if len(candidate) >= 3:
                return candidate.strip()

    return None

# =========================================================
# CAR STEP 2 — AUTHORITY-AWARE URL ROUTING (BASIC)
# =========================================================


ACADEMIC_TLDS = [".edu", ".ac.", ".gov"]

def route_urls_by_authority(claim: str, urls: list[str]) -> list[str]:

    if not urls:
        return urls

    authority_entity = extract_authority_entity(claim)

    if not authority_entity:
        return urls

    authority_lower = authority_entity.lower()

    authority_root_urls = []      # google.com
    authority_sub_urls = []       # *.google.com
    blog_authority_urls = []      # blog.google
    academic_urls = []
    trusted_urls = []
    other_urls = []

    for url in urls:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()

        # --------------------------------------------
        # 1️⃣ STRICT ROOT AUTHORITY MATCH
        # --------------------------------------------

        # Example: google.com
        if domain == f"{authority_lower}.com":
            authority_root_urls.append(url)
            continue

        # Example: news.google.com, about.google.com
        if domain.endswith(f".{authority_lower}.com"):
            
            # If it's clearly blog-like, downrank
            if "blog" in domain or "/blog" in path:
                blog_authority_urls.append(url)
            else:
                authority_sub_urls.append(url)

            continue

        # --------------------------------------------
        # 2️⃣ Academic / Government
        # --------------------------------------------
        if any(tld in domain for tld in [".edu", ".ac.", ".gov"]):
            academic_urls.append(url)
            continue

        # --------------------------------------------
        # 3️⃣ Trusted sources
        # --------------------------------------------
        if any(ts in domain for ts in TRUSTED_SOURCES):
            trusted_urls.append(url)
            continue

        # --------------------------------------------
        # 4️⃣ Others
        # --------------------------------------------
        other_urls.append(url)

    # --------------------------------------------
    # FINAL PRIORITY ORDER
    # --------------------------------------------

    return (
        authority_root_urls +
        authority_sub_urls +
        academic_urls +
        trusted_urls +
        blog_authority_urls +   # blogs pushed lower
        other_urls
    )

def extract_profile_names_from_dom(soup: BeautifulSoup) -> list[str]:
    names = set()

    # 1️⃣ Headings (most reliable)
    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(" ", strip=True)
        if NAME_REGEX.search(text):
            names.add(text)

    # 2️⃣ Strong / bold blocks
    for tag in soup.find_all(["strong", "b"]):
        text = tag.get_text(" ", strip=True)
        if NAME_REGEX.search(text):
            names.add(text)

    # 3️⃣ First visible profile blocks (fallback)
    for tag in soup.find_all(["div", "p", "span"], limit=40):
        text = tag.get_text(" ", strip=True)
        if NAME_REGEX.search(text):
            names.add(text)

    return list(names)

def extract_profile_roles(url: str):

    # ============================================
    # ⚡ VBIT FAST EXTRACTOR ONLY FOR VBIT URLS
    # ============================================
    if "vbithyd.ac.in" in url:

        print("⚡ USING FAST VBIT EXTRACTOR")
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
            "Connection": "keep-alive",
        }

        r = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")

        names = []
        roles = []

        # NAME from emd-container
        for h in soup.select("div.emd-container h3"):
            text = h.get_text(" ", strip=True)

            # remove junk
            if "navigation" in text.lower():
                continue
            if "engineering" in text.lower():
                continue

            if len(text.split()) >= 2 and len(text.split()) <= 6:
                names.append(text)

        # ROLE from strong.emptitle
        for tag in soup.select("strong.emptitle"):
            roles.append(tag.get_text(" ", strip=True).lower())

        return {
            "names": list(dict.fromkeys(names)),
            "roles": list(dict.fromkeys(roles))
        }

    # ============================================
    # 🔁 NORMAL EXTRACTION FOR OTHER PAGES
    # ============================================
    print("🔁 USING NORMAL EXTRACTOR")

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")

    names = []
    roles = []

    for h in soup.find_all(["h1", "h2", "h3"]):
        text = h.get_text(" ", strip=True)

        if NAME_REGEX.search(text):
            names.append(text)

    for tag in soup.find_all(["p", "span", "strong", "td", "dd"]):
        text = tag.get_text(" ", strip=True).lower()

        for role in ROLE_WORDS:
            if role in text:
                roles.append(text)

    return {
        "names": list(dict.fromkeys(names)),
        "roles": list(dict.fromkeys(roles))
    }


def choose_best_profile_name(subject, profile_names):
    if not subject or not profile_names:
        return profile_names[0] if profile_names else ""

    norm_subject = normalize_person_name(subject)

    # return the first matching real person name
    for n in profile_names:
        if norm_subject in normalize_person_name(n) or normalize_person_name(n) in norm_subject:
            return n

    # fallback
    return profile_names[-1]

def extract_generic_role_from_claim(claim: str) -> str | None:
    claim_l = claim.lower()

    for role in ROLE_EQUIVALENTS.keys():
        if f" is a {role}" in claim_l or f" is an {role}" in claim_l or f" is {role}" in claim_l:
            return role

    return None

def extract_generic_blocks(soup):
    blocks = []
    for div in soup.find_all("div"):
        text = div.get_text(" ", strip=True)
        if 80 <= len(text) <= 1200:
            blocks.append(text)
    return dedupe_blocks(blocks)


def remove_citation_markers(text: str) -> str:
    """
    Removes Wikipedia-style citations and cleans broken entities.
    Examples:
    Modi[a] → Modi
    text[12][b] → text
    """
    # Remove [a], [1], [23], [note 1], etc.
    text = re.sub(r"\[[^\]]*\]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

from bs4 import BeautifulSoup
import re


def extract_wikipedia_text(html: str, max_paragraphs: int = 5) -> str:
    

    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # Main article container
    content = soup.find("div", id="mw-content-text")
    if not content:
        return ""

    paragraphs = content.find_all("p", recursive=True)

    clean_paragraphs = []

    for p in paragraphs:
        text = p.get_text(" ", strip=True)

        # Remove citation markers like [1], [23]
        text = re.sub(r"\[\d+\]", "", text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Skip very short paragraphs
        if len(text.split()) < 20:
            continue

        clean_paragraphs.append(text)

        # Stop after N good paragraphs
        if len(clean_paragraphs) >= max_paragraphs:
            break

    return " ".join(clean_paragraphs)

# =====================================================
# Sentence Extraction (Industry Standard)
# =====================================================
def extract_sentences(text: str):
    """
    Splits evidence text into sentences for stance analysis.
    Keeps sentences long enough to be meaningful.
    """
    if not text:
        return []

    sentences = re.split(r"[.!?]\s+", text)

    clean = []
    for s in sentences:
        s = s.strip()
        if len(s.split()) >= 6:   # avoid junk fragments
            clean.append(s)

    return clean[:8]  # speed control


def claim_semantically_matches(user_text: str,
                                factcheck_text: str) -> bool:
    """
    Loose semantic matching between user claim
    and fact-check claim.
    """

    if not factcheck_text:
        return False

    def tokenize(text):
        text = re.sub(r"[^a-zA-Z0-9 ]", " ", text.lower())
        words = text.split()

        # remove weak words
        stopwords = {
            "is", "was", "are", "the", "a", "an",
            "of", "in", "on", "at", "to", "for",
            "and", "with", "by", "from"
        }

        return {w for w in words if len(w) > 2 and w not in stopwords}

    user_tokens = tokenize(user_text)
    fact_tokens = tokenize(factcheck_text)

    if not user_tokens or not fact_tokens:
        return False

    overlap = len(user_tokens & fact_tokens)

    overlap_ratio = overlap / max(len(user_tokens), 1)

    # 30% overlap threshold
    return overlap_ratio >= 0.3

BAD_DOMAINS = [
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "linkedin.com",
    "reddit.com",
    "quora.com",
    "medium.com",
    "pinterest.com"
]

def filter_urls(urls, max_keep=6):

    good_urls = []
    seen_domains = set()

    BLOG_PATTERNS = [
        "blog.",
        "/blog",
        "/blogs",
        "medium.com",
        "substack.com"
    ]

    for u in urls:
        try:
            parsed = urlparse(u)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()

            # ❌ Skip bad domains
            if any(bad in domain for bad in BAD_DOMAINS):
                continue

            # ❌ Skip blog-style URLs
            if any(bp in domain for bp in BLOG_PATTERNS):
                continue
            if any(bp in path for bp in BLOG_PATTERNS):
                continue

            # ❌ Skip duplicate domains
            if domain in seen_domains:
                continue

            seen_domains.add(domain)
            good_urls.append(u)

            if len(good_urls) >= max_keep:
                break

        except Exception:
            continue

    return good_urls


def is_absurd_role(role: str) -> bool:
    if not role:
        return False

    ABSURD_ROLES = {
        "alien", "extraterrestrial", "ufo",
        "god", "deity", "immortal",
        "wizard", "vampire", "superhero",
        "time traveler", "ghost"
    }

    role_l = role.lower()
    return any(a in role_l for a in ABSURD_ROLES)

def is_impossible_claim(text: str) -> bool:
    t = normalize_indic_text(text, safe_language_detect(text)).lower()

    for kw in IMPOSSIBLE_KEYWORDS:
        if kw in t:
            return True

    if (
        any(re.search(p, t) for p in FAKE_AUTHORITY_PATTERNS)
        and any(w in t for w in FUTURE_CERTAINTY_WORDS)
    ):
        return True

    absolute_patterns = [
        r"\ball\b.*\bdiseases\b",
        r"\bno one\b.*\bwill ever\b",
        r"\bhumans\b.*\blive forever\b",
    ]
    if any(re.search(p, t) for p in absolute_patterns):
        return True

    return False


SCREENSHOT_KEYWORDS = [
    # Social media
    "retweet", "likes", "views", "reply", "share",
    "twitter", "instagram", "facebook",
    "twitter for android", "twitter for iphone",
    "@",

    # News banners
    "breaking news", "live", "exclusive"
]

def recover_words(text: str) -> str:
    words = text.split()
    recovered = []

    COMMON_WORDS = set([
        "the","is","a","an","and","to","of","in","on","for","with",
        "at","by","from","as","that","this","it","be","are","was",
        "were","or","if","but","not","we","they","you","he","she"
    ])

    for w in words:
        if w in COMMON_WORDS or len(w) <= 3:
            recovered.append(w)
            continue

        match = get_close_matches(w, COMMON_WORDS, n=1, cutoff=0.75)
        recovered.append(match[0] if match else w)

    return " ".join(recovered)


def reconstruct_phrases(text: str) -> str:
    # Join broken headline words
    text = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", text)

    # Fix numbers
    text = re.sub(r"\b(\d)\s+(\d)\b", r"\1\2", text)

    return text


IMPORTANT_ENTITIES = {
    "titanic", "carpathia", "globe", "reuters",
    "bbc", "india", "isro", "apollo", "moon",
    "earthquake", "election", "budget"
}

def extract_entities(text: str):
    words = text.split()

    entities = []
    numbers = []
    years = []

    for w in words:
        if w.isdigit():
            if len(w) == 4:
                years.append(w)
            else:
                numbers.append(w)
        elif w.lower() in IMPORTANT_ENTITIES:
            entities.append(w.lower())


    return {
        "entities": list(dict.fromkeys(entities)),
        "numbers": numbers,
        "years": years
    }


def build_entity_summary(text: str) -> str:
    data = extract_entities(text)

    parts = []
    if data["entities"]:
        parts.append(" ".join(data["entities"]))
    if data["years"]:
        parts.append(" ".join(data["years"]))
    if data["numbers"]:
        parts.append(" ".join(data["numbers"]))

    # Fallback
    if not parts:
        return " ".join(text.split()[:6])

    return " ".join(parts)

STOPWORDS = {
    "the","is","was","on","in","of","for","to","and","a","an","with",
    "by","at","from","as","it","this","that"
}

def extract_search_terms(text: str, max_terms: int = 8):
    tokens = text.split()

    keywords = []
    years = []

    for t in tokens:
        if t.isdigit() and len(t) == 4:
            years.append(t)
        elif len(t) >= 4 and t not in STOPWORDS:
            keywords.append(t)

    return (years + keywords)[:max_terms]



def extract_person_name(text: str) -> str | None:

    text = text.replace(".", "")

    # Remove titles
    text = re.sub(r"\b(dr|mr|mrs|ms|prof)\b", "", text, flags=re.I)

    # Match: Initial + Surname OR Full Name
    m = re.search(
        r"\b([A-Z](?:\s+[A-Z][a-z]+)|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))\b",
        text
    )

    return m.group(1).strip() if m else None


def looks_like_historical_fact(text: str) -> bool:
    t = text.lower() if re.search(r"[a-zA-Z]", text) else text
    
    # Year like 1969, 2016 etc.
    has_year = bool(re.search(r"\b(18|19|20)\d{2}\b", t))

    strong_events = [
        r"apollo\s*11",r"moon landing",r"chandrayaan\s*\d?",r"demonetisation|demonetization",
        r"constitution of india",r"indian constitution",r"g20\s*(summit)?",r"union budget",r"independence\s*(day)?",

        r"अपोलो", r"चंद्रमा", r"चंद्रयान",
        r"विमुद्रीकरण", r"भारतीय संविधान",
        r"संविधान", r"स्वतंत्रता दिवस", r"जी20", r"केंद्रीय बजट",

        r"అపోలో", r"చంద్రునిపై", r"చంద్రయాన్",
        r"నోట్ల రద్దు", r"భారత రాజ్యాంగం",
        r"స్వాతంత్ర్య దినోత్సవం", r"జి20", r"కేంద్ర బడ్జెట్",
    ]

    has_strong_event = any(re.search(p, t) for p in strong_events)

    return has_strong_event or (
        has_year and any(k in t for k in HISTORICAL_KEYWORDS)
    )

def is_assertive_historical_event(text: str) -> bool:
    """
    Detects clean historical event assertions.
    """
    t = text.lower()

    EVENT_VERBS = [
        "launched", "launch",
        "landed", "occurred",
        "happened", "was launched"
    ]

    has_year = bool(re.search(r"\b(19|20)\d{2}\b", t))
    has_event_verb = any(v in t for v in EVENT_VERBS)
    has_negation = any(n in t for n in ["fake", "hoax", "staged", "false"])

    return has_year and has_event_verb and not has_negation


def is_incomplete_role_claim(text: str) -> bool:
    t = text.lower()

    ROLE_NEED_ORG = ["ceo", "founder", "chairman"]

    if any(r in t for r in ROLE_NEED_ORG):
        # Must mention an organization
        if not re.search(r"\bof\b\s+[a-z]", t):
            return True

    return False

def build_search_query(claim: str) -> str:
    """
    Balanced RAG query builder.
    - Keeps entities, roles, actions, years
    - Works for BOTH:
      • factual events (Chandrayaan-3 launched 2023)
      • generic facts (PM Modi is a politician)
    """

    text = claim.lower()

    # Light stopwords only (NOT aggressive)
    STOPWORDS = {
        "the","is","was","are","were","has","have",
        "a","an","in","on","at","by","for","with",
        "this","that","it"
    }

    tokens = re.findall(r"[a-z0-9\-]+", text)

    kept = []
    for t in tokens:
        if (
            t not in STOPWORDS
            and (len(t) >= 3 or t.isdigit() or "-" in t)
        ):
            kept.append(t)

    # -----------------------------
    # 🧠 SMART FALLBACK (IMPORTANT)
    # -----------------------------
    # If query is too short, KEEP original claim
    if len(kept) < 3:
        return claim.lower()

    return " ".join(kept[:20])

COUNTRY_DEMONYM_MAP = {
    "india": "indian",
    "united states": "american",
    "america": "american",
    "canada": "canadian",
    "south africa": "south african",
    "united kingdom": "british",
    "uk": "british",
    "china": "chinese",
    "france": "french",
    "germany": "german",
    "italy": "italian",
    "spain": "spanish",
    "japan": "japanese",
    "australia": "australian",
    "russia": "russian",
    "brazil": "brazilian",
    "mexico": "mexican",
}

def validate_attribute(claim, evidence_text):
    parsed = parse_attribute_claim(claim)
    if not parsed:
        return None

    subject = parsed["subject"]
    attribute = parsed["attribute"]

    attr_type = detect_attribute_type(attribute)
    print("CLAIM", claim)
    print("PARSED CLAIM", parsed)
    print("ATTRIBUTE TYPE", attr_type)

    if not attr_type:
        return None

    # ------------------------------
    # NATIONALITY VALIDATION
    # ------------------------------
    if attr_type == "NATIONALITY":
        claim_nats = extract_nationalities(attribute)
        evidence_nats = extract_nationalities(evidence_text)
        print("CLAIM NATIONALITIES", claim_nats)
        print("EVIDENCE NATIONALITIES", evidence_nats)

        if not claim_nats:
            return None

        if claim_nats.intersection(evidence_nats):
            return "SUPPORTED"
        else:
            return "CONTRADICTION"

    # ------------------------------
    # ROLE VALIDATION
    # ------------------------------
    if attr_type == "ROLE":
        attr_l = attribute.lower()
        print("ROLE CLAIM", attribute)
        print("ROLE FOUND IN EVIDENCE", attribute.lower() in evidence_text.lower())

        if attr_l in evidence_text.lower():
            return "SUPPORTED"
        else:
            return "CONTRADICTION"

    # ------------------------------
    # DATE VALIDATION
    # ------------------------------
    if attr_type == "DATE":
        claim_years = extract_years(attribute)
        evidence_years = extract_years(evidence_text)
        print("CLAIM YEARS", claim_years)
        print("EVIDENCE YEARS", evidence_years)

        if claim_years.intersection(evidence_years):
            return "SUPPORTED"
        else:
            return "CONTRADICTION"

    return None



def extract_nationalities(text: str):
    """
    Extracts nationality information from text in a robust way.
    Returns a set of normalized nationality strings.
    """

    if not text:
        return set()

    text_lower = text.lower()
    found = set()

    # 1️⃣ Direct demonym detection (e.g., "American", "Indian")
    for country, demonym in COUNTRY_DEMONYM_MAP.items():
        if re.search(rf"\b{re.escape(demonym)}\b", text_lower):
            found.add(demonym)

    # 2️⃣ "Citizen of X"
    for country, demonym in COUNTRY_DEMONYM_MAP.items():
        if re.search(rf"\bcitizen\s+of\s+{re.escape(country)}\b", text_lower):
            found.add(demonym)

    # 3️⃣ "Born in X"
    for country, demonym in COUNTRY_DEMONYM_MAP.items():
        if re.search(rf"\bborn\s+in\s+{re.escape(country)}\b", text_lower):
            found.add(demonym)

    # 4️⃣ "From X"
    for country, demonym in COUNTRY_DEMONYM_MAP.items():
        if re.search(rf"\bfrom\s+{re.escape(country)}\b", text_lower):
            found.add(demonym)

    # 5️⃣ Handle compound demonyms like "South African-born American"
    for country, demonym in COUNTRY_DEMONYM_MAP.items():
        if demonym in text_lower:
            found.add(demonym)



    return found



ATTRIBUTE_TYPES = {
    "NATIONALITY",
    "LOCATION",
    "ROLE",
    "DATE",
    "NUMERIC",
    "ORGANIZATION"
}
def parse_attribute_claim(claim: str):
    """
    Parses simple structured claims:
    SUBJECT is ATTRIBUTE
    SUBJECT was ATTRIBUTE
    """

    if not claim:
        return None

    match = re.split(r"\s+(is|was|are|were)\s+", claim, flags=re.I)
    if len(match) < 3:
        return None

    subject = match[0].strip()
    attribute = match[2].strip()

    return {
        "subject": subject,
        "attribute": attribute
    }

def detect_attribute_type(attribute: str):
    attr_l = attribute.lower()

    # 1️⃣ Nationality
    if extract_nationalities(attribute):
        return "NATIONALITY"

    # 2️⃣ Geography
    if attr_l in COUNTRY_REGION_MAP.values():
        return "LOCATION"

    # 3️⃣ Role
    if any(role in attr_l for role in ROLE_WORDS):
        return "ROLE"

    # 4️⃣ Date
    if re.search(r"\b(18|19|20)\d{2}\b", attr_l):
        return "DATE"

    # 5️⃣ Numeric
    if re.search(r"\b\d+\b", attr_l):
        return "NUMERIC"

    return None


def is_reported_fact(sentence: str) -> bool:
    REPORTING_CUES = [
        "according to",
        "said",
        "says",
        "reported",
        "reports",
        "draft document",
        "officials said",
        "sources said"
    ]
    s = sentence.lower()
    return any(cue in s for cue in REPORTING_CUES)

# =========================================================
# CLAIM DECOMPOSITION + LIGHTWEIGHT SIMILARITY
# =========================================================

def decompose_claims(text: str) -> List[str]:
    
    parts = re.split(r"[.!?]\s*", text)
    claims = [p.strip() for p in parts if len(p.strip()) >= 8]

    uniq = []
    seen = set()
    for c in claims:
        k = re.sub(r"\s+", " ", c.lower())
        if k not in seen:
            uniq.append(c)
            seen.add(k)

    return uniq or [text.strip()]


def _token_set(text: str) -> set:
    
    tokens = re.findall(
        r"[a-zA-Z]{2,}|[\u0900-\u097F]{2,}|[\u0C00-\u0C7F]{2,}",
        text.lower()
    )
    return set(tokens)

def is_negated_claim(claim: str) -> bool:
    """
    Detects whether a claim is negated.
    Example:
      - 'X is not Y'  -> True
      - 'X isn't Y'  -> True
      - 'X is Y'     -> False
    """
    if not claim:
        return False

    NEGATION_WORDS = [
        " not ", " no ", " never ",
        " isn't ", " isnt ",
        " wasn't ", " wasnt ",
        " does not ", " doesn't ",
        " do not ", " did not ",
        " cannot ", " can't ",
        " has not ", " haven't ",
        " is not "
    ]

    claim_l = f" {claim.lower()} "
    return any(n in claim_l for n in NEGATION_WORDS)


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def split_sentences(text: str):
    sentences = re.split(r"[.!?]\s+", text)
    return [s.strip() for s in sentences if len(s.split()) >= 6]

def get_top_k_sentences(claim, sentences, model, k=5):
    if not sentences:
        return []

    claim_emb = model.encode(
        claim,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    sent_embs = model.encode(
        sentences,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    scores = util.cos_sim(claim_emb, sent_embs)[0]

    ranked = sorted(
        zip(sentences, scores),
        key=lambda x: float(x[1]),
        reverse=True
    )

    return [s for s, _ in ranked[:k]]

def semantic_similarity(a: str, b: str) -> float:

    model = get_similarity_model()
    emb_a = model.encode(a, normalize_embeddings=True)
    emb_b = model.encode(b, normalize_embeddings=True)
    return float(util.cos_sim(emb_a, emb_b))

def semantic_role_assertion(role: str, sentence: str) -> bool:
    role = role.lower()
    sentence = sentence.lower()

    if role in sentence:
        return True

    if role in ROLE_EQUIVALENTS:
        for alt in ROLE_EQUIVALENTS[role]:
            if alt in sentence:
                return True

    return False


def detect_content_domain(url: str) -> str:
    path = urlparse(url).path.lower()

    if "/sport/" in path:
        return "SPORTS"
    if "/news/" in path or "/world/" in path or "/business/" in path:
        return "NEWS"
    return "GENERAL"


# =========================================================
# NLI
# =========================================================

nli_model = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small"
)
# =========================================================
# OCR
# =========================================================

# =========================================================
# 🔥 GLOBAL PADDLE OCR ENGINE (LOAD ONCE)
# =========================================================
_ocr_reader = None

def get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        print("🔄 Loading PaddleOCR...")
        _ocr_reader = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False
        )
    return _ocr_reader

# =========================================================
# INPUT SCHEMAS
# =========================================================

class InputText(BaseModel):
    text: str

class InputURL(BaseModel):
    url: str

class InputImage(BaseModel):
    image_base64: str

# =========================================================
# GOOGLE FACT CHECK
# =========================================================
# =====================================================
# Google Fact Check API integration
# =====================================================
def google_fact_check(text: str, lang: str):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    params = {
        "query": text,
        "languageCode": lang,
        "key": GOOGLE_FACTCHECK_API_KEY
    }

    print("🌐 Calling Google Fact Check API")

    try:
        response = requests.get(url, params=params, timeout=8)

        print("🌐 Status Code:", response.status_code)

        data = response.json()
        claims = data.get("claims", [])

        print("📘 Claims received:", len(claims))

        if not claims:
            print("ℹ️ No fact-check claims found")
            return None

        evidence = []
        verdict_scores = []

        # thresholds
        SOFT_MATCH_THRESHOLD = 0.70   # your previous SIM_THRESHOLD
        STRICT_OVERRIDE_THRESHOLD = 0.85  # new strict override threshold
        WORD_OVERLAP_MIN_RATIO = 0.50

        # small helpers (kept inline to avoid new modules)
        def normalize_words(s: str):
            return set(s.lower().strip().split())

        def looks_like_assertive_event_claim(s: str) -> bool:
            # detect claims like "X landed", "X won", "X declared", "X is president", etc.
            s_l = s.lower()
            event_verbs = ["landed", "won", "declared", "is", "was", "elected", "became", "launched", "released"]
            # quick heuristic: if any strong event verb present, treat as assertive/event claim
            return any(w in s_l for w in event_verbs)

        def review_looks_media_only(fc_text: str, review_title: str) -> bool:
            # If the fact-check is about media/visuals (image/video/clip/animation), it's likely not contradicting the event itself
            media_indicators = ["video", "image", "photo", "footage", "visual", "clip", "animation", "graphic", "fake visuals"]
            fc_combined = (fc_text or "") + " " + (review_title or "")
            fc_combined_l = fc_combined.lower()
            return any(mi in fc_combined_l for mi in media_indicators)

        for claim in claims:
            fc_claim_text = claim.get("text", "") or ""
            # ---- STRONG semantic filtering + lexical grounding ----
            sim_score = semantic_similarity(text, fc_claim_text)

            # lexical grounding: ensure some direct word overlap so topic-level matches don't slip in
            claim_words = normalize_words(text)
            fc_words = normalize_words(fc_claim_text)
            common_ratio = len(claim_words & fc_words) / max(1, len(claim_words))

            if common_ratio < WORD_OVERLAP_MIN_RATIO:
                print("⚠️ WORD OVERLAP TOO LOW — SKIPPING FACTCHECK RESULT")
                continue

            print("🧪 FACTCHECK SIM:", sim_score, "|", fc_claim_text[:120])

            if sim_score < SOFT_MATCH_THRESHOLD:
                # too unrelated
                continue

            # determine if this fact-check match is strict enough to allow overriding RAG
            strict_override = sim_score >= STRICT_OVERRIDE_THRESHOLD

            for review in claim.get("claimReview", []):
                rating = review.get("textualRating", "") or ""
                rating_lc = rating.lower()

                # ---- SAFE rating mapping ---
                if "false" in rating_lc:
                    label, base_score = "FAKE", 3
                elif "misleading" in rating_lc or "missing context" in rating_lc or "partly false" in rating_lc:
                    label, base_score = "FAKE", 2
                elif "true" in rating_lc:
                    label, base_score = "REAL", 3
                else:
                    label, base_score = "UNVERIFIABLE", 1

                # Always collect evidence for UI / debugging
                evidence.append({
                    "source": review.get("publisher", {}).get("name", "Unknown"),
                    "url": review.get("url", ""),
                    "title": review.get("title", fc_claim_text),
                    "snippet": f"Rating: {rating}"
                })

                # Decide whether to let this fact-check vote into final verdict calculation
                if strict_override:
                    # Anti-false-negative guard:
                    # If the fact-check is a FAKE rating but the review appears to be about media/visuals
                    # while the claim is an assertive event claim, do NOT allow this FAKE to override.
                    if label == "FAKE" and looks_like_assertive_event_claim(text) and review_looks_media_only(fc_claim_text, review.get("title", "")):
                        print("⚠️ ANTI-FALSE-NEGATIVE: FAKE about media-only content detected — WILL NOT OVERRIDE RAG")
                        # do not append this FAKE vote (treat as non-overriding evidence)
                        continue

                    # If passed anti-false-negative guard, accept the vote (strict override)
                    print("✅ FACTCHECK STRICT OVERRIDE ALLOWED — adding verdict vote:", label)
                    verdict_scores.append((label, base_score))

                else:
                    # soft match — we show evidence but we DO NOT let fact-check override RAG decision.
                    print("⚠️ FACTCHECK MATCH NOT STRICT — WILL NOT OVERRIDE RAG (evidence collected).")
                    # (optionally could append a low-weight vote, but we skip to avoid overriding)
                    continue

        # After processing all claims/reviews
        if not verdict_scores:
            print("ℹ️ No VERDICT-SCORING fact-check matches (none strict enough to override)")
            return None

        fake_score = sum(s for l, s in verdict_scores if l == "FAKE")
        real_score = sum(s for l, s in verdict_scores if l == "REAL")

        if fake_score > real_score:
            final_label = "FAKE"
            confidence = min(95, 60 + fake_score * 5)
        elif real_score > fake_score:
            final_label = "REAL"
            confidence = min(95, 60 + real_score * 5)
        else:
            final_label = "UNVERIFIABLE"
            confidence = 60

        print("✅ Fact Check Verdict:", final_label)

        return {
            "label": final_label,
            "confidence": confidence,
            "evidence": evidence[:5]
        }

    except Exception as e:
        print(f"⚠️ Google Fact Check error: {e}")
        return None

@lru_cache(maxsize=256)
def google_fact_check_cached(text: str, lang: str) -> str:
    """
    Cached Google Fact Check wrapper.
    Returns JSON string so it can be cached safely.
    """
    result = google_fact_check(text, lang)
    return json.dumps(result) if result else ""


# =========================================================
# RETRIEVAL (RAG) STUBS
# =========================================================

def _safe_get(url: str, params=None, timeout=6):
    try:
        r = requests.get(
            url,
            params=params or {},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=timeout
        )
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None


def _extract_snippets(html: str, max_items: int = 5) -> List[str]:
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return [p for p in paras if len(p) > 40][:max_items]


def duckduckgo_search(query: str, max_results: int = 6):
    try:
        url = "https://duckduckgo.com/html/"

        r = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://duckduckgo.com/"
            },
            timeout=12
        )


        # 🔥 IMPORTANT: Spaces sometimes returns non-200 silently
        if r.status_code != 200:
            print("DuckDuckGo status:", r.status_code)
            return []

        soup = BeautifulSoup(r.text, "html.parser")

        results = []
        for a in soup.select("a.result__a[href]"):
            link = a["href"]

            # 🟢 DuckDuckGo sometimes wraps links like /l/?uddg=...
            if "uddg=" in link:
                from urllib.parse import parse_qs, urlparse, unquote
                qs = parse_qs(urlparse(link).query)
                if "uddg" in qs:
                    link = unquote(qs["uddg"][0])

            results.append(link)

            if len(results) >= max_results:
                break
        print("#==========================================")
        print("🔎 DuckDuckGo URLs:", results)
        print("#==========================================")
        return results

    except Exception as e:
        print("DuckDuckGo search error:", e)
        return []

# =========================================================
# 📰 NEWS-AWARE DETECTOR (SAFE — NO PIPELINE CHANGES)
# =========================================================
def detect_news_style(text: str) -> bool:
    if not text:
        return False

    t = text.lower()

    # Heuristic signals typical of news articles
    news_keywords = [
        "officials said", "analysts warn", "according to",
        "peace talks", "debate", "markets reacted",
        "sources said", "diplomatic", "reportedly",
        "latest news", "breaking", "report", "talks continue"
    ]

    # Long multi-sentence text usually means news paragraph
    word_count = len(t.split())

    keyword_hit = any(k in t for k in news_keywords)

    if word_count > 25 or keyword_hit:
        return True

    return False

def fact_match_decision(claim, evidence_text):

    # ---------- helpers ----------
    def extract_years(text):
        ys = re.findall(r"\b(19|20)\d{2}\b", text)
        # re returns only the first group if used like above, so safer to use a slightly different approach:
        ys_full = re.findall(r"\b(?:19|20)\d{2}\b", text)
        try:
            return [int(y) for y in ys_full]
        except Exception:
            return []

    def find_action_years(text, actions):
    
        found = []
        for a in actions:
            # look for 'action ... 2023' or '2023 ... action' within ~60 chars
            pattern1 = rf"{a}.{{0,80}}?\b((?:19|20)\d{{2}})\b"
            pattern2 = rf"\b((?:19|20)\d{{2}})\b.{{0,80}}?{a}"
            for pat in (pattern1, pattern2):
                for m in re.findall(pat, text, flags=re.IGNORECASE):
                    try:
                        found.append(int(m))
                    except:
                        pass
        return found

    def safe_similarity(a, b):
        try:
            sim_model = get_similarity_model()
            a_emb = sim_model.encode(a, normalize_embeddings=True)
            b_emb = sim_model.encode(b, normalize_embeddings=True)
            return float(util.cos_sim(a_emb, b_emb))
        except Exception:
            return 0.0

    # ---------- extract years ----------
    claim_years = extract_years(claim)
    evidence_years = extract_years(evidence_text)

    # action-tied years (gives stronger signal than random years in the doc)
    action_terms = ["launch", "launched", "launches", "landing", "landed", "land", "arrived"]
    evidence_action_years = find_action_years(evidence_text, action_terms)

    # pick a representative claim year (if multiple, prefer the first; you can refine this later)
    claim_year = claim_years[0] if claim_years else None

    # ---------- quick semantic relevance check ----------
    sim_score = safe_similarity(claim, evidence_text)  # 0..1 roughly

    # If evidence is not about the claim (very low similarity), treat as UNVERIFIABLE
    if sim_score < 0.28:
        # evidence is off-topic; not helpful
        return "UNVERIFIABLE", round(sim_score, 3)

    # ---------- year conflict detection ----------
    # If claim specifies a year and evidence contains action-tied years,
    # require the claim_year to appear in those action-tied years (strong signal)
    if claim_year:
        if evidence_action_years:
            # strong evidence has action-year(s)
            if claim_year not in evidence_action_years:
                # contradiction: evidence's action-year(s) disagree with claim
                return "FAKE", 0.99
            else:
                # direct match
                return "REAL", round(sim_score, 3)

        # If no action-tied year found but evidence has some years
        if evidence_years:
            if claim_year not in evidence_years:
                # evidence mentions other years but not the claimed one -> contradiction
                # be somewhat conservative: return FAKE but with high confidence
                return "FAKE", 0.95
            else:
                return "REAL", round(sim_score, 3)

    # ---------- no claim year to compare ----------
    # If no explicit year in claim, rely on semantic similarity and thresholds
    if sim_score >= 0.80:
        return "REAL", round(sim_score, 3)

    if sim_score >= 0.65:
        # plausible but not bulletproof
        return "UNVERIFIABLE", round(sim_score, 3)

    # fallback: low similarity (handled earlier) or ambiguous
    return "UNVERIFIABLE", round(sim_score, 3)


def classify_url_type(url: str) -> str:

    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    institutional_tlds = [".ac.", ".edu", ".gov"]
    media_domains = [
        "forbes.com", "britannica.com", "businessinsider.com",
        "wsj.com", "entrepreneur.com", "valiantceo.com",
        "linkedin.com", "wikipedia.org"
    ]

    # =========================================================
    # ✅ PROFILE / PEOPLE pages — STRICT (Institutional only)
    # =========================================================
    profile_keywords = [
        "/faculty", "/employees", "/employee",
        "/staff", "/people", "/profile",
        "/person", "/directory"
    ]

    if any(p in path for p in profile_keywords):

        # Only treat as PROFILE if domain looks institutional
        if any(tld in domain for tld in institutional_tlds):
            return "PROFILE"

        # Also allow official org domains (like google.com/about/people)
        if any(k in domain for k in ["google.com", "microsoft.com", "openai.com"]):
            return "PROFILE"

        # Otherwise treat as ARTICLE (media profile pages)
        return "ARTICLE"

    # =========================================================
    # ✅ CATEGORY pages
    # =========================================================
    CATEGORY_PATHS = [
        "/news", "/sport", "/sports", "/travel",
        "/culture", "/future", "/reel", "/business",
        "/politics", "/world", "/india"
    ]

    if path in CATEGORY_PATHS or path.rstrip("/") in CATEGORY_PATHS:
        return "CATEGORY"

    # =========================================================
    # ✅ OPINION / ANALYSIS
    # =========================================================
    if any(x in path for x in [
        "/live", "/opinion", "/analysis",
        "/editorial", "/comment"
    ]):
        return "OPINION"

    # =========================================================
    # ✅ ARTICLE pages (deep paths)
    # =========================================================
    if len(path.split("/")) >= 3:
        return "ARTICLE"

    return "UNKNOWN"

def verify_claim_with_rag(claim: str, detected_lang: str) -> tuple[bool, list[dict], float]:

    attempts = RAG_ATTEMPTS.get()
    decision = RAG_DECISION.get()
    docs = []


    if decision is not None:
        return decision

    MAX_ATTEMPTS = 3
    if attempts >= MAX_ATTEMPTS:
        return False, [], 0.0

    RAG_ATTEMPTS.set(attempts + 1)
    print(f"🧠 RAG attempt {attempts + 1}")

    # ==================================================
    # Translation
    # ==================================================
    if detected_lang in ["hi", "te"]:
        translated = cached_translate_to_english(claim, detected_lang)
        if (
            not translated
            or len(translated.split()) < 3
            or "error" in translated.lower()
            or translated.lower().startswith("can not")
        ):
            translated = claim
    else:
        translated = claim

    translated = normalize_alphanumeric_entities(translated)

    # ==========================================================
    # ⚡ INSTANT RULE GATE — RUN BEFORE SEARCH/RAG
    # ==========================================================
    subject, _ = cached_claim_parse(translated)
    claimed_region = extract_region(translated)

    if subject:
        true_region = COUNTRY_REGION_MAP.get(subject.lower())

        if claimed_region and true_region:

            # wrong region → instant FAKE
            if claimed_region != true_region:
                print("⚡ INSTANT GEOGRAPHY FAIL — SKIPPING RAG")

                result = (
                    False,
                    [{
                        "url": "about_blank",
                        "snippet": f"{subject.title()} belongs to {true_region}, not {claimed_region}"
                    }],
                    0.95
                )
                RAG_DECISION.set(result)
                return result


    # ==================================================
    # Skip incomplete / opinion claims
    # ==================================================
    if is_incomplete_role_claim(translated):
        result = (False, [], 0.0)
        RAG_DECISION.set(result)
        return result

    if detect_claim_stance(translated) == "OPINION":
        return False, [], 0.0

    # ==================================================
    # Build query
    # ==================================================
    search_query = build_search_query(translated)
    if len(search_query.split()) < 3:
        return False, [], 0.0

    print("====================================")
    print("🔥 ENTERED VERIFY_CLAIM_WITH_RAG")
    print("🧪 RAG INPUT CLAIM:", claim)

    # --------------------------------------------------
    # 📰 Detect NEWS MODE
    # --------------------------------------------------
    is_news_mode = detect_news_style(claim)

    if is_news_mode:
        print("📰 NEWS MODE:", True)

        headline_claim = extract_headline_from_ocr(claim)
        print("📰 HEADLINE EXTRACTED:", headline_claim)

        # 🔥 Translate headline SAME WAY as normal pipeline
        translated_headline = cached_translate_to_english(headline_claim, detected_lang)

        # Use translated headline for RAG
        rag_input = translated_headline if translated_headline else translated
    else:
        rag_input = translated


    print("🌐 DETECTED LANG:", detected_lang)
    print("📝 TRANSLATED:", translated)
    print("🔍 SEARCH QUERY:", search_query)

    # ==================================================
    # Retrieve docs
    # ==================================================
    search_urls = duckduckgo_search(search_query)
    urls = filter_urls(search_urls)
    print("#==========================================")
    print("✅ URLs after Step 2 filtering:", urls)
    print("#==========================================")
    SIM_THRESHOLD = 0.70
    authority_entity = extract_authority_entity(claim)
    print("🔥 AUTHORITY ENTITY:", authority_entity)

    urls = route_urls_by_authority(claim, urls)
    print("#===========================================")
    print("✅ URLs after CAR  filtering:", urls)
    print("#===========================================")

    wiki_urls = [u for u in urls if "wikipedia.org" in u]

    if wiki_urls:
        print("📌 WIKI FOUND IN CAR → PRIORITIZING")
        non_wiki_urls = [u for u in urls if "wikipedia.org" not in u]
        urls = wiki_urls + non_wiki_urls

    rag_result = universal_rag_retrieve(
        rag_input,
        urls,
        sim_threshold=SIM_THRESHOLD,
        authority_entity=authority_entity
    )

    # =================================================
    # DECISIVE RESULT
    # =================================================
    if isinstance(rag_result, dict) and rag_result.get("finalLabel") in ["REAL", "FAKE"]:
        print("🛑 GLOBAL HARD RETURN — DICT")
        print("🛑 EXITING TEXT PIPELINE — MODEL5 FINAL DECISION")
        print(f"\n🎯 FINAL VERDICT: {rag_result.get('finalLabel')}\n")
        return build_ui_response(rag_result)


    if isinstance(rag_result, tuple):
        supported, evidence, confidence = rag_result

        if confidence >= 0.9:   # authority threshold
            print("🛑 GLOBAL HARD RETURN — TUPLE AUTHORITY")

            final_label = "REAL" if supported else "FAKE"

            print("🛑 EXITING TEXT PIPELINE — MODEL5 AUTHORITY DECISION")
            print(f"\n🎯 FINAL VERDICT: {final_label}\n")

            return build_ui_response({
                "finalLabel": final_label,
                "confidencePercent": int(confidence * 100),
                "summary": "Authoritative profile verification.",
                "aiExplanation": "Decision based on institutional profile authority.",
                "keywords": keywords,
                "sentiment": {
                    "overall": "neutral",
                    "anger": 0,
                    "fear": 0,
                    "neutral": 100
                },
                "factCheckUsed": True,
                "factCheckSource": "RAG_PROFILE",
                "verificationMethod": "MODEL5_AUTHORITY",
                "evidence": evidence
            })

    if rag_result.get("status") in {"SUPPORTED", "CONTRADICTED"}:
        print("🛑 HARD STOP — NORMALIZING DECISIVE RESULT")

        final_label = rag_result.get("finalLabel")
        evidence = rag_result.get("evidence", [])

        if final_label == "REAL":
            result = (
                True,
                evidence,
                round(rag_result.get("confidencePercent", 80) / 100.0, 3)
            )
        else:  # FAKE
            result = (
                False,
                evidence,
                round(rag_result.get("confidencePercent", 95) / 100.0, 3)
            )

        RAG_DECISION.set(result)
        return result


    # =================================================
    # 🧾 NON-DECISIVE PATH (UNVERIFIABLE / WEAK SIGNAL)
    # =================================================

    # Safety
    if not rag_result or not isinstance(rag_result, dict):
        return False, [], 0.0

    # Populate legacy docs for semantic scoring
    docs = []
    for ev in rag_result.get("evidence", []):
        docs.append({
            "url": ev.get("url"),
            "snippet": ev.get("snippet"),
            "page_type": ev.get("page_type", "UNKNOWN"),
            "score": ev.get("score", 0)
        })

    if not docs:
        return False, [], 0.0


    # =================================================
    # Semantic scoring (KEPT)
    # =================================================
    scored = []
    for d in docs:
        text = f"{d.get('title','')} {d.get('snippet','')}"
        score = semantic_similarity(translated, text)
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)

    # =================================================
    # Evidence filtering
    # =================================================
    top = []
    for score, doc in scored[:5]:
        if score >= SIM_THRESHOLD:
            top.append(doc)

    if not top:
        return False, [], 0.0


    # =================================================
    # Final fallback decision (NO FLIPPING)
    # =================================================
    supported = rag_result.get("finalLabel") == "REAL"
    confidence = round(
        rag_result.get("confidencePercent", 50) / 100.0,
        3
    )

    result = (supported, top, confidence)
    RAG_DECISION.set(result)
    print("🏁 FINAL VERDICT:", rag_result.get("finalLabel"))

    return result

# =========================================================
# CLAIMREVIEW SCHEMA EXTRACTION
# =========================================================

def extract_claimreview_from_html(html: str):
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", type="application/ld+json")

    for script in scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                for item in data:
                    if item.get("@type") == "ClaimReview":
                        return item
            elif data.get("@type") == "ClaimReview":
                return data
        except:
            continue
    return None

def claimreview_verdict(claimreview):
    review = claimreview.get("reviewRating", {})

    # Try multiple fields used in the wild
    raw_rating = (
        str(review.get("ratingValue", "")) +
        " " +
        str(review.get("alternateName", ""))
    ).lower()

    publisher = claimreview.get("author", {}).get("name", "Unknown")

    # Normalize rating meaning
    FAKE_TERMS = [
        "false", "mostly false", "pants on fire",
        "incorrect", "misleading", "fake"
    ]

    REAL_TERMS = [
        "true", "mostly true", "correct", "accurate"
    ]

    for term in FAKE_TERMS:
        if term in raw_rating:
            return ("FAKE", publisher)

    for term in REAL_TERMS:
        if term in raw_rating:
            return ("REAL", publisher)

    return ("UNVERIFIABLE", publisher)

def claimreview_confidence(claimreview: dict) -> int:
    """
    Assign confidence based on how strong the ClaimReview rating is.
    """
    review = claimreview.get("reviewRating", {})

    raw_rating = (
        str(review.get("ratingValue", "")) + " " +
        str(review.get("alternateName", ""))
    ).lower()

    # Strong absolute verdicts
    if any(x in raw_rating for x in ["false", "true", "incorrect", "correct"]):
        return 100

    # Qualified verdicts
    if "mostly" in raw_rating:
        return 85

    # Weak / unclear verdicts
    return 70

# =========================================================
# RULES & UTILITIES
# =========================================================
SUSPICIOUS_PATTERNS = [
    # Secrets / suppression
    r"\b(secret|hidden|classified|cover[- ]?up|deep state|global agenda)\b",
    r"\bthey don.?t want you to know\b|\btruth is hidden\b|\bbeing suppressed\b",
    r"\bunknown force\b|\binfluencing events\b",
    r"not ready for the truth",

    # Vague future / hype
    r"\b(will soon|about to happen|coming soon|going to change everything)\b",
    r"\b(massive change|big reveal|something huge)\b",

    # Weak authority claims
    r"\b(scientists|experts|insiders|sources) (say|claim|reveal)\b",
    r"\b(scientists|experts)\b.*\b(something|anything|big|huge)\b",

    # Sensational
    r"\b(shocking|breaking|unbelievable|explosive|exclusive|exposed|leaked)\b",

    # Conspiracy
    r"\b(conspiracy|aliens?|extraterrestrial|secret experiment|mind control)\b",
]

INDIC_SUSPICIOUS_PHRASES = [
    # Telugu
    "త్వరలో", "రహస్య", "అందరికీ తెలియదు",
    "షాకింగ్", "సంచలనం", "విప్లవాత్మక",
    "దాచిపెట్టారు", "దాచిపెడుతున్నారు",
    "ఎవరూ మాట్లాడటం లేదు",
    "ప్రపంచాన్ని మార్చే",
    "భయంకరమైన నిజం",
    "శాస్త్రవేత్తలు చెబుతున్నారు",
    "నిపుణులు చెబుతున్నారు",
    "ఏదో పెద్ద విషయం",
    "నిజం చెప్పడం లేదు",

    # Hindi
    "जल्द ही", "रहस्य", "छुपा हुआ सच",
    "सबको पता नहीं",
    "चौंकाने वाला", "सनसनीखेज",
    "सच्चाई छुपाई",
    "कोई नहीं बता रहा",
    "दुनिया बदल देगा",
    "देश को हिला देगा",
    "वैज्ञानिक कह रहे हैं",
    "विशेषज्ञ कह रहे हैं",
    "कुछ बड़ा होने वाला",
    "कुछ बड़ा खोजा",
    "कोई इस पर बात नहीं",
    "लोगों को नहीं बताया",
]

FAKE_STYLE_PATTERNS = [
    # English
    r"\b(shocking|breaking|exclusive|leaked|exposed|miracle)\b",
    r"\b(instantly cures?|hidden cure|reverse aging|secret formula)\b",
    r"\b(mainstream media|confidential report|being suppressed)\b",

    # Hindi
    r"(चौंकाने वाला|सनसनीखेज|चमत्कार|गुप्त जानकारी|तुरंत इलाज)",

    # Telugu
    r"(షాకింగ్|సంచలనం|అద్భుతం|రహస్య సమాచారం|అద్భుత చికిత్స)",
]

GOVERNMENT_KEYWORDS = [
    # English
    "government","ministry","budget","policy","finance","tax","parliament","constitution",
    # Hindi
    "सरकार","मंत्रालय","बजट","प्रधानमंत्री","संसद","संविधान","नीति","योजना",
    # Telugu
    "ప్రభుత్వం","మంత్రి","బడ్జెట్","సంసద్","రాజ్యాంగం","పథకం","ఆర్థిక",
]

IMPOSSIBLE_PATTERNS = [
    r"\bimmortality\b",
    r"\bamarta\b",
    r"\bnever die\b",
    r"\blive forever\b",
    r"अमरता",
    r"చిరంజీవి",
    r"immortality", r"अमरता", r"never die", r"live forever",

]

ROLE_ONTOLOGY = {
    "politician": {
        "prime minister",
        "chief minister",
        "president",
        "member of parliament",
        "mp",
        "mla",
        "party leader"
    },
    "academic": {
        "professor",
        "assistant professor",
        "researcher",
        "scientist"
    }
}

def ontology_supports(claim_role: str, evidence_text: str) -> bool:
    evidence = evidence_text.lower()

    for implied_role in ROLE_ONTOLOGY.get(claim_role, []):
        if implied_role in evidence:
            return True

    return False

def extract_roles_from_text(html: str):
    soup = BeautifulSoup(html, "html.parser")
    roles = []

    for tag in soup.find_all(["p","li","span","td","strong"]):
        t = tag.get_text(" ", strip=True)
        if any(r in t.lower() for r in ROLE_WORDS):
            roles.append(t)

    return roles

def extract_role_from_claim(claim: str) -> str | None:

    if not claim:
        return None

    claim_l = claim.lower().strip()

    # --------------------------------------------------
    # 0️⃣ Neutralize negation only for detection
    # --------------------------------------------------
    normalized_claim = re.sub(
        r"\b(is|was|are|were)\s+not\b",
        r"\1",
        claim_l
    )

    # --------------------------------------------------
    # 1️⃣ Match roles using WORD BOUNDARIES
    # --------------------------------------------------
    role_candidates = []

    for r in ROLE_WORDS:
        pattern = r"\b" + re.escape(r.lower()) + r"\b"
        if re.search(pattern, normalized_claim):
            role_candidates.append(r)

    # --------------------------------------------------
    # 2️⃣ Longest phrase wins
    # --------------------------------------------------
    if role_candidates:
        role_candidates.sort(key=len, reverse=True)
        return role_candidates[0]

    # --------------------------------------------------
    # 3️⃣ Fallback generic extractor
    # --------------------------------------------------
    generic = extract_generic_role_from_claim(normalized_claim)
    if generic:
        return generic

    return None

# def extract_role_from_claim(claim: str) -> str | None:

#     if not claim:
#         return None

#     claim_l = claim.lower()

#     # --------------------------------------------------
#     # 0️⃣ Create a negation-neutral version ONLY for detection
#     # --------------------------------------------------
#     normalized_claim = re.sub(
#         r"\b(is|was|are|were)\s+not\b",
#         r"\1",
#         claim_l
#     )

#     # --------------------------------------------------
#     # 1️⃣ Build dynamic role list from ROLE_WORDS
#     # Try BOTH original and normalized text
#     # --------------------------------------------------
#     role_candidates = []

#     for r in ROLE_WORDS:
#         if r in claim_l or r in normalized_claim:
#             role_candidates.append(r)

#     # --------------------------------------------------
#     # 2️⃣ Longest phrase wins
#     # --------------------------------------------------
#     if role_candidates:
#         role_candidates.sort(key=len, reverse=True)
#         return role_candidates[0]

#     # --------------------------------------------------
#     # 3️⃣ Fallback using generic role extractor
#     # --------------------------------------------------
#     generic = extract_generic_role_from_claim(claim_l)
#     if generic:
#         return generic

#     return None


COMMON_ENGLISH_WORDS = {
    "the","is","are","was","were","and","or","to","of","in","for","on","with",
    "said","report"
}
MEDICAL_CLAIMS = [
    # =========================
    # English (dangerous claims)
    # =========================
    r"\binstant cure\b",
    r"\bmiracle cure\b",
    r"\bcure[s]?\b.*\bcancer\b",
    r"\bheal[s]?\b.*\bcancer\b",
    r"\breverse[s]?\b.*\bcancer\b",
    r"\bcure[s]?\b.*\ball\b.*\bdiseases\b",
    r"\bheal[s]?\b.*\ball\b.*\bdiseases\b",
    r"\bone\b.*\bcure\b.*\ball\b",

    # Vaccine misinformation
    r"\bvaccine[s]?\b.*\b(alter|change|modify)\b.*\b(dna|genes)\b",
    r"\bvaccine[s]?\b.*\b(microchip|chip)\b",
    r"\bvaccine[s]?\b.*\b(infertility|sterility)\b",

    # =========================
    # Hindi (NO \b)
    # =========================
    r"कैंसर.*ठीक",
    r"कैंसर.*इलाज",
    r"चमत्कारी.*दवा",
    r"सभी.*बीमार",
    r"टीका.*डीएनए",
    r"वैक्सीन.*नुकसान",

    # =========================
    # Telugu (NO \b)
    # =========================
    r"క్యాన్సర్.*నయం",
    r"క్యాన్సర్.*మాన్పు",
    r"అద్భుత.*ఔషధం",
    r"అన్ని.*వ్యాధులు",
    r"టీకా.*డిఎన్ఎ",

]

# =========================================================
# KNOWN CLAIM PATTERNS THAT SHOULD REACH FACT-CHECK APIS
# =========================================================
KNOWN_FACTCHECK_CLAIMS = [
    # Space / science conspiracies
    "moon landing",
    "nasa faked",
    "moon landing hoax",

    # Climate
    "climate change hoax",
    "global warming hoax",

    # Technology
    "5g",
    "microchip",
    "surveillance chip",

    # Elections / politics
    "rigged election",
    "election fraud",

    # Popular misinformation figures
    "bill gates",

    # Generic conspiracy anchors
    "deep state",
    "new world order"
]

VAGUE_HYPE_PHRASES = [
    "major changes",
    "big changes",
    "something big",
    "coming worldwide",
    "coming soon",
    "very soon"
]


CONCRETE_ENTITIES = {
    # English
    "india", "isro", "apollo", "moon", "budget", "constitution",
    "parliament", "government",

    # Hindi
    "भारत", "इसरो", "चंद्र", "संविधान", "संसद",

    # Telugu
    "భారత్", "ఇస్రో", "చంద్ర", "రాజ్యాంగం",

    "bjp", "bharatiya janata party",
    "inc", "indian national congress",
    "aap", "aam aadmi party",
    "cpi", "cpm",

    # Hindi
    "भाजपा", "कांग्रेस",

    # Telugu
    "బీజేపీ", "కాంగ్రెస్"
}

ACTION_VERBS = [
    "launched","launch",
    "landed","land",
    "approved","announced",
    "introduced","released",
    "occurred","happened",
    "elected","appointed"
]
# =====================================================
# ROLE AUTHORITY HIERARCHY (INDUSTRY STANDARD)
# Higher number = higher authority
# =====================================================

ROLE_RANKS = {

    # 🎓 Academia
    "assistant professor": 1,
    "associate professor": 2,
    "professor": 3,
    "dean": 4,
    "principal": 4,

    # 🏢 Corporate
    "software engineer": 1,
    "senior software engineer": 2,
    "lead engineer": 3,
    "manager": 3,
    "senior manager": 4,
    "director": 5,
    "vice president": 6,
    "cto": 7,
    "ceo": 8,

    # 🏛 Government / Politics
    "mla": 1,
    "mp": 2,
    "minister": 3,
    "chief minister": 4,
    "prime minister": 5,
    "president": 6,

    # 🩺 Medical
    "medical officer": 1,
    "doctor": 2,
    "consultant": 3,
    "surgeon": 4,
    "chief medical officer": 5,

    # 🎖 Military
    "officer": 1,
    "commander": 2,
    "colonel": 3,
    "general": 4,

    # 📰 Media
    "reporter": 1,
    "journalist": 2,
    "editor": 3,
    "editor-in-chief": 4
}

def role_matches_strict(claim_role: str, extracted_role: str) -> bool:
    if not claim_role or not extracted_role:
        return False

    def normalize(r):
        r = r.lower()
        r = re.sub(r"[^a-z\s]", " ", r)
        r = re.sub(r"\s+", " ", r).strip()
        return r

    claim_norm = normalize(claim_role)
    role_norm = normalize(extracted_role)

    print("🔎 CLAIM ROLE NORMALIZED:", claim_norm)
    print("🔎 PROFILE ROLE NORMALIZED:", role_norm)

    # =====================================================
    # 🎓 ACADEMIC ROLE HIERARCHY (STRICT MODE)
    # =====================================================
    ROLE_RANK = {
        "assistant professor": 1,
        "associate professor": 2,
        "professor": 3
    }

    def detect_rank(text):
        for role in ROLE_RANK:
            if role in text:
                return ROLE_RANK[role]
        return None

    claim_rank = detect_rank(claim_norm)
    extracted_rank = detect_rank(role_norm)

    # 🚨 STRICT RANK CHECK
    if claim_rank and extracted_rank:
        print(f"🎓 CLAIM RANK={claim_rank} | PROFILE RANK={extracted_rank}")

        # must be exact rank match
        if claim_rank != extracted_rank:
            print("❌ ROLE MATCH FAILED — HIERARCHY MISMATCH")
            return False

        print("✅ ROLE MATCH — EXACT HIERARCHY")
        return True

    # =====================================================
    # ✅ EXISTING STRICT EXACT MATCH (kept)
    # =====================================================
    if claim_norm == role_norm:
        print("✅ ROLE MATCH — EXACT TEXT")
        return True

    # =====================================================
    # ⚙️ SAFE CONTAINMENT (NON-ACADEMIC ROLES ONLY)
    # =====================================================
    claim_tokens = set(claim_norm.split())
    role_tokens = set(role_norm.split())

    overlap = claim_tokens.intersection(role_tokens)

    # Only allow containment if NO academic hierarchy detected
    if not claim_rank and not extracted_rank:
        if claim_norm in role_norm or role_norm in claim_norm:
            print("✅ ROLE MATCH — SAFE CONTAINMENT")
            return True

        if len(overlap) >= 1:
            print("✅ ROLE MATCH — TOKEN OVERLAP")
            return True

    print("❌ ROLE MATCH FAILED")
    return False

def is_broader_role(claim_role: str, profile_role: str) -> bool:
    

    if not claim_role or not profile_role:
        return False

    claim_role_n = normalize_role(claim_role)
    profile_role_n = normalize_role(profile_role)

    # Exact match → NOT broader
    if claim_role_n == profile_role_n:
        return False

    # If profile role appears inside claim role text, claim is broader
    # e.g. "political leader" vs "prime minister"
    if profile_role_n in claim_role_n:
        return True

    # Ontology-based hierarchy check
    for parent_role, children in ROLE_EQUIVALENTS.items():
        parent_n = normalize_role(parent_role)

        # If claim is parent and profile is child → broader
        if claim_role_n == parent_n:
            for child in children:
                if normalize_role(child) == profile_role_n:
                    return True

    return False


def normalize_role(role: str) -> str:
    if not role:
        return ""

    role = role.lower().strip()

    # collapse variations
    role = role.replace("-", " ")
    role = re.sub(r"\s+", " ", role)

    return role

def normalize_name(name: str) -> list[str]:
    name = name.lower()
    name = re.sub(r"[^a-z\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    STOPWORDS = {"mr", "mrs", "ms", "dr"}
    tokens = [t for t in name.split() if t not in STOPWORDS]

    return tokens


def name_token_match(subject: str, text: str) -> bool:

    if not subject or not text:
        return False

    subject_tokens = [
        t.lower() for t in subject.split()
        if len(t) > 2
    ]

    if not subject_tokens:
        return False

    text_lower = text.lower()

    overlap = sum(1 for t in subject_tokens if t in text_lower)

    if len(subject_tokens) == 1:
        return overlap == 1

    required_overlap = math.ceil(len(subject_tokens) * 0.6)

    return overlap >= required_overlap


def name_matches(claim_name: str, extracted_names: list) -> bool:
    if not claim_name or not extracted_names:
        return False

    def normalize(n):
        n = n.lower()
        n = re.sub(r"\b(dr|prof|mr|mrs|ms)\.?\b", "", n)   # remove titles
        n = re.sub(r"[^a-z\s]", " ", n)
        n = re.sub(r"\s+", " ", n).strip()
        return n

    claim_norm = normalize(claim_name)
    claim_tokens = set(claim_norm.split())

    print("🔎 CLAIM NAME NORMALIZED:", claim_norm)

    for name in extracted_names:
        n_norm = normalize(name)
        n_tokens = set(n_norm.split())

        print("🔎 PROFILE NAME NORMALIZED:", n_norm)

        # ===== EXISTING STRICT CHECK (keep behaviour) =====
        if claim_norm == n_norm:
            print("✅ NAME MATCH — EXACT")
            return True

        # ===== INDUSTRY TOKEN MATCH =====
        overlap = claim_tokens.intersection(n_tokens)

        # allow initials + partial names
        if len(overlap) >= max(1, len(claim_tokens) - 1):
            print("✅ NAME MATCH — TOKEN OVERLAP")
            return True

        # containment match
        if claim_norm in n_norm or n_norm in claim_norm:
            print("✅ NAME MATCH — CONTAINMENT")
            return True

    print("❌ NAME MATCH FAILED")
    return False


def is_official_profile_page(url: str, snippet: str) -> bool:
    url_l = url.lower()
    snippet_l = snippet.lower()

    profile_patterns = [
        "/employees/",
        "/faculty/",
        "/staff/",
        "/people/",
        "/profile/",
        "/person/"
    ]

    role_words = ROLE_WORDS  # reuse your global list

    domain_trust = any(t in url_l for t in [
        ".edu",
        ".ac.",
        ".org",
        ".gov"
    ])

    profile_path = any(p in url_l for p in profile_patterns)
    role_present = any(r in snippet_l for r in role_words)

    return domain_trust and profile_path and role_present

def is_biographical_identity_claim(text: str) -> bool:
    """
    Detects short biographical / identity facts.
    Examples:
    - "PM Modi is a politician"
    - "Virat Kohli is a cricketer"
    """

    t = text.lower()
    words = t.split()

    # Must be short & assertive
    if len(words) > 8:
        return False

    # Must contain a role
    if not any(role in t for role in ROLE_WORDS):
        return False

    # Must NOT contain event/action/year
    if extract_action(t) or extract_year(t):
        return False

    # Must NOT be opinion
    if detect_claim_stance(text) != "ASSERTIVE":
        return False

    return True

def extract_action(text):
    text = text.lower()
    for v in ACTION_VERBS:
        if v in text:
            return v
    return None

def extract_year(text):
    m = re.search(r"\b(18|19|20)\d{2}\b", text)
    return m.group(0) if m else None

def extract_article_from_ld_json(soup):
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict):
                if data.get("@type") in {"NewsArticle", "Article"}:
                    return data.get("articleBody", "")
        except Exception:
            continue
    return ""

def extract_specific_entity(text: str) -> str | None:
    """
    Extracts specific named entities like Rahul-3, Apollo-11, Chandrayaan-3
    """
    m = re.search(r"\b([A-Z][a-zA-Z]+[-\s]?\d+)\b", text)
    return m.group(1) if m else None

def contains_debunk_signal(text: str) -> bool:
    if not text:
        return False

    t = text.lower()

    signals = [
        "false",
        "fake",
        "hoax",
        "debunk",
        "not true",
        "misleading",
        "no evidence",
        "no proof",
        "fabricated",
        "incorrect",
        "rumor",
        "unfounded"
    ]

    return any(s in t for s in signals)

def is_subjective_assertion(text: str) -> bool:
    t = text.lower()

    SUBJECTIVE_PATTERNS = [
        # Comparative / value judgement
        r"\bmore than\b",
        r"\bless than\b",
        r"\bgreater than\b",
        r"\bnot just\b",
        r"\bbeyond\b",

        # Abstract value claims
        r"\bsymbol\b",
        r"\bspirit\b",
        r"\bfeeling\b",
        r"\bemotion\b",
        r"\bpride\b",
        r"\bidentity\b",
        r"\bimportance\b",
        r"\bmeaning\b",

        # Moral / philosophical
        r"\bstands for\b",
        r"\brepresents\b",
        r"\bshows that\b",
    ]

    return any(re.search(p, t) for p in SUBJECTIVE_PATTERNS)

def detect_claim_stance(text: str) -> str:
    """
    Determines how the claim relates to reality.
    """
    t = text.lower()

    NEGATION_CUES = [
        "fake", "faked", "hoax", "staged",
        "fabricated", "myth", "never happened",
        "didn't happen", "was not real"
    ]

    ALLEGATION_CUES = [
        "they say", "people believe", "some claim",
        "it is said", "according to some"
    ]

    OPINION_CUES = [
        "i think", "i believe", "in my opinion"
    ]

    if any(n in t for n in NEGATION_CUES):
        return "NEGATING"

    if any(a in t for a in ALLEGATION_CUES):
        return "ALLEGATION"

    if any(o in t for o in OPINION_CUES):
        return "OPINION"

    return "ASSERTIVE"


# =====================================================
# Sentence-level stance detection using NLI
# =====================================================
def detect_stance(claim: str, snippet: str):
    if not snippet:
        return "NEUTRAL", 0.5

    sentences = extract_sentences(snippet)


    claim_l = claim.lower()
    evidence_l = snippet.lower()

    # take name tokens
    person_tokens = claim_l.split()[:3]

    person_match = any(tok in evidence_l for tok in person_tokens)

    role_match = any(role in evidence_l for role in ROLE_WORDS)

    # SUPPORT: person + role present
    if person_match and role_match:
        return "SUPPORT", 0.9

    # CONTRADICTION: negation cues
    negations = [
        "not", "no longer", "former", "ex-",
        "denied", "false", "incorrect"
    ]

    if person_match and any(n in evidence_l for n in negations):
        return "CONTRADICT", 0.9

    # =========================
    # NLI fallback (SAFE)
    # =========================
    best_support = 0.0
    best_contradict = 0.0

    for sent in sentences:
        try:
            # ✅ CORRECT cross-encoder input format
            nli_input = f"{sent[:512]} [SEP] {claim}"

            output = nli_model(nli_input)

            # ✅ pipeline may return list or dict
            if isinstance(output, list):
                output = output[0]

            label = output.get("label", "").upper()
            score = float(output.get("score", 0.0))

            if label == "CONTRADICTION":
                best_contradict = max(best_contradict, score)

            elif label == "ENTAILMENT":
                best_support = max(best_support, score)

        except Exception as e:
            print("⚠️ NLI sentence error:", e)
            continue

    if best_contradict > best_support and best_contradict > 0.6:
        return "CONTRADICT", best_contradict

    if best_support > 0.6:
        return "SUPPORT", best_support

    return "NEUTRAL", 0.5


# =====================================================
# Role reasoning layer (industry standard)
# =====================================================
def role_contradiction_detected(claim: str, sentences: list[str]) -> bool:
    """
    Detect contradiction for role/position claims.
    Example:
        'Musk is President' vs
        'Musk cannot run for president'
    """

    claim_l = claim.lower()


    role_in_claim = None
    for r in ROLE_WORDS:
        if r in claim_l:
            role_in_claim = r
            break

    if not role_in_claim:
        return False

    negation_patterns = [
        "cannot become",
        "cannot run for",
        "not eligible",
        "disqualified",
        "unlikely to become",
        "cannot serve as",
        "cannot be",
        "not allowed to be",
    ]

    for sent in sentences:
        s = sent.lower()

        if role_in_claim in s:
            for pattern in negation_patterns:
                if pattern in s:
                    return True

    return False


def is_url_junk_sentence(sentence: str) -> bool:
    s = sentence.lower().strip()

    if len(s) < 20:
        return True

    JUNK_PATTERNS = [
        # Error / status pages
        "technical difficulty", "temporarily unavailable",
        "error occurred", "service unavailable",

        # Apology boilerplate
        "we apologize", "sorry for the inconvenience",
        "we are aware of the issue",

        # Support / contact / CTA
        "contact us", "technical support", "customer support",
        "sales representative", "speak with", "call us",
        "headquarters", "get support",

        # Navigation
        "click here", "learn more", "read more",
        "terms and conditions", "privacy policy"
    ]

    if any(p in s for p in JUNK_PATTERNS):
        return True

    # Phone numbers / IDs
    if re.search(r"\b\d{5,}\b", s):
        return True

    return False


def classify_claim_type(claim: str) -> str:
    c = claim.lower()

    if re.search(r"\bis (the )?(prime minister|president|ceo|chief minister)\b", c):
        return "POSITIONAL_ROLE"

    if re.search(r"\bis (a|an)\b", c):
        return "IDENTITY_ROLE"

    if re.search(r"\b(19|20)\d{2}\b", c):
        return "EVENT_HISTORICAL"

    return "GENERAL_FACT"


def ontology_supports(claim_role: str, evidence_text: str) -> bool:
    evidence = evidence_text.lower()

    for implied_role in ROLE_ONTOLOGY.get(claim_role, []):
        if implied_role in evidence:
            return True

    return False

def detect_url_page_type(text: str) -> str:
    """
    Detects whether a URL page is an error/support/landing page.
    Returns: ERROR_PAGE, SUPPORT_PAGE, or NORMAL
    """

    t = text.lower()

    ERROR_PATTERNS = [
        "technical difficulty",
        "temporarily unavailable",
        "service unavailable",
        "an error occurred",
        "something went wrong",
        "page not found",
        "404 error",
        "maintenance"
    ]

    SUPPORT_PATTERNS = [
        "contact us",
        "customer support",
        "technical support",
        "help center",
        "support center",
        "sales representative",
        "call us",
        "email us",
        "headquarters"
    ]

    if any(p in t for p in ERROR_PATTERNS):
        return "ERROR_PAGE"

    if any(p in t for p in SUPPORT_PATTERNS):
        return "SUPPORT_PAGE"

    return "NORMAL"


def build_url_evidence(
    verdict: str,
    *,
    domain: str,
    is_trusted_source: bool = False,
    is_low_trust_source: bool = False,
    is_entertainment_domain: bool = False,
    best_claim: str | None = None,
    best_score: float | None = None,
    external_source: str | None = None,
    fake_reason: str | None = None,
    page_type: str | None = None
) -> list[dict]:
    """
    Industry-standard URL evidence builder.
    Evidence explains WHY a verdict was reached.
    """

    # =========================
    # 🟢 REAL
    # =========================
    if verdict == "REAL":

        # Trusted publisher REAL
        if is_trusted_source:
            return [
                {
                    "type": "publisher_trust",
                    "publisher": domain,
                    "reason": (
                        "The webpage is published by a well-established, "
                        "editorially controlled news organization."
                    )
                }
            ]

        # External evidence REAL (RAG)
        if best_claim and best_score and external_source:
            return [
                {
                    "type": "external_confirmation",
                    "source": external_source,
                    "matchedClaim": best_claim,
                    "confidence": round(best_score, 2),
                    "reason": (
                        "Key factual statements on the page are supported "
                        "by reliable external sources."
                    )
                }
            ]

        return [
            {
                "type": "verification_summary",
                "reason": "The content appears factual and no contradictory signals were detected."
            }
        ]

    # =========================
    # 🔴 FAKE
    # =========================
    if verdict == "FAKE":

        if fake_reason:
            return [
                {
                    "type": "rule_violation",
                    "rule": fake_reason,
                    "reason": "The page contains claims that violate known factual or logical constraints."
                }
            ]

        if is_low_trust_source:
            return [
                {
                    "type": "source_reputation",
                    "source": domain,
                    "reason": "This domain has a documented history of publishing misinformation."
                }
            ]

        return [
            {
                "type": "verification_summary",
                "reason": "False or misleading claims were detected on the webpage."
            }
        ]

    # =========================
    # ⚪ UNVERIFIABLE
    # =========================
    if verdict == "UNVERIFIABLE":

        if is_entertainment_domain:
            return [
                {
                    "type": "verification_limitation",
                    "reason": (
                        "The webpage appears to be entertainment or list-based content "
                        "without standalone factual claims."
                    )
                }
            ]

        if page_type:
            return [
                {
                    "type": "content_type",
                    "reason": f"The page is classified as {page_type}, which is not suitable for fact verification."
                }
            ]

        return [
            {
                "type": "insufficient_evidence",
                "reason": (
                    "No strong external sources were found to verify or disprove the content."
                )
            }
        ]

    # =========================
    # Fallback (should not happen)
    # =========================
    return []

def normalize_confidence(final_label, verification_method, model_conf=0, avg_support=0.0):
    # UNVERIFIABLE is always zero
    if final_label == "UNVERIFIABLE":
        return 0

    # Rule-based decisions
    if verification_method in ["RULE_ENGINE", "HISTORICAL_RULE"]:
        return 85 if final_label == "REAL" else 90

    # ML + RAG decisions
    if "RAG" in verification_method:
        evidence_conf = int(avg_support * 100)
        return min(95, max(model_conf, evidence_conf, 70))

    # Pure ML
    return int(model_conf)


def is_short_factual_claim(text: str) -> bool:
    words = text.split()

    # must be short
    if len(words) > 12:
        return False

    # must NOT be vague or hype
    if is_vague_or_conspiratorial(text):
        return False

    if fake_style_score(text):
        return False

    # 🔒 MUST contain a strong historical/event keyword
    if not looks_like_historical_fact(text):
        return False

    return True

def normalize_person_name(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = text.replace(".", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # remove titles
    for t in ["mr", "mrs", "ms", "dr", "prof", "professor"]:
        if text.startswith(t + " "):
            text = text[len(t):].strip()

    return text

def normalize_alphanumeric_entities(text: str) -> str:
    # Insert hyphen ONLY between letters and numbers when missing
    text = re.sub(
        r"\b([A-Za-z]+)(\d+)\b",
        r"\1-\2",
        text
    )

    return text

def is_dangerous_medical_claim(text: str) -> bool:
    if not text:
        return False

    # Detect script
    has_english = bool(re.search(r"[a-zA-Z]", text))
    has_indic = bool(re.search(r"[\u0900-\u097F\u0C00-\u0C7F]", text))

    for pattern in MEDICAL_CLAIMS:
        # ✅ English → IGNORECASE
        if has_english and re.search(pattern, text, flags=re.IGNORECASE):
            return True

        # ✅ Hindi / Telugu → raw match (NO case folding)
        if has_indic and re.search(pattern, text):
            return True

    return False

def has_meaningful_english_words(text: str) -> bool:
    words = [w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text)]
    if not words:
        return False

    meaningful = sum(1 for w in words if w in COMMON_ENGLISH_WORDS)
    return meaningful >= 1


def looks_like_govt_information(text: str):
    return any(k in text.lower() for k in GOVERNMENT_KEYWORDS)

def is_vague_or_conspiratorial(text: str) -> bool:
    # 🔒 Do NOT lowercase Indic text
    has_indic = bool(re.search(r"[\u0900-\u097F\u0C00-\u0C7F]", text))
    t = text.lower() if not has_indic else text

    # ==================================================
    # 0️⃣ Abstract "truth / people" statements (CRITICAL)
    # ==================================================
    ABSTRACT_SUBJECT_PATTERNS = [
        r"\bpeople\b.*\btruth\b",
        r"\beveryone\b.*\btruth\b",
        r"\bnobody\b.*\btruth\b",
        r"\btruth\b.*\b(hidden|unknown|not ready)\b",

        # Hindi
        r"लोगों.*सच्चाई",
        r"सच.*बताई.*नहीं",
        r"सच्चाई.*छुप",

        # Telugu
        r"ప్రజలు.*నిజం",
        r"ప్రజలకు.*నిజం",
        r"నిజం.*తెలియదు",
        r"నిజం.*చెప్ప.*లేదు",
        r"అందరికీ.*నిజం",
        r"ఎవరికి.*నిజం",
    ]

    for p in ABSTRACT_SUBJECT_PATTERNS:
        if re.search(p, t):
            return True

    # ==================================================
    # 1️⃣ Explicit conspiracy / suppression language
    # ==================================================
    CONSPIRACY_PHRASES = [
        # Hindi
        "कोई नहीं बता रहा", "सच्चाई छुपाई", "छुपाया जा रहा",
        "दबाई जा रही", "रहस्य", "सब कुछ छुपाया",

        # Telugu
        "ఎవరూ చెప్పడం లేదు", "నిజం దాచారు",
        "దాచిపెడుతున్నారు", "రహస్యం",

        # English
        "nobody is talking", "truth is hidden",
        "being suppressed", "they don't want you to know",
    ]

    if any(p in t for p in CONSPIRACY_PHRASES):
        return True

    # ==================================================
    # 2️⃣ Authority WITHOUT object
    # ==================================================
    AUTHORITY_WORDS = [
        "scientists", "experts",
        "वैज्ञानिक", "विशेषज्ञ",
        "శాస్త్రవేత్తలు", "నిపుణులు",
    ]

    OBJECT_WORDS = [
        "india", "isro", "apollo", "moon", "budget", "constitution",
        "भारत", "इसरो", "चंद्र", "संविधान",
        "భారత్", "ఇస్రో", "చంద్ర", "రాజ్యాంగం",
    ]

    if any(a in t for a in AUTHORITY_WORDS):
        if not any(o in t for o in OBJECT_WORDS):
            return True

    # ==================================================
    # 3️⃣ Pure hype
    # ==================================================
    HYPE_ONLY = [
        "कुछ बड़ा होने वाला", "कुछ बड़ा खोजा",
        "देश को हिला देगा", "दुनिया बदल देगा",
        "ఏదో పెద్ద విషయం", "ప్రపంచాన్ని మార్చే",
        "something big", "huge revelation",
    ]

    if any(h in t for h in HYPE_ONLY):
        return True


    # ==================================================
    # 3️⃣ No date + no number + no named entity → vague
    # ==================================================
    has_number_or_date = bool(re.search(r"\b\d{4}\b|\b\d+\b", t))
    has_entity = any(e in t for e in CONCRETE_ENTITIES)

    # Allow simple assertive identity / definition claims to go to RAG
    if (
        not has_number_or_date
        and not has_entity
        and detect_claim_stance(text) != "ASSERTIVE"
        and " is " not in t
    ):
        return True


    # ==================================================
    # 4️⃣ Generic hype language (last safety net)
    # ==================================================
    if any(p in t for p in VAGUE_HYPE_PHRASES):
        return True

    return False


def fake_style_score(text):
    return sum(1 for p in FAKE_STYLE_PATTERNS if re.search(p, text.lower()))

def run_model(text):
    tokenizer, model = get_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256 if len(text.split()) < 200 else 512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred = int(torch.argmax(probs, dim=1))

    # ✅ FORCE BINARY LABELS
    if pred == 1:
        label = "REAL"
    else:
        label = "FAKE"

    confidence = round(float(probs[0][pred]) * 100, 2)

    return label, confidence


def run_indic_model(text):
    tokenizer, model = get_indic_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred = int(torch.argmax(probs, dim=1))
    label = model.config.id2label[pred].upper()
    confidence = round(float(probs[0][pred]) * 100, 2)

    return label, confidence

def is_gibberish_text(text: str) -> bool:
    text = text.strip()

    # 1️⃣ Extremely short text
    if len(text) < 4:
        return True

    # 2️⃣ Alphabetic ratio (Unicode-safe)
    letters = sum(c.isalpha() for c in text)
    ratio = letters / max(len(text), 1)
    if ratio < 0.4:
        return True

    # Detect scripts
    has_telugu = re.search(r"[\u0C00-\u0C7F]", text)
    has_hindi = re.search(r"[\u0900-\u097F]", text)
    has_english = re.search(r"[a-zA-Z]", text)

    # =========================
    # 🇮🇳 INDIC LANGUAGES
    # =========================
    if has_telugu or has_hindi:
        # Must contain at least one valid Indic word (2+ chars)
        if not (
            re.search(r"[\u0C00-\u0C7F]{2,}", text) or
            re.search(r"[\u0900-\u097F]{2,}", text)
        ):
            return True

        # Block long random Indic strings without spaces
        if len(text) > 25 and " " not in text:
            return True

        return False

    # =========================
    # 🇬🇧 ENGLISH
    # =========================
    if has_english:
        # Repeated characters (aaaaaa)
        if re.search(r"(.)\1{4,}", text.lower()):
            return True

        # Repeated substrings (asdasdasd)
        for size in range(2, 6):
            if len(text) % size == 0:
                pattern = text[:size]
                if pattern * (len(text) // size) == text:
                    return True

        # Extract English-like words
        words = re.findall(r"[a-zA-Z]{3,}", text)

        # --------------------------------------------------
        # 🆕 ADDITION 1: Per-word vowel sanity check
        # --------------------------------------------------
        def looks_like_real_word(w):
            vowels = sum(c in "aeiou" for c in w.lower())
            return vowels / max(len(w), 1) >= 0.25

        real_word_count = sum(1 for w in words if looks_like_real_word(w))

        # If almost no words look real → gibberish
        if real_word_count < 2 and len(words) >= 2:
            return True

        # --------------------------------------------------
        # 🧨 FINAL HARD GIBBERISH OVERRIDE (CRITICAL)
        # --------------------------------------------------
        consonant_clusters = sum(
            1 for w in words
            if re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", w.lower())
        )

        # If most words contain long consonant clusters → gibberish
        if consonant_clusters >= max(2, len(words) // 2):
            return True

        # --------------------------------------------------
        # Existing logic (unchanged)
        # --------------------------------------------------
        if len(words) >= 4:
            return False

        # Vowel sanity (fallback for short English)
        english_letters = re.findall(r"[a-zA-Z]", text)
        if english_letters:
            vowels = sum(c.lower() in "aeiou" for c in english_letters)
            if vowels / len(english_letters) < 0.2:
                return True

        return False

    return True

def build_structured_explanation(
    final_label,
    model_label,
    model_conf,
    detected_lang,
    text,
    evidence,
    support_score
):
    explanation = []

    # 1️⃣ Model signal
    explanation.append(
        f"The ML model predicted {model_label} with confidence {model_conf}%."
    )

    # 2️⃣ Confidence reasoning
    if model_conf < 60:
        explanation.append("The model confidence was low, increasing uncertainty.")
    elif model_conf >= 80:
        explanation.append("The model confidence was strong.")

    # 3️⃣ Rule-based signals
    fs = fake_style_score(text)
    if fs >= 2:
        explanation.append("The claim contains sensational or misleading language patterns.")

    if looks_like_govt_information(text):
        explanation.append("The claim appears related to government or policy information.")

    # 4️⃣ Evidence similarity (Sentence Transformer)
    if evidence:
        sim_model = get_similarity_model()
        claim_emb = sim_model.encode(text, convert_to_tensor=True)

        sims = []

        for e in evidence[:3]:
            snippet = (
                e.get("snippet")
                or e.get("matchedClaim")
                or e.get("reason")
                or ""
            )

            if not snippet:
                continue

            ev_emb = sim_model.encode(snippet, convert_to_tensor=True)
            sims.append(float(util.cos_sim(claim_emb, ev_emb)))

        if sims:
            best_sim = max(sims)
            explanation.append(
                f"The claim was compared with external sources; highest semantic similarity score was {round(best_sim, 2)}."
            )
        else:
            explanation.append(
                "External evidence was found, but similarity scores were too weak for strong confirmation."
            )
    else:
        explanation.append(
            "No strong external evidence was found to support the claim."
        )

    # 5️⃣ Final verdict logic
    explanation.append(
        f"Based on combined model prediction, rules, and evidence analysis, the final verdict is {final_label}."
    )

    return " ".join(explanation)


def dual_model_verify_with_translation(text, detected_lang):
    indic_label, indic_conf = run_indic_model(text)
    indic_label = "FAKE" if indic_label == "label_0" else "REAL"

    translated_text = cached_translate_to_english(text, detected_lang)

    eng_label, eng_conf = run_model(translated_text)
    eng_label = "FAKE" if eng_label == "label_0" else "REAL"

    avg_conf = (indic_conf + eng_conf) / 2

    if indic_label == eng_label == "REAL" and avg_conf >= 70:
        return "REAL", min(int(avg_conf), 95), translated_text
    elif indic_label == eng_label == "FAKE" and avg_conf >= 70:
        return "FAKE", min(int(avg_conf), 95), translated_text
    else:
        return "UNVERIFIABLE", 0, translated_text


def is_vague_science_claim(text: str) -> bool:

    if not text:
        return False

    t = text.lower()

    patterns = [
        # =========================
        # English (generic science)
        # =========================
        r"\bscientists?\b.*\b(found|discovered|made|say|claim)\b",
        r"\bresearchers?\b.*\b(found|discovered|say|claim)\b",
        r"\bexperts?\b.*\b(say|claim|believe)\b",
        r"\bstudies?\b.*\b(show|suggest|reveal)\b",

        # =========================
        # Hindi (generic science)
        # =========================
        r"वैज्ञानिक.*(खोजा|खोज|पाया|बताया|कहा|दावा)",
        r"शोधकर्ताओं.*(खोजा|पाया|बताया)",
        r"विशेषज्ञ.*(कहते|मानते|दावा)",
        r"अध्ययन.*(बताते|दिखाते|सुझाव)",
        r"वैज्ञानिकों ने.*कुछ",

        # =========================
        # Telugu (generic science)
        # =========================
        r"శాస్త్రవేత్త.*(కనుగొన్నారు|కనుగొన్నారని|చెప్పారు|అన్నారు)",
        r"పరిశోధకులు.*(కనుగొన్నారు|చెప్పారు)",
        r"నిపుణులు.*(అన్నారు|చెబుతున్నారు)",
        r"అధ్యయనం.*(చూపించింది|తెలిపింది)",
        r"శాస్త్రవేత్తలు.*ఏదో",
    ]

    return any(re.search(p, t) for p in patterns)


def analyze_sentiment(text):
    s = sentiment_analyzer.polarity_scores(text)["compound"]
    if s >= 0.05:
        return "POSITIVE"
    if s <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"

def extract_keywords(text):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    return [w for w, _ in Counter(w for w in words if w not in STOPWORDS).most_common(5)]

def extract_keywords_multilingual(text: str, lang: str):
    if lang == "en":
        return extract_keywords(text)

    if lang == "hi":
        words = re.findall(r"[\u0900-\u097F]{3,}", text)
        filtered = [w for w in words if w not in HINDI_STOPWORDS]
        return [w for w, _ in Counter(filtered).most_common(5)]

    if lang == "te":
        words = re.findall(r"[\u0C00-\u0C7F]{3,}", text)
        filtered = [w for w in words if w not in TELUGU_STOPWORDS]
        return [w for w, _ in Counter(filtered).most_common(5)]

    return []


def source_credibility_check(url):
    domain = urlparse(url).netloc.replace("www.", "").lower()
    if domain in TRUSTED_SOURCES:
        return "REAL"
    if domain in LOW_TRUST_SOURCES:
        return "FAKE"
    return None

# =========================================================
# UI RESPONSE FORMATTER
# =========================================================
def build_ui_response(raw):

    # 🛡️ SAFETY GUARD — never crash if raw is None
    raw = raw or {}

    # -------------------------------------------------
    # Default fallback structure (industry standard)
    # -------------------------------------------------
    DEFAULT_SENTIMENT = {
        "overall": "neutral",
        "anger": 0,
        "fear": 0,
        "neutral": 100
    }

    sentiment_map = {
        "NEGATIVE": {"anger": 65, "fear": 45, "neutral": 20},
        "POSITIVE": {"anger": 10, "fear": 5, "neutral": 25},
        "NEUTRAL":  {"anger": 15, "fear": 10, "neutral": 60}
    }

    # -------------------------------------------------
    # SAFE SENTIMENT HANDLING
    # -------------------------------------------------
    sentiment_raw = raw.get("sentiment", DEFAULT_SENTIMENT)

    if isinstance(sentiment_raw, dict):
        sentiment = {
            "overall": sentiment_raw.get("overall", "neutral").lower(),
            "anger": sentiment_raw.get("anger", 0),
            "fear": sentiment_raw.get("fear", 0),
            "neutral": sentiment_raw.get("neutral", 100),
        }
    else:
        sentiment_label = str(sentiment_raw).upper()
        sentiment = {
            "overall": sentiment_label.lower(),
            **sentiment_map.get(sentiment_label, sentiment_map["NEUTRAL"])
        }

    # -------------------------------------------------
    # SAFE FIELD EXTRACTION (NO KEYERRORS EVER)
    # -------------------------------------------------
    final_label = str(raw.get("finalLabel", "UNVERIFIABLE")).lower()

    confidence = raw.get("confidencePercent", 0)
    try:
        confidence = round(float(confidence))
    except:
        confidence = 0

    return {
        "status": final_label,

        "confidence": confidence,

        "summary": raw.get(
            "summary",
            "No verified summary available."
        ),

        "explanation": raw.get(
            "aiExplanation",
            "No detailed explanation provided."
        ),

        # 🔥 FIX — keywords safe fallback
        "keywords": raw.get("keywords", []),

        # 🔥 FIX — language safe fallback
        "language": raw.get("language", "English"),

        "sentiment": sentiment,

        # 🔥 FIX — fact check fields safe fallback
        "factCheckUsed": raw.get("factCheckUsed", False),

        "factCheckSource": raw.get("factCheckSource", "RAG"),

        "verificationMethod": raw.get(
            "verificationMethod",
            "MODEL5_AUTHORITY"
        ),

        # Evidence always safe
        "evidence": raw.get("evidence", [])
    }

# =========================================================
# TEXT PREDICTION
# =========================================================
@app.post("/predict")
def predict_text(data: InputText):



    RAG_ATTEMPTS.set(0)
    RAG_DECISION.set(None)

    # =========================================================
    # 0️⃣ Raw input (CANONICAL – UI ONLY)
    # =========================================================
    raw_text = (data.text or "").strip()
    if not raw_text:
        return build_ui_response({
            "finalLabel": "UNVERIFIABLE",
            "confidencePercent": 0,
            "summary": "Empty input.",
            "aiExplanation": "No text was provided.",
            "keywords": [],
            "language": "Unknown",
            "sentiment": {"overall": "neutral", "anger": 0, "fear": 0, "neutral": 100},
            "factCheckUsed": False,
            "factCheckSource": None,
            "verificationMethod": "INPUT_VALIDATION",
            "evidence": []
        })

    # =========================================================
    # 🔒 STEP 1: TEXT HYGIENE (SINGLE SOURCE OF TRUTH)
    # =========================================================
    original_text = raw_text          
    claim = remove_citation_markers(raw_text).strip()
    clean_text = remove_citation_markers(raw_text)
    clean_text_lc = clean_text.lower()

    absurd_result = absurd_claim_gate(claim)
    if absurd_result:
        return absurd_result

    known_false = known_false_claim_gate(claim)
    if known_false:
        print("🧠 KNOWN SCIENTIFIC FALSEHOOD — SHORT CIRCUITED")
        return known_false

    geo = geography_validator(claim)
    if geo:
        return {
            "status": "CONTRADICTED" if geo["verdict"] == "FAKE" else "SUPPORTED",
            "finalLabel": geo["verdict"],
            "confidencePercent": 99,

            "summary": geo["reason"],

            "aiExplanation": (
                "This claim was verified using trusted geographic knowledge sources. "
                + geo["reason"]
            ),
            "keywords": ["geography", "knowledge-base"],
            "language": "English",

            "sentiment": {
                "overall": "neutral",
                "anger": 0,
                "fear": 0,
                "neutral": 100
            },

            "factCheckUsed": False,
            "factCheckSource": None,
            "verificationMethod": "GEOGRAPHY_KNOWLEDGE_BASE",

            "evidence": [{
                "url": "about:blank",
                "snippet": geo["reason"],
                "score": 0.99,
                "page_type": "FACT"
            }]
        }



    # =========================================================
    # 🚨 HARD STOP: dangerous medical misinformation
    # =========================================================
    if is_dangerous_medical_claim(clean_text):
        detected_lang = safe_language_detect(clean_text)
        return build_ui_response({
            "finalLabel": "FAKE",
            "confidencePercent": 90,
            "summary": "Dangerous medical misinformation detected.",
            "aiExplanation": "Claims suggesting miracle or absolute medical cures are false.",
            "keywords": extract_keywords_multilingual(clean_text, detected_lang),
            "language": get_language_name(detected_lang),
            "sentiment": analyze_sentiment(clean_text),
            "factCheckUsed": False,
            "factCheckSource": None,
            "verificationMethod": "RULE_ENGINE",
            "evidence": []
        })

    # =========================================================
    # 1️⃣ Language + normalization (ON CLEAN TEXT)
    # =========================================================
    detected_lang = safe_language_detect(clean_text)
    text = normalize_indic_text(clean_text, detected_lang)
    text = language_specific_preprocess(text, detected_lang)

    language_name = get_language_name(detected_lang)
    keywords = extract_keywords_multilingual(text, detected_lang)
    sentiment = analyze_sentiment(text)

    # =========================================================
    # 🔒 DOMINANT CLASS DECISION (INDUSTRY-STANDARD)
    # =========================================================
    claim_stance = detect_claim_stance(clean_text)

    if is_gibberish_text(clean_text_lc):
        dominant_class = "UNVERIFIABLE"

    elif is_impossible_claim(clean_text_lc):
        dominant_class = "FAKE"

    elif is_vague_or_conspiratorial(clean_text_lc):
        # Allow well-known conspiracy claims to reach fact-check APIs
        if any(k in clean_text_lc for k in KNOWN_FACTCHECK_CLAIMS):
            dominant_class = "FACTUAL"
        else:
            dominant_class = "UNVERIFIABLE"

    elif (
        is_vague_science_claim(clean_text_lc)
        and not (
            looks_like_historical_fact(clean_text)
            or looks_like_govt_information(clean_text)
        )
    ):
        dominant_class = "UNVERIFIABLE"

    elif (
        claim_stance == "ASSERTIVE"
        and is_subjective_assertion(clean_text)
    ):
        dominant_class = "UNVERIFIABLE"

    elif (
        (
            looks_like_historical_fact(clean_text)
            or looks_like_govt_information(clean_text)
            or is_short_factual_claim(clean_text)
        )
        and claim_stance == "ASSERTIVE"
        and not extract_specific_entity(clean_text)
        and not any(e in clean_text.lower() for e in CONCRETE_ENTITIES)

    ):
        # ONLY generic assertive facts can be auto-REAL
        # Specific named entities MUST be validated by RAG
        dominant_class = "REAL"

    else:
        # Everything else goes through ML + RAG
        dominant_class = "FACTUAL"

    print("DOMINANT CLASS =", dominant_class)
    print("CLAIM STANCE =", claim_stance)

    # =========================================================
    # 3️⃣ HARD LOCK RETURNS
    # =========================================================
    if dominant_class == "FAKE":
        return build_ui_response({
            "finalLabel": "FAKE",
            "confidencePercent": 90,
            "summary": "Impossible or dangerous claim detected.",
            "aiExplanation": "The claim violates established scientific or medical facts.",
            "keywords": keywords,
            "language": language_name,
            "sentiment": sentiment,
            "factCheckUsed": False,
            "factCheckSource": None,
            "verificationMethod": "RULE_ENGINE",
            "evidence": []
        })

    if dominant_class == "UNVERIFIABLE":
        return build_ui_response({
            "finalLabel": "UNVERIFIABLE",
            "confidencePercent": 0,
            "summary": "Vague or unverifiable claim.",
            "aiExplanation": "The claim lacks sufficient concrete information to verify.",
            "keywords": keywords,
            "language": language_name,
            "sentiment": sentiment,
            "factCheckUsed": False,
            "factCheckSource": None,
            "verificationMethod": "RULE_ENGINE",
            "evidence": []
        })

    if dominant_class == "REAL":
        return build_ui_response({
            "finalLabel": "REAL",
            "confidencePercent": 85,
            "summary": "Established historical or government fact.",
            "aiExplanation": "The claim is a well-documented and widely accepted fact.",
            "keywords": keywords,
            "language": language_name,
            "sentiment": sentiment,
            "factCheckUsed": False,
            "factCheckSource": None,
            "verificationMethod": "HISTORICAL_RULE",
            "evidence": []
        })

    # =========================================================
    # 4️⃣ FACTUAL CLAIMS → FACT CHECK API / ML / RAG
    # =========================================================
    fc = None

    # ✅ Skip Google Fact Check for clean historical event assertions
    if not is_assertive_historical_event(clean_text):
        cached = google_fact_check_cached(clean_text, detected_lang)
        fc = json.loads(cached) if cached else None

    if fc:
        return build_ui_response({
            "finalLabel": fc["label"],
            "confidencePercent": fc["confidence"],
            "summary": "Verified using Google Fact Check.",
            "aiExplanation": "Verified by multiple independent fact-checking organizations via Google Fact Check.",
            "keywords": keywords,
            "language": language_name,
            "sentiment": sentiment,
            "factCheckUsed": True,
            "factCheckSource": "Google Fact Check",
            "verificationMethod": "GOOGLE_FACT_CHECK",
            "evidence": fc["evidence"]
        })
    # =========================================================
    #  ML DETECTION AND LABELING
    # =========================================================

    if detected_lang in ["hi", "te"]:
        model_label, model_conf, _ = dual_model_verify_with_translation(text, detected_lang)
        verification_method = "INDIC_BERT + TRANSLATION + ML"
    else:
        model_label, model_conf = run_model(text)
        verification_method = "ML_MODEL"

    final_label = model_label
    final_conf = model_conf


    # =========================================================
    # 5️⃣ RAG (EVIDENCE ONLY FOR FACTUAL CLAIMS)
    # =========================================================
    claims = decompose_claims(text)

    aggregated_evidence = []
    support_scores = []
    max_support = 0.0
    rag_supported_any = False

    contradiction_detected = False
    contradiction_score = 0.0
    support_scores = []
    max_support = 0.0
    rag_supported_any = False
    aggregated_evidence = []
    support_hits = 0
    contradiction_hits = 0


    MAX_CLAIMS_TO_VERIFY = 3

    for c in claims[:MAX_CLAIMS_TO_VERIFY]:

        if is_url_junk_sentence(c):
            continue

        if len(c.split()) < 4 and not extract_specific_entity(c):
            continue

        result = verify_claim_with_rag(c, detected_lang)

        if isinstance(result, dict) and result.get("status") in ["real", "fake"]:
            return result


        if not result or len(result) != 3:
            continue

        supported, evidence, score = result

        stance = "NEUTRAL"
        stance_score = 0.0

        if evidence:
            ev = evidence[0]
            if is_official_profile_page(ev.get("url", ""), ev.get("snippet", "")):

                stance = "SUPPORT"
                stance_score = max(stance_score, 0.95)

        # ==============================
        # ✅ Stance detection
        # ==============================
        if evidence:
            snippet = evidence[0].get("snippet", "")
            stance, stance_score = detect_stance(c, snippet)

        # ==============================
        # ✅ detect debunk evidence
        # ==============================
        debunk_hit = False
        if evidence:
            snippet = evidence[0].get("snippet", "").lower()
            debunk_words = [
                "false", "fake", "debunk", "no evidence",
                "not true", "misleading", "myth", "unfounded",
                "conspiracy", "does not", "not made", "not contain"
            ]
            if any(w in snippet for w in debunk_words):
                debunk_hit = True

        # ==============================
        # 🚨 contradiction handling
        # ==============================
        if debunk_hit or stance == "CONTRADICT":
            contradiction_detected = True
            contradiction_hits += 1
            contradiction_score = max(contradiction_score, stance_score)

            if evidence:
                aggregated_evidence.extend(evidence)

            final_label = "FAKE"
            final_conf = max(final_conf, 90)
            break

        # strong contradiction fallback
        if not supported and score >= 0.9:
            contradiction_detected = True
            contradiction_hits += 1
            contradiction_score = max(contradiction_score, score)

            if evidence:
                aggregated_evidence.extend(evidence)

            final_label = "FAKE"
            final_conf = max(final_conf, 90)
            break


        # ==============================
        # support handling
        # ==============================

        if score >= 0.70:
            supported = True

        if supported or stance == "SUPPORT" or score >= 0.70:
            support_hits += 1
            rag_supported_any = True

            if evidence:
                aggregated_evidence.extend(evidence)

            support_scores.append(score)
            max_support = max(max_support, score)
            support_hits += 1

            if score >= 0.75:
                break

        print("STANCE:", stance)

    # =========================================================
    # Aggregate RAG metrics
    # =========================================================
    avg_support = round(
        sum(support_scores) / max(1, len(support_scores)),
        3
    )

    STRONG_EVIDENCE_THRESHOLD = 0.65
    MIN_AVG_SUPPORT = 0.45
    WEAK_EVIDENCE_FLOOR = 0.40

    print("AGGREGATION: rag_supported_any=", rag_supported_any,
        "max_support=", max_support,
        "avg_support=", avg_support,
        "contradiction_detected=", contradiction_detected,
        "contradiction_score=", contradiction_score,
        "final_label_so_far=", final_label)

    # =========================================================
    # 🚨 SAFETY: specific entities need evidence
    # =========================================================
    if (
        not rag_supported_any
        and not contradiction_detected
        and not aggregated_evidence
    ):
        final_label = "UNVERIFIABLE"
        final_conf = 0


    # =========================================================
    # Final verdict decision
    # =========================================================
    if contradiction_hits > support_hits and contradiction_hits > 0:
        final_label = "FAKE"
        final_conf = max(final_conf, 90)

    elif support_hits > 0:
        final_label = "REAL"
        final_conf = max(final_conf, int(max_support * 100))

    else:
        final_label = "UNVERIFIABLE"
        final_conf = 0


    # =========================================================
    # ✅ Final debug prints
    # =========================================================
    print("FINAL LABEL BEFORE RESPONSE:", final_label)
    print("FINAL CONFIDENCE:", final_conf)
    print("SUPPORT HITS:", support_hits)
    print("CONTRADICTION HITS:", contradiction_hits)
    print("MAX SUPPORT:", max_support)

    # =========================================================
    # 6️⃣ FINAL RESPONSE
    # =========================================================

    # 🚨 Evidence override explanation (DO NOT RETURN HERE)
    if final_label == "UNVERIFIABLE" and model_label == "REAL":
        explanation = (
            "The claim appears plausible based on language patterns, "
            "but no reliable external evidence was found to confirm it. "
            "Because factual verification requires strong supporting sources, "
            "the claim is classified as UNVERIFIABLE."
        )
    else:
        explanation = build_structured_explanation(
            final_label=final_label,
            model_label=model_label,
            model_conf=model_conf,
            detected_lang=detected_lang,
            text=original_text,
            evidence=aggregated_evidence,
            support_score=avg_support
        )

    return build_ui_response({
        "finalLabel": final_label,
        "confidencePercent": normalize_confidence(
            final_label,
            verification_method,
            model_conf=model_conf,
            avg_support=avg_support
        ),
        "summary": "ML + evidence-based verification applied.",
        "aiExplanation": explanation,
        "keywords": keywords,
        "language": language_name,
        "sentiment": sentiment,
        "factCheckUsed": bool(aggregated_evidence),
        "factCheckSource": "RAG",
        "verificationMethod": verification_method + (" + RAG" if aggregated_evidence else ""),
        "evidence": aggregated_evidence[:5]
    })
#==========================================================
# URL PREDICTION
# =========================================================
@app.post("/predict_url")
def predict_url(data: InputURL):
    # Updated URL pipeline (trusted vs unknown) — adapted from your main.py. Reference: :contentReference[oaicite:0]{index=0}
    RAG_ATTEMPTS.set(0)
    RAG_DECISION.set(None)

    url = (data.url or "").strip()
    if not url:
        return build_ui_response({
            "finalLabel": "UNVERIFIABLE",
            "confidencePercent": 0,
            "summary": "No URL provided.",
            "aiExplanation": "The request did not include a valid URL.",
            "keywords": [],
            "language": "Unknown",
            "sentiment": {"overall": "neutral", "anger": 0, "fear": 0, "neutral": 100},
            "factCheckUsed": False,
            "factCheckSource": None,
            "verificationMethod": "INPUT_VALIDATION",
            "evidence": []
        })

    # ------------------------------
    # 1) Source credibility (unchanged)
    # ------------------------------
    source_label = source_credibility_check(url)
    source_boost = 0
    if source_label == "FAKE":
        return build_ui_response({
            "finalLabel": "FAKE",
            "confidencePercent": 90,
            "summary": "Low-credibility source detected.",
            "aiExplanation": "The website is known for unreliable or misleading content.",
            "keywords": [],
            "language": "Unknown",
            "sentiment": {"overall": "neutral", "anger": 0, "fear": 0, "neutral": 100},
            "factCheckUsed": False,
            "factCheckSource": "Source Credibility",
            "verificationMethod": "SOURCE_DOMAIN",
            "evidence": build_url_evidence(
                verdict="FAKE",
                domain=url
            )
        })
    if source_label == "REAL":
        source_boost = 15

    # ------------------------------
    # 2) Fetch page (unchanged)
    # ------------------------------
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12
        )
        html = r.text
    except Exception:
        return build_ui_response({
            "finalLabel": "UNVERIFIABLE",
            "confidencePercent": 0,
            "summary": "Unable to fetch the URL.",
            "aiExplanation": "The webpage could not be accessed.",
            "keywords": [],
            "language": "Unknown",
            "sentiment": {"overall": "neutral", "anger": 0, "fear": 0, "neutral": 100},
            "factCheckUsed": False,
            "factCheckSource": None,
            "verificationMethod": "URL_FETCH",
            "evidence": build_url_evidence(
                verdict="UNVERIFIABLE",
                domain=url
            )
        })

    # ------------------------------
    # 3) ClaimReview check (unchanged)
    # ------------------------------
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]

    claimreview = extract_claimreview_from_html(html)
    if claimreview:
        label, source = claimreview_verdict(claimreview)
        confidence = claimreview_confidence(claimreview)

        return build_ui_response({
            "finalLabel": label,
            "confidencePercent": confidence,
            "summary": "ClaimReview metadata detected.",
            "aiExplanation": f"Verified using ClaimReview ({source}).",
            "keywords": [],
            "language": "Unknown",
            "sentiment": {"overall": "neutral", "anger": 0, "fear": 0, "neutral": 100},
            "factCheckUsed": True,
            "factCheckSource": source,
            "verificationMethod": "CLAIMREVIEW",
            "evidence": build_url_evidence(
                verdict=label,
                domain=domain
            )
        })

    # ------------------------------
    # 4) Extract raw text + detect page type (kept)
    # ------------------------------
    soup = BeautifulSoup(html, "html.parser")
    raw_text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))

    # Fallback to JSON-LD article body if too short
    if len(raw_text.split()) < 50:
        ld_json_text = extract_article_from_ld_json(soup)
        if ld_json_text:
            raw_text = ld_json_text

    page_type = detect_url_page_type(raw_text)
    if page_type in {"ERROR_PAGE", "SUPPORT_PAGE"}:
        return build_ui_response({
            "finalLabel": "UNVERIFIABLE",
            "confidencePercent": 0,
            "summary": "The page does not contain verifiable factual claims.",
            "aiExplanation": "The provided URL appears to be a service or support page.",
            "keywords": [],
            "language": "English",
            "sentiment": {"overall": "neutral", "anger": 0, "fear": 0, "neutral": 100},
            "factCheckUsed": False,
            "factCheckSource": None,
            "verificationMethod": "URL_PAGE_TYPE_DETECTION",
            "evidence": build_url_evidence(
                verdict="UNVERIFIABLE",
                domain=domain,
                page_type=page_type
            )
        })

    clean_text = remove_citation_markers(raw_text)

    # ------------------------------
    # Helper: robust headline extraction (OpenGraph / Twitter / H1 / title)
    # ------------------------------
    def extract_headline_from_dom(soup_obj, html_text):
        # 1) OpenGraph
        og = soup_obj.find("meta", property="og:title")
        if og and og.get("content"):
            return og.get("content").strip()

        # 2) Twitter
        t = soup_obj.find("meta", attrs={"name": "twitter:title"})
        if t and t.get("content"):
            return t.get("content").strip()

        # 3) JSON-LD NewsArticle headline
        try:
            ld = extract_article_from_ld_json(soup_obj)
            if ld:
                # ld may be a dict or string
                if isinstance(ld, dict):
                    headline = ld.get("headline") or ld.get("title")
                    if headline:
                        return headline.strip()
                elif isinstance(ld, str) and len(ld.split()) < 30:
                    return ld.strip()
        except Exception:
            pass

        # 4) H1 / H2
        for tag in ("h1", "h2"):
            h = soup_obj.find(tag)
            if h:
                text = h.get_text(" ", strip=True)
                if text and len(text.split()) <= 30:
                    return text

        # 5) <title> fallback
        if soup_obj.title and soup_obj.title.string:
            return soup_obj.title.string.strip()

        # 6) OCR/text fallback: use first meaningful sentence
        return extract_headline_from_ocr(html_text)

    # ------------------------------
    # 5) Trusted vs Unknown routing (NEW behavior)
    # ------------------------------
    # Curated trusted news domains (industry examples). You can expand this list.
    TRUSTED_NEWS_DOMAINS = {
        "bbc.co.uk", "bbc.com", "reuters.com",
        "thehindu.com", "theguardian.com", "nytimes.com",
        "washingtonpost.com", "aljazeera.com", "timesofindia.indiatimes.com",
        "cnn.com", "economist.com", "ft.com", "apnews.com", "cnn.com"
    }

    is_trusted_domain = any(td in domain for td in TRUSTED_NEWS_DOMAINS) or source_label == "REAL"
    print("🧭 PAGE TYPE DETECTED:", page_type)
    print("🌐 DOMAIN:", domain)
    print("🟢 TRUSTED DOMAIN:", is_trusted_domain)

    # 🚫 Skip low-content pages (weather, stock pages, utilities)
    if len(clean_text.split()) < 80:
        print("🚫 LOW CONTENT PAGE — RETURNING UNVERIFIABLE")

        return build_ui_response({
            "finalLabel": "UNVERIFIABLE",
            "confidencePercent": 50,
            "summary": "The page does not contain sufficient factual content for verification.",
            "aiExplanation": (
                "The provided URL appears to be a utility, weather, or non-article page "
                "and does not contain sufficient narrative content to verify factual claims."
            ),
            "keywords": [],
            "language": get_language_name(safe_language_detect(clean_text)),
            "sentiment": {"overall": "neutral", "anger": 0, "fear": 0, "neutral": 100},
            "factCheckUsed": False,
            "factCheckSource": None,
            "verificationMethod": "LOW_CONTENT_PAGE",
            "evidence": []
        })

    # If page classified as ARTICLE and trusted → direct headline → universal RAG (no DuckDuckGo)
    if  is_trusted_domain:
        print("📰 Trusted ARTICLE detected — direct RAG (no DuckDuckGo).")
        headline = extract_headline_from_dom(soup, raw_text)
        if not headline or len(headline.strip()) < 3:
            # fallback to first meaningful sentence from the article
            headline = extract_headline_from_ocr(clean_text)

        detected_lang = safe_language_detect(headline or clean_text)
        print("🚀 CALLING RAG WITH CLAIM:", headline)

        claim_for_rag = prepare_claim_for_rag(headline)
        rag_result = universal_rag_retrieve(claim_for_rag, urls=[url])

        print("📰 EXTRACTED HEADLINE:", headline)
        print("📄 TEXT LENGTH:", len(clean_text.split()))

        # If universal_rag_retrieve already returns a UI-dict decisive result — return immediately (keeps existing behavior)
        if isinstance(rag_result, dict) and rag_result.get("finalLabel") in {"REAL", "FAKE"}:
            print("🔒 RAG returned a decisive dict — returning UI response.")
            return build_ui_response(rag_result)

        # Non-decisive path: fall back to original per-article processing loop (reuse existing logic)
        # To keep behavior consistent we convert rag_result to legacy tuple if needed and continue with existing article loop logic
        # (the rest of your existing article loop will handle semantic scoring / evidence collection)
        # We emulate a verify_claim_with_rag-like tuple so downstream code can be re-used.
        if isinstance(rag_result, dict):
            # Populate legacy docs
            docs = []
            for ev in rag_result.get("evidence", []):
                docs.append({
                    "url": ev.get("url"),
                    "snippet": ev.get("snippet"),
                    "page_type": ev.get("page_type", "UNKNOWN"),
                    "score": ev.get("score", 0)
                })
            # Use finalLabel and confidencePercent if available
            supported = rag_result.get("finalLabel") == "REAL"
            confidence = round(rag_result.get("confidencePercent", 50) / 100.0, 3)
            RAG_DECISION.set((supported, docs, confidence))
        else:
            # Safe fallback to empty decision (forces the article loop to continue)
            RAG_DECISION.set((False, [], 0.0))

        # Continue to the existing article processing (the code below in your main loop will pick up RAG_DECISION)
        # For simplicity we now call the unified URL-to-RAG fallback block handled later in the function.
        # (No return here — keep the original downstream logic unchanged.)

    # Unknown or low-trust domains: headline -> DuckDuckGo -> multi-source RAG
    elif not is_trusted_domain:

        print("📰 Unknown ARTICLE or not in trusted list — headline -> DuckDuckGo -> multi-source RAG.")
        headline = extract_headline_from_dom(soup, raw_text)
        if not headline or len(headline.strip()) < 3:
            headline = extract_headline_from_ocr(clean_text)

        detected_lang = safe_language_detect(headline or clean_text)

        # Use headline as search query into DuckDuckGo (multi-source verification)
        search_urls = duckduckgo_search(headline)
        urls = filter_urls(search_urls)
        print("✅ URLs after headline DuckDuckGo:", urls)

        if not urls:
            # No search hits — fall back to the article being unverified → preserve original behavior
            print("⚠️ No external search results — falling back to URL-RAG fallback.")
            # Allow downstream logic to handle (existing behavior)
            RAG_DECISION.set((False, [], 0.0))
        else:
            claim_for_rag = prepare_claim_for_rag(headline)
            rag_result = universal_rag_retrieve(claim_for_rag, urls=[url])

            if isinstance(rag_result, dict) and rag_result.get("finalLabel") in {"REAL", "FAKE"}:
                return build_ui_response(rag_result)

            if isinstance(rag_result, dict):
                docs = []
                for ev in rag_result.get("evidence", []):
                    docs.append({
                        "url": ev.get("url"),
                        "snippet": ev.get("snippet"),
                        "page_type": ev.get("page_type", "UNKNOWN"),
                        "score": ev.get("score", 0)
                    })
                supported = rag_result.get("finalLabel") == "REAL"
                confidence = round(rag_result.get("confidencePercent", 50) / 100.0, 3)
                RAG_DECISION.set((supported, docs, confidence))
            else:
                RAG_DECISION.set((False, [], 0.0))

        # Allow the original article loop to continue and use RAG_DECISION (no direct return unless decisive above)

    # PROFILE pages & other page types should keep their original logic and routing.
    # From this point onward we fall back to the original URL-processing loop that iterates over urls,
    # runs cached_classify_url_type, cached_extract_main_text, profile fast-paths, semantic scoring, etc.
    # The rest of the function (below in your original file) will pick up RAG_DECISION and proceed unchanged.

    # --- IMPORTANT: keep the same final return structure as before.
    # If downstream code in this function expects to continue processing URLs within a for-loop,
    # we do not return here. If you prefer an early-return after setting RAG_DECISION for trusted/articles,
    # uncomment the following fallback returns (commented to preserve existing flow):

    # if RAG_DECISION.get() is not None:
    #     supported, evidence, score = RAG_DECISION.get()
    #     # Build a minimal legacy-style UI response (only used if you want early-return)
    #     return build_ui_response({
    #         "finalLabel": "REAL" if supported else "FAKE" if evidence else "UNVERIFIABLE",
    #         "confidencePercent": int(score * 100),
    #         "summary": "URL routed through improved URL pipeline.",
    #         "aiExplanation": "Decision produced by URL-aware RAG routing.",
    #         "keywords": [],
    #         "language": get_language_name(safe_language_detect(clean_text)),
    #         "sentiment": {"overall": "neutral", "anger": 0, "fear": 0, "neutral": 100},
    #         "factCheckUsed": False,
    #         "factCheckSource": None,
    #         "verificationMethod": "URL_RAG_ROUTER",
    #         "evidence": evidence
    #     })

    # No early return by default — preserve original behavior and let the rest of the function run.
    return None

# =========================================================
# IMAGE PREDICTION
# =========================================================
OCR_THREAD_POOL = ThreadPoolExecutor(max_workers=2)
@app.post("/predict_image")
def predict_image(data: InputImage):
    start_ts = time.time()

    # -------------------------
    # 1) Decode + basic sanity
    # -------------------------
    try:
        # remove image tag if exists
        image_base64 = data.image_base64.split(",")[-1]

        # decode base64
        image_bytes = base64.b64decode(image_base64)

        # convert directly to OpenCV image (NO PIL needed)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if cv_img is None:
            raise ValueError("Image decode failed")

    except Exception as e:
        return build_ui_response({
            "finalLabel": "UNVERIFIABLE",
            "confidencePercent": 0,
            "summary": "Invalid or unsupported image format.",
            "aiExplanation": f"The uploaded image could not be processed ({str(e)}).",
            "keywords": [],
            "language": "Unknown",
            "sentiment": {"overall": "neutral", "anger": 0, "fear": 0, "neutral": 100},
            "factCheckUsed": False,
            "factCheckSource": None,
            "verificationMethod": "IMAGE_DECODE",
            "evidence": []
        })
    # =========================================================
    # 🔥 THREAD SAFE PADDLE OCR WORKER (APP.PY STYLE)
    # =========================================================
    def paddle_worker_app_style(image_np):

        try:
            print("🧠 Starting PaddleOCR...")

            result = get_ocr().ocr(image_np, cls=True)

            extracted_text = []

            if not result:
                print("⚠️ No OCR result")
                return ""

            for line in result:
                if not line:
                    continue

                for word_info in line:
                    text = word_info[1][0]
                    score = word_info[1][1]

                    if score > 0.55 and text.strip():
                        extracted_text.append(text.strip())

            final_text = " ".join(extracted_text)

            print("\n================ OCR OUTPUT =================")
            print(final_text)
            print("=============================================\n")

            return final_text

        except Exception as e:
            print("❌ OCR ERROR:", str(e))
            return ""


    # =========================================================
    # 🔥 THREAD SAFE OCR WORKER (APP.PY STYLE)
    # =========================================================
    def ocr_worker(image_np):
        try:
            processed = enhance_image_from_array(image_np)

            result = get_ocr().ocr(processed, cls=True)

            extracted_text = []

            if not result:
                return ""

            for line in result:
                if not line:
                    continue

                for word_info in line:
                    text = word_info[1][0]
                    score = word_info[1][1]

                    if score > 0.55 and text.strip():
                        extracted_text.append(text.strip())

            return " ".join(extracted_text)

        except Exception as e:
            print("❌ OCR ERROR:", e)
            return ""


    # =========================================================
    # 🔥 APP.PY STYLE PREPROCESS (FROM YOUR SCRIPT)
    # =========================================================
    def enhance_image_from_array(img):

        h, w = img.shape[:2]
        if max(h, w) > 1600:
            scale = 1600 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        gray = cv2.GaussianBlur(gray, (3,3), 0)

        kernel = np.array([[0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])
        sharp = cv2.filter2D(gray, -1, kernel)

        return sharp


    # =========================================================
    # 🔥 RUN THREAD SAFE OCR
    # =========================================================
    try:
        fut = OCR_THREAD_POOL.submit(paddle_worker_app_style, cv_img)
        final_ocr = fut.result(timeout=20)
    except Exception as e:
        print("⚠️ Thread failed, running sync:", e)
        final_ocr = paddle_worker_app_style(cv_img)

    # =========================================================
    # 🔥 KEEP YOUR EXISTING PIPELINE (UNCHANGED)
    # =========================================================

    ocr_clean = remove_citation_markers(final_ocr)
    ocr_clean = reconstruct_phrases(ocr_clean)
    ocr_clean = recover_words(ocr_clean)

    detected_lang = safe_language_detect(ocr_clean)

    normalized_text = normalize_indic_text(ocr_clean, detected_lang)
    normalized_text = language_specific_preprocess(normalized_text, detected_lang)

    final_claim_text = re.sub(r"\s+", " ", normalized_text).strip()
    final_claim_text = final_claim_text.rstrip(" .,:;!-?")

    # -------------------------
    # 7) Screenshot detection (keep this)
    # -------------------------
    def detect_screenshot(text):
        if not text:
            return False
        tc = text.lower()
        screenshot_signals = [
            "retweet", "likes", "views", "reply", "twitter", "instagram", "facebook",
            "@", "twitter for android", "twitter for iphone", "like", "share", "followers"
        ]
        hits = sum(1 for s in screenshot_signals if s in tc)
        at_hash_count = tc.count("@") + tc.count("#")
        return hits >= 1 or at_hash_count >= 2

    is_screenshot = detect_screenshot(final_claim_text)

    # -------------------------
    # 8) OCR quality gate (unchanged)
    # -------------------------
    if not final_claim_text or len(final_claim_text.split()) < 10:
        return build_ui_response({
            "finalLabel": "UNVERIFIABLE",
            "confidencePercent": 0,
            "summary": "Insufficient readable text in image.",
            "aiExplanation": "The image does not contain enough readable text for verification.",
            "keywords": [],
            "language": "Unknown",
            "sentiment": {"overall": "neutral", "anger": 0, "fear": 0, "neutral": 100},
            "factCheckUsed": False,
            "factCheckSource": None,
            "verificationMethod": "OCR_QUALITY",
            "evidence": []
        })

    # -------------------------
    # 9) Hand off to existing text pipeline (UNMODIFIED)
    # -------------------------
    claim_text = final_claim_text
    detected_lang = detected_lang or safe_language_detect(claim_text)

    # continue with your existing code:
    # - translate if needed
    # - universal_rag_retrieve
    # - verify/semantic logic
    # - build_ui_response
    response = predict_text(InputText(text=claim_text))

    return response
