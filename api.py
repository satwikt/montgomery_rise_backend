"""
Montgomery RISE — Unified API Server  (v4.0.0)
==============================================
Combines the parcel-scoring pipeline (rise_selector_v3) with the RAG
chatbot (ChromaDB + sentence-transformers + Groq) in a single FastAPI app.

Endpoints
---------
Scoring pipeline
  GET  /                            Health check (pipeline + RAG status)
  GET  /parcels                     List hero parcels (metadata only)
  GET  /parcels/{id}/score          Full scoring pipeline  (A | B | C)
  POST /parcels/custom              Full pipeline for arbitrary lat/lon

RAG chatbot
  GET  /parcels/{id}/answer         Knowledge-base AI summary for a hero parcel
  POST /chat/ask                    Free-form RAG question (JSON response)
  POST /chat/stream                 Streaming RAG answer  (Server-Sent Events)

Environment variables
---------------------
  GEMINI_API_KEY   Gemini 2.5 Pro for scoring-pipeline recommendations
  GROQ_API_KEY     Groq LLaMA-3.3-70B for RAG chatbot answers

Quick start
-----------
  pip install -r requirements.txt
  uvicorn api:app --reload --port 8000
  # ChromaDB is populated automatically on first startup — no manual step needed.

Railway deploy
--------------
  Start command: uvicorn api:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path as FilePath
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── optional dotenv (not installed on some Railway images) ────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # rely on shell / Railway environment variables

# ── scoring pipeline ──────────────────────────────────────────────────────────
from rise_selector_v3 import (
    ANCHORS,
    GEMINI_API_KEY,
    HERO_PARCELS,
    analyse_with_gemini,
    calculate_distance,
    compute_score,
    get_foot_traffic,
    get_grant_data,
    merge_grants,
)

# ── RAG chatbot ───────────────────────────────────────────────────────────────
from rise_rag.app.chatbot import RiseChatbot
from rise_rag.app.ingestion import Chunk, load_all_chunks
from rise_rag.app.llm import get_llm_info, is_llm_available

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rise.api")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VALID_PARCEL_IDS: frozenset[str] = frozenset({"A", "B", "C"})

# Index map so hero parcel lookups are O(1) and self-documenting.
PARCEL_INDEX: dict[str, int] = {"A": 0, "B": 1, "C": 2}

# Canned questions used by GET /parcels/{id}/answer.
_RAG_SUMMARY_QUESTIONS: dict[str, str] = {
    "A": (
        "Give a complete summary of Parcel A: all 8 RISE dimension scores, "
        "the top 3 land-reuse recommendations, open grant windows with "
        "deadlines and coverage amounts, and the community health context."
    ),
    "B": (
        "Give a complete summary of Parcel B: all 8 RISE dimension scores, "
        "the top 3 land-reuse recommendations, open grant windows, and "
        "the Maxwell AFB / IX Hub workforce-development context."
    ),
    "C": (
        "Give a complete summary of Parcel C: all 8 RISE dimension scores, "
        "the top 3 land-reuse recommendations, both open grant windows with "
        "deadlines and coverage amounts, and the food-desert / health-equity "
        "context."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Application lifecycle
# ─────────────────────────────────────────────────────────────────────────────

# Module-level reference so endpoint handlers can reach the chatbot.
_chatbot: RiseChatbot | None = None


def _auto_ingest(chatbot: RiseChatbot) -> None:
    """
    Ingest the knowledge base into ChromaDB if the collection is empty.

    Called once at startup so judges never need to run a manual script —
    hitting any endpoint is enough.  If the collection already has chunks
    (e.g. a persisted Railway volume) this is a no-op, so restarts are fast.
    """
    from rise_rag.app.config import DATA_DIR

    if chatbot.is_ready():
        logger.info(
            "Knowledge base already populated (%d chunks) — skipping ingest.",
            chatbot.knowledge_base_size(),
        )
        return

    logger.info("Knowledge base is empty — auto-ingesting from %s …", DATA_DIR)

    txt_files = sorted(DATA_DIR.glob("*.txt"))
    if not txt_files:
        logger.error(
            "No .txt files found in %s — RAG will not work until data files "
            "are present.",
            DATA_DIR,
        )
        return

    try:
        chunks = load_all_chunks(DATA_DIR)
        ingested = chatbot._store.upsert_chunks(chunks)  # noqa: SLF001
        logger.info(
            "Auto-ingest complete: %d chunks from %d files loaded into ChromaDB.",
            ingested,
            len(txt_files),
        )
    except Exception:
        logger.exception("Auto-ingest failed — RAG endpoints may return fallback responses.")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Initialise long-lived resources and auto-ingest the knowledge base."""
    global _chatbot  # noqa: PLW0603

    logger.info("Montgomery RISE v4.0.0 — starting up")

    try:
        _chatbot = RiseChatbot()

        # Auto-ingest on startup so no manual step is ever needed.
        # Judges only interact with the UI — this runs silently in the background.
        _auto_ingest(_chatbot)

        logger.info(
            "RAG chatbot ready — %d chunks in knowledge base.",
            _chatbot.knowledge_base_size(),
        )

        if not is_llm_available():
            logger.warning(
                "GROQ_API_KEY is not configured. "
                "RAG endpoints will return fallback context only. "
                "Get a free key at https://console.groq.com"
            )
    except Exception:
        logger.exception("RAG chatbot failed to initialise — RAG endpoints disabled.")
        _chatbot = None

    yield

    logger.info("Montgomery RISE v4.0.0 — shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Montgomery RISE API",
    description=(
        "Revitalization Intelligence & Smart Empowerment — "
        "parcel scoring pipeline + RAG chatbot for Montgomery, Alabama. "
        "v4.0.0 unified server."
    ),
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────


class CustomParcelRequest(BaseModel):
    """Request body for POST /parcels/custom."""

    lat: float = Field(..., examples=[32.3789], description="Latitude (WGS84)")
    lon: float = Field(..., examples=[-86.3109], description="Longitude (WGS84)")
    label: str = Field("Custom Parcel", examples=["My Parcel"])
    story: str = Field("Custom", examples=["Economic Urgency"])
    address: str = Field(
        "Montgomery, Alabama", examples=["123 Main St, Montgomery AL"]
    )
    parcel_id: str = Field("CUSTOM-001", examples=["CUSTOM-001"])
    acres: float = Field(1.0, examples=[2.5], description="Parcel size in acres")
    owner: str = Field("City of Montgomery", examples=["City of Montgomery"])
    zone_context: str = Field(
        "heritage",
        examples=["heritage"],
        description="One of: heritage | ix_hub | food_desert",
    )
    health_flags: dict[str, Any] = Field(
        default_factory=dict,
        examples=[{"food_insecurity_pct": 35, "asthma_rate_multiplier": 1.8}],
    )
    grant_flags: list[dict[str, Any]] = Field(
        default_factory=list,
        examples=[
            [
                {
                    "name": "USDA Rural Economic Development Q3",
                    "status": "open",
                    "days_remaining": 23,
                    "eligibility_pct": 70,
                    "match": "20%",
                }
            ]
        ],
    )


class WeightsRequest(BaseModel):
    """Request body for POST /config/{city_id}/weights."""

    weights: dict[str, float] = Field(
        ...,
        description="Score dimension weights. Keys: heritage, industrial, activity, proximity, economic, vacancy, flood, 311. Values must sum to 1.0.",
        examples=[{
            "heritage": 0.25, "industrial": 0.20, "activity": 0.15,
            "proximity": 0.10, "economic": 0.10, "vacancy": 0.10,
            "flood": 0.05, "311": 0.05,
        }],
    )


class AskRequest(BaseModel):
    """Request body for POST /chat/ask and POST /chat/stream."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description=(
            "Question about RISE parcels, scoring, grants, or community context."
        ),
        examples=["What is the most urgent grant window right now?"],
    )
    parcel_filter: Optional[str] = Field(
        default=None,
        description="Restrict search to one parcel: 'A', 'B', or 'C'.",
        examples=["C"],
    )


class SourceModel(BaseModel):
    """A single knowledge-base source cited in an RAG answer."""

    document_title: str
    parcel_id: str
    topic: str
    source_file: str


class AskResponse(BaseModel):
    """Response body for POST /chat/ask."""

    question: str
    answer: str
    sources: list[SourceModel]
    num_chunks_retrieved: int
    used_fallback: bool
    parcel_filter: Optional[str]
    generated_at: str


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers — scoring pipeline
# ─────────────────────────────────────────────────────────────────────────────


def _nearest_anchor(lat: float, lon: float) -> tuple[str, float]:
    """Return ``(name, distance_miles)`` for the closest anchor point."""
    best_name = "Unknown"
    best_dist = float("inf")
    for anchor in ANCHORS:
        dist = calculate_distance(lat, lon, anchor["lat"], anchor["lon"])
        if dist < best_dist:
            best_dist, best_name = dist, anchor["name"]
    return best_name, round(best_dist, 3)


def _run_pipeline(parcel: dict[str, Any]) -> dict[str, Any]:
    """
    Execute the full RISE scoring pipeline for *parcel* and return a
    serialisable response payload.

    Steps
    -----
    1. ArcGIS foot-traffic query (1-mile radius)
    2. 8-dimension scorer
    3. Gemini 2.5 Pro recommendation analysis (or mock fallback)
    """
    t_start = time.perf_counter()

    # Deep-copy so we never mutate the in-memory HERO_PARCELS list.
    p = copy.deepcopy(parcel)

    # Merge static grant flags with live grants.gov results
    live_grant_data = get_grant_data()
    p["grant_flags"] = merge_grants(p.get("grant_flags", []), live_grant_data.get("grants", []))

    foot_traffic: dict[str, Any] = get_foot_traffic(p)
    scores: dict[str, Any] = compute_score(p, foot_traffic)
    p["scores"] = scores
    ai_analysis: dict[str, Any] = analyse_with_gemini(p, foot_traffic)

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    # Reshape sub-objects to match the shape demo_3.html expects.
    flood_risk = {
        "score": scores["flood"],
        "zone": scores["flood_zone"],
        "label": scores["flood_label"],
    }
    distress_311 = {
        "score": scores["distress"],
        "density_per_sq_mi": scores["destress_density"],
        "total_calls_90days": scores["destress_calls_90days"],
        "top_complaints": scores["destress_top_complaints"],
        "label": scores["destress_label"],
    }
    scores_out = {**scores, "urgency": ai_analysis.get("urgency_flag", "medium")}

    gemini_live = GEMINI_API_KEY not in ("", "YOUR_GEMINI_KEY")

    return {
        # Parcel identity
        "label": p["label"],
        "story": p["story"],
        "address": p["address"],
        "parcel_id": p["parcel_id"],
        "acres": p.get("acres"),
        "nearest_anchor": p["nearest_anchor"],
        "min_dist_miles": p["min_dist"],
        "zone_context": p.get("zone_context", ""),
        "owner": p.get("owner", "City of Montgomery"),
        # Scoring
        "scores": scores_out,
        # Signal sub-panels
        "flood_risk": flood_risk,
        "distress_311": distress_311,
        "foot_traffic": {
            "score": foot_traffic["score"],
            "location_count": foot_traffic["location_count"],
            "total_visits": foot_traffic["total_visits"],
            "nearest_name": foot_traffic["nearest_name"],
            "nearest_dist_mi": foot_traffic["nearest_dist_mi"],
            "source": foot_traffic["source"],
            "top_locations": foot_traffic["top_locations"][:5],
        },
        # Grant + health panels
        "grant_flags": p.get("grant_flags", []),
        "health_flags": p.get("health_flags", {}),
        # AI analysis
        "ai_analysis": {
            **ai_analysis,
            "ai_source": "gemini-2.5-pro" if gemini_live else "mock",
        },
        # Metadata
        "meta": {
            "generated_at": _utcnow(),
            "pipeline": "RISE v4 — ArcGIS + Gemini 2.5 Pro + Groq RAG",
            "pipeline_ms": elapsed_ms,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Real-time RAG ingestion — called after every scoring pipeline run
# ─────────────────────────────────────────────────────────────────────────────


def _build_live_document(pid: str, result: dict[str, Any]) -> str:
    """
    Convert a live pipeline result into a plain-text knowledge document
    suitable for embedding and retrieval.

    The document mirrors the structure of the static ``.txt`` knowledge-base
    files so the retriever treats it exactly like any other chunk.  Because
    it is upserted with the same ``chunk_id`` every time, it overwrites the
    previous live snapshot — ChromaDB never accumulates stale duplicates.

    Parameters
    ----------
    pid:
        Parcel letter — ``'A'``, ``'B'``, or ``'C'``.
    result:
        The serialisable dict returned by :func:`_run_pipeline`.
    """
    scores      = result.get("scores", {})
    ft          = result.get("foot_traffic", {})
    flood       = result.get("flood_risk", {})
    d311        = result.get("distress_311", {})
    ai          = result.get("ai_analysis", {})
    grants      = result.get("grant_flags", [])
    health      = result.get("health_flags", {})
    recs        = ai.get("recommendations", [])
    generated   = result.get("meta", {}).get("generated_at", _utcnow())

    # ── grant summary ─────────────────────────────────────────────────────────
    grant_lines: list[str] = []
    for g in grants:
        status = g.get("status", "unknown").upper()
        days   = g.get("days_remaining", "?")
        pct    = g.get("eligibility_pct", "?")
        match  = g.get("match", "?")
        grant_lines.append(
            f"  - {g.get('name', 'Unknown grant')}: {status}, "
            f"{days} days remaining, covers {pct}% of costs, {match} match required."
        )
    grants_text = "\n".join(grant_lines) if grant_lines else "  No open grants found."

    # ── recommendation summary ────────────────────────────────────────────────
    rec_lines: list[str] = []
    for r in recs[:3]:
        rec_lines.append(
            f"  {r.get('rank', '?')}. {r.get('name', 'Unknown')} "
            f"(fit score {r.get('fit_score', '?')}/100, "
            f"{r.get('cost_tier', '?')}): {r.get('explanation', '')}"
        )
    recs_text = "\n".join(rec_lines) if rec_lines else "  No recommendations generated."

    # ── top foot-traffic locations ────────────────────────────────────────────
    ft_locations = ft.get("top_locations", [])
    ft_lines = [
        f"  - {loc.get('name', '?')}: {loc.get('visits', '?'):,} visits "
        f"({loc.get('dist_mi', '?')} miles)"
        for loc in ft_locations[:5]
    ]
    ft_text = "\n".join(ft_lines) if ft_lines else "  No foot-traffic locations found."

    # ── health flags ──────────────────────────────────────────────────────────
    health_lines = [f"  - {k}: {v}" for k, v in health.items()]
    health_text  = "\n".join(health_lines) if health_lines else "  No health flags recorded."

    return f"""\
DOCUMENT: Parcel {pid} — Live Real-Time Score ({generated})
PARCEL_ID: {pid}
TOPIC: live_scores

This document contains the LATEST REAL-TIME scores for Parcel {pid}, fetched
live from ArcGIS, FEMA flood hazard, Montgomery 311, and Gemini AI at
{generated}. Use this document in preference to any static score documents
when answering questions about current scores, recommendations, or foot traffic.

PARCEL IDENTITY
  Label          : {result.get("label", "?")}
  Address        : {result.get("address", "?")}
  Parcel ID      : {result.get("parcel_id", "?")}
  Acres          : {result.get("acres", "?")}
  Zone context   : {result.get("zone_context", "?")}
  Nearest anchor : {result.get("nearest_anchor", "?")} ({result.get("min_dist_miles", "?")} miles)
  Owner          : {result.get("owner", "?")}

RISE SCORES (live — fetched {generated})
  Final RISE score : {scores.get("final", "?")} / 100
  Urgency flag     : {scores.get("urgency", "?")}
  Heritage         : {scores.get("heritage", "?")} / 100
  Industrial       : {scores.get("industrial", "?")} / 100
  Activity         : {scores.get("activity", "?")} / 100
  Proximity        : {scores.get("proximity", "?")} / 100
  Economic         : {scores.get("economic", "?")} / 100
  Vacancy          : {scores.get("vacancy", "?")} / 100
  Flood risk       : {scores.get("flood", "?")} / 5  (zone: {flood.get("zone", "?")} — {flood.get("label", "?")})
  311 distress     : {scores.get("distress", "?")} / 10 — {d311.get("label", "?")}
    Density        : {d311.get("density_per_sq_mi", "?")} calls/sq mi (90-day window)
    Total calls    : {d311.get("total_calls_90days", "?")}
    Top complaints : {", ".join(str(c.get("type", c)) if isinstance(c, dict) else str(c) for c in d311.get("top_complaints", [])) or "none"}

FOOT TRAFFIC (live from ArcGIS Most Visited Locations)
  Score           : {ft.get("score", "?")} / 100
  Locations found : {ft.get("location_count", "?")} within 1 mile
  Total visits    : {ft.get("total_visits", "?"):,} (weighted)
  Nearest venue   : {ft.get("nearest_name", "?")} ({ft.get("nearest_dist_mi", "?")} miles)
  Source          : {ft.get("source", "?")}
  Top locations:
{ft_text}

AI RECOMMENDATIONS (generated by {ai.get("ai_source", "Gemini")} at {generated})
  One-line summary : {ai.get("one_line_summary", "?")}
{recs_text}

OPEN GRANT WINDOWS
{grants_text}

COMMUNITY HEALTH FLAGS
{health_text}
"""


def _upsert_live_scores(pid: str, result: dict[str, Any]) -> None:
    """
    Build a live-score document from *result* and upsert it into ChromaDB.

    Runs in a daemon background thread so the HTTP response is never delayed.
    The chunk ID is deterministic (``live_scores_{pid}__000``) so each call
    overwrites the previous snapshot — no stale duplicates accumulate.

    Parameters
    ----------
    pid:
        Parcel letter — ``'A'``, ``'B'``, or ``'C'``.
    result:
        The serialisable dict returned by :func:`_run_pipeline`.
    """
    if _chatbot is None:
        return  # RAG not available — nothing to upsert into

    def _worker() -> None:
        try:
            document_text = _build_live_document(pid, result)
            generated     = result.get("meta", {}).get("generated_at", _utcnow())

            chunk = Chunk(
                chunk_id       = f"live_scores_{pid}__000",
                text           = document_text,
                source_file    = f"live_scores_parcel_{pid.lower()}",
                document_title = f"Parcel {pid} — Live Real-Time Score ({generated})",
                parcel_id      = pid,
                topic          = "live_scores",
                metadata       = {
                    "source_file"    : f"live_scores_parcel_{pid.lower()}",
                    "document_title" : f"Parcel {pid} — Live Real-Time Score ({generated})",
                    "parcel_id"      : pid,
                    "topic"          : "live_scores",
                    "block_index"    : 0,
                    "generated_at"   : generated,
                },
            )

            _chatbot._store.upsert_chunks([chunk])  # noqa: SLF001
            logger.info(
                "Live scores for Parcel %s upserted into ChromaDB (%s).",
                pid,
                generated,
            )
        except Exception:
            # Never let a background upsert crash the main thread.
            logger.exception(
                "Background RAG upsert failed for Parcel %s — "
                "RAG will use the previous snapshot.",
                pid,
            )

    thread = threading.Thread(target=_worker, daemon=True, name=f"rag-upsert-{pid}")
    thread.start()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers — RAG chatbot
# ─────────────────────────────────────────────────────────────────────────────


def _require_rag() -> RiseChatbot:
    """
    Return the global chatbot instance or raise an appropriate HTTP error.

    Raises
    ------
    HTTPException 503
        If the chatbot is not initialised or the knowledge base is empty.
    """
    if _chatbot is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG chatbot is not available. "
                "Check server logs for initialisation errors."
            ),
        )
    if not _chatbot.is_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG knowledge base is empty. "
                "Run:  python rise_rag/scripts/ingest.py"
            ),
        )
    return _chatbot


def _parse_parcel_filter(raw: Optional[str]) -> Optional[str]:
    """
    Normalise and validate an optional parcel-filter value.

    Returns the upper-cased single-character ID or ``None``.

    Raises
    ------
    HTTPException 422
        If a value is provided but is not one of A / B / C.
    """
    if raw is None:
        return None
    normalised = raw.upper().strip()
    if normalised not in VALID_PARCEL_IDS:
        raise HTTPException(
            status_code=422,
            detail=f"parcel_filter must be one of {sorted(VALID_PARCEL_IDS)}.",
        )
    return normalised


def _build_sources(raw: list[dict[str, Any]]) -> list[SourceModel]:
    """Convert raw ChromaDB metadata dicts into ``SourceModel`` objects."""
    return [
        SourceModel(
            document_title=s.get("document_title", "Unknown"),
            parcel_id=s.get("parcel_id", "general"),
            topic=s.get("topic", "general"),
            source_file=s.get("source_file", "unknown"),
        )
        for s in raw
    ]


def _utcnow() -> str:
    """Return the current UTC time as an ISO-8601 string with timezone."""
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Score weights — per-city persistence
# ─────────────────────────────────────────────────────────────────────────────

_WEIGHTS_FILE = FilePath(__file__).parent / "city_weights.json"

_DEFAULT_WEIGHTS: dict[str, float] = {
    "heritage":   0.25,
    "industrial": 0.20,
    "activity":   0.15,
    "proximity":  0.10,
    "economic":   0.10,
    "vacancy":    0.10,
    "flood":      0.05,
    "311":        0.05,
}


def _load_weights(city_id: str) -> dict[str, float]:
    """Return saved weights for *city_id*, falling back to defaults."""
    if _WEIGHTS_FILE.exists():
        try:
            data = json.loads(_WEIGHTS_FILE.read_text(encoding="utf-8"))
            if city_id in data:
                return data[city_id]["weights"]
        except Exception:
            pass
    return dict(_DEFAULT_WEIGHTS)


def _save_weights(city_id: str, weights: dict[str, float]) -> None:
    """Persist *weights* for *city_id* to disk."""
    data: dict[str, Any] = {}
    if _WEIGHTS_FILE.exists():
        try:
            data = json.loads(_WEIGHTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    data[city_id] = {
        "weights": weights,
        "updated_at": _utcnow(),
    }
    _WEIGHTS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# ArcGIS parcel query — city-owned vacant lots
# ─────────────────────────────────────────────────────────────────────────────

import requests as _requests

_ARCGIS_VIOLATIONS_URL = (
    "https://services7.arcgis.com/xNUwUjOJqYE54USz/arcgis/rest/services/"
    "Code_Violations/FeatureServer/0/query"
)

_ARCGIS_PARCEL_URL = (
    "https://services7.arcgis.com/xNUwUjOJqYE54USz/arcgis/rest/services/"
    "SURPLUS_CITY_PROPERTIES_polygon/FeatureServer/0/query"
)


# Maps ArcGIS CaseType values to the frontend 311 distress signal categories.
_CASE_TYPE_CATEGORY: dict[str, str] = {
    "NUISANCE":             "Illegal Dumping",
    "OPEN VACANT":          "Vacant Lot",
    "DEMOLITION":           "Abandoned Building",
    "REPAIR":               "Abandoned Building",
    "GENERIC":              "Abandoned Building",
    "PARKING ON FRONT LAWN": "Illegal Dumping",
}

# All possible frontend categories — used to ensure every parcel has a full set.
_ALL_SIGNAL_CATEGORIES = [
    "Overgrown Vegetation",
    "Illegal Dumping",
    "Abandoned Building",
    "Drug Activity",
    "Vacant Lot",
    "Noise Complaint",
    "Graffiti",
]


def _casetype_to_category(case_type: str | None) -> str:
    """Map an ArcGIS CaseType to a frontend display category."""
    return _CASE_TYPE_CATEGORY.get((case_type or "").upper(), "Abandoned Building")


def _fetch_violations_bulk(parcel_ids: list[str]) -> dict[str, dict]:
    """
    Fetch code violations for all parcel IDs in one query using an IN clause.

    Returns a dict keyed by parcel_id with violation summary including
    total, open count, and per-category counts.
    """
    if not parcel_ids:
        return {}

    ids_sql = ", ".join(f"'{pid}'" for pid in parcel_ids)
    params = {
        "where": f"ParcelNo IN ({ids_sql})",
        "outFields": "ParcelNo,CaseType,CaseStatus",
        "f": "json",
        "resultRecordCount": 5000,
    }
    try:
        resp = _requests.get(_ARCGIS_VIOLATIONS_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Bulk violations query failed: %s", exc)
        return {}

    # Build per-parcel summary.
    summaries: dict[str, dict] = {}
    for feat in data.get("features", []):
        a = feat.get("attributes", {})
        pid = a.get("ParcelNo")
        if not pid:
            continue
        if pid not in summaries:
            summaries[pid] = {
                "violation_count": 0,
                "open_violations": 0,
                "signal_categories": {cat: 0 for cat in _ALL_SIGNAL_CATEGORIES},
            }
        summaries[pid]["violation_count"] += 1
        if (a.get("CaseStatus") or "").upper() == "OPEN":
            summaries[pid]["open_violations"] += 1
        category = _casetype_to_category(a.get("CaseType"))
        summaries[pid]["signal_categories"][category] += 1

    return summaries


def _fetch_vacant_city_parcels() -> list[dict[str, Any]]:
    """
    Query the Montgomery ArcGIS surplus city properties layer.

    Filters: DISPLAY = 'YES' (publicly listed lots only — 79 parcels).
    Also joins code violation counts (within exact parcel boundary) per lot.

    Returns a list of dicts ready to be serialised as GeoJSON-style features.
    """
    params = {
        "where": "DISPLAY = 'YES'",
        "outFields": "FID,TAX_MAP,PARCEL_NUM,STREET_NUM,STREET_NAM,LOCATION,NOTES,STRATEGY,District,CALC_ACRE,SQ_FT",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "json",
        "resultRecordCount": 2000,
    }
    try:
        resp = _requests.get(_ARCGIS_PARCEL_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("ArcGIS parcel query failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"ArcGIS query failed: {exc}") from exc

    features = []
    parcel_ids = []

    for feat in data.get("features", []):
        attrs = feat.get("attributes", {})
        geom  = feat.get("geometry", {})

        lat, lon = None, None
        if "rings" in geom:
            ring = geom["rings"][0]
            if ring:
                lon = sum(pt[0] for pt in ring) / len(ring)
                lat = sum(pt[1] for pt in ring) / len(ring)

        street_num = (attrs.get("STREET_NUM") or "").strip()
        street_nam = (attrs.get("STREET_NAM") or "").strip()
        address = f"{street_num} {street_nam}".strip() if street_num else street_nam
        tax_map = attrs.get("TAX_MAP")
        if tax_map:
            parcel_ids.append(tax_map)

        features.append({
            "parcel_id":  tax_map,
            "parcel_num": attrs.get("PARCEL_NUM"),
            "address":    address,
            "location":   (attrs.get("LOCATION") or "").strip(),
            "notes":      (attrs.get("NOTES") or "").strip(),
            "strategy":   attrs.get("STRATEGY"),
            "district":   attrs.get("District"),
            "acres":      attrs.get("CALC_ACRE"),
            "sq_ft":      attrs.get("SQ_FT"),
            "lat":        lat,
            "lon":        lon,
            "rings":      geom.get("rings", []),
        })

    # Batch-fetch violations for all parcels in one query and join.
    violations_by_parcel = _fetch_violations_bulk(parcel_ids)
    empty_signals = {cat: 0 for cat in _ALL_SIGNAL_CATEGORIES}

    for feature in features:
        v = violations_by_parcel.get(feature["parcel_id"], {})
        feature["violation_count"]    = v.get("violation_count", 0)
        feature["open_violations"]    = v.get("open_violations", 0)
        feature["signal_categories"]  = v.get("signal_categories", dict(empty_signals))

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 1 — Health check
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/", summary="Health check", tags=["System"])
def health_check() -> dict[str, Any]:
    """
    Returns the operational status of both subsystems:

    * **Scoring pipeline** — Gemini configuration and hero-parcel count.
    * **RAG chatbot** — ChromaDB chunk count and Groq LLM configuration.

    ``demo_3.html`` polls this endpoint to display the LIVE / OFFLINE badge.
    """
    gemini_status = (
        "configured" if GEMINI_API_KEY not in ("", "YOUR_GEMINI_KEY") else "mock"
    )
    llm_info = get_llm_info()
    kb_size = _chatbot.knowledge_base_size() if _chatbot else 0
    kb_ready = _chatbot.is_ready() if _chatbot else False

    return {
        "status": "ok",
        "service": "Montgomery RISE Unified API",
        "version": "4.0.0",
        "scoring_pipeline": "rise_selector_v3",
        "gemini": gemini_status,
        "hero_parcels": len(HERO_PARCELS),
        "rag_knowledge_base_chunks": kb_size,
        "rag_knowledge_base_ready": kb_ready,
        "rag_llm_provider": llm_info["provider"],
        "rag_llm_model": llm_info["model"],
        "rag_llm_configured": llm_info["configured"],
        "timestamp": _utcnow(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — Score weights config
# ─────────────────────────────────────────────────────────────────────────────

_VALID_WEIGHT_KEYS = frozenset(_DEFAULT_WEIGHTS.keys())


@app.get("/config/{city_id}/weights", summary="Get score weights for a city", tags=["Config"])
def get_weights(city_id: str = Path(..., examples=["montgomery"])) -> dict[str, Any]:
    """Returns the current score weights for *city_id*, or defaults if none saved."""
    return {
        "city_id": city_id,
        "weights": _load_weights(city_id),
    }


@app.post("/config/{city_id}/weights", summary="Save score weights for a city", tags=["Config"])
def save_weights(
    city_id: str = Path(..., examples=["montgomery"]),
    body: WeightsRequest = ...,
) -> dict[str, Any]:
    """
    Persist score weights for *city_id*.

    All 8 keys are required: ``heritage``, ``industrial``, ``activity``,
    ``proximity``, ``economic``, ``vacancy``, ``flood``, ``311``.
    Values must sum to 1.0 (±0.01 tolerance).
    """
    missing = _VALID_WEIGHT_KEYS - body.weights.keys()
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing weight keys: {sorted(missing)}")

    unknown = body.weights.keys() - _VALID_WEIGHT_KEYS
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown weight keys: {sorted(unknown)}")

    total = sum(body.weights.values())
    if abs(total - 1.0) > 0.01:
        raise HTTPException(status_code=422, detail=f"Weights must sum to 1.0, got {total:.4f}")

    _save_weights(city_id, body.weights)
    logger.info("Score weights saved for city '%s': %s", city_id, body.weights)

    return {
        "city_id":    city_id,
        "weights":    body.weights,
        "saved":      True,
        "updated_at": _utcnow(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 2 — List hero parcels
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/parcels", summary="List hero parcels", tags=["Parcels"])
def list_parcels() -> dict[str, Any]:
    """
    Returns lightweight metadata for all three hero parcels.

    Use ``GET /parcels/{id}/score`` to run the full scoring pipeline or
    ``GET /parcels/{id}/answer`` for the RAG AI summary.
    """
    parcels = []
    for key, idx in PARCEL_INDEX.items():
        p = HERO_PARCELS[idx]
        open_grants = sum(
            1 for g in p.get("grant_flags", []) if g.get("status") == "open"
        )
        parcels.append(
            {
                "id": key,
                "label": p["label"],
                "story": p["story"],
                "address": p["address"],
                "parcel_id": p["parcel_id"],
                "acres": p.get("acres"),
                "zone_context": p.get("zone_context"),
                "nearest_anchor": p.get("nearest_anchor"),
                "min_dist_miles": p.get("min_dist"),
                "open_grants": open_grants,
                "coords": {"lat": p["coords"][0], "lon": p["coords"][1]},
            }
        )
    return {"parcels": parcels}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint — City-owned vacant parcels (map layer)
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/map/vacant-parcels", summary="City-owned vacant parcels", tags=["Map"])
def vacant_parcels() -> dict[str, Any]:
    """
    Returns all city-owned vacant parcels from the Montgomery ArcGIS parcel layer.

    Filters: ``Owner LIKE '%CITY OF MONTGOMERY%'`` and ``ImpValue = 0``.

    Each feature includes ``lat``/``lon`` (WGS84 centroid), ``parcel_id``,
    ``address``, ``acres``, and ``land_value`` — ready to drop onto a map.
    """
    features = _fetch_vacant_city_parcels()
    return {
        "count": len(features),
        "parcels": features,
        "fetched_at": _utcnow(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint — Code violations for a surplus parcel (0.3 mi radius)
# ─────────────────────────────────────────────────────────────────────────────

_VIOLATIONS_RADIUS_MILES = 0.3
_VIOLATIONS_RADIUS_FEET  = _VIOLATIONS_RADIUS_MILES * 5280  # 1584 ft


@app.get(
    "/map/vacant-parcels/{parcel_id}/violations",
    summary="311 distress signals within 0.3 mi of a surplus parcel",
    tags=["Map"],
)
def parcel_violations(
    parcel_id: str = Path(
        ...,
        description="TAX_MAP parcel ID from /map/vacant-parcels, e.g. '04 09 32 1 019 008.000'",
        examples=["04 09 32 1 019 008.000"],
    ),
) -> dict[str, Any]:
    """
    Returns code enforcement violations within 0.3 miles of the parcel centroid.

    Each violation is mapped to one of the frontend 311 distress signal categories:
    Overgrown Vegetation, Illegal Dumping, Abandoned Building, Drug Activity,
    Vacant Lot, Noise Complaint, Graffiti.

    Summary includes total count, open/closed split, parcels scanned,
    and per-category counts.
    """
    # First fetch the parcel centroid from the surplus layer.
    parcel_params = {
        "where": f"TAX_MAP = '{parcel_id.replace(chr(39), '')}'",
        "outFields": "TAX_MAP",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "json",
        "resultRecordCount": 1,
    }
    try:
        pr = _requests.get(_ARCGIS_PARCEL_URL, params=parcel_params, timeout=15)
        pr.raise_for_status()
        pdata = pr.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Parcel lookup failed: {exc}") from exc

    feats = pdata.get("features", [])
    if not feats:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found.")

    geom = feats[0].get("geometry", {})
    lat, lon = None, None
    if "rings" in geom:
        ring = geom["rings"][0]
        if ring:
            lon = sum(pt[0] for pt in ring) / len(ring)
            lat = sum(pt[1] for pt in ring) / len(ring)

    if lat is None or lon is None:
        raise HTTPException(status_code=422, detail="Could not compute parcel centroid.")

    # Spatial query — violations within 0.3 miles of the centroid.
    params = {
        "where": "1=1",
        "outFields": "ParcelNo,OffenceNum,CaseDate,CaseType,CaseStatus,LienStatus,ComplaintRem,Year",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelWithin",
        "distance": _VIOLATIONS_RADIUS_FEET,
        "units": "esriSRUnit_Foot",
        "orderByFields": "CaseDate DESC",
        "f": "json",
        "resultRecordCount": 1000,
    }
    try:
        resp = _requests.get(_ARCGIS_VIOLATIONS_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("ArcGIS violations query failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"ArcGIS query failed: {exc}") from exc

    violations = []
    parcels_scanned: set[str] = set()
    category_counts = {cat: 0 for cat in _ALL_SIGNAL_CATEGORIES}

    for feat in data.get("features", []):
        a = feat.get("attributes", {})
        category = _casetype_to_category(a.get("CaseType"))
        category_counts[category] += 1
        if a.get("ParcelNo"):
            parcels_scanned.add(a["ParcelNo"])
        violations.append({
            "offence_num": a.get("OffenceNum"),
            "case_date":   a.get("CaseDate"),
            "case_type":   a.get("CaseType"),
            "category":    category,
            "case_status": a.get("CaseStatus"),
            "lien_status": a.get("LienStatus"),
            "complaint":   (a.get("ComplaintRem") or "").strip(),
            "year":        a.get("Year"),
            "parcel_no":   a.get("ParcelNo"),
        })

    open_count   = sum(1 for v in violations if (v["case_status"] or "").upper() == "OPEN")
    closed_count = len(violations) - open_count
    active_categories = [cat for cat, cnt in category_counts.items() if cnt > 0]

    return {
        "parcel_id":         parcel_id,
        "radius_miles":      _VIOLATIONS_RADIUS_MILES,
        "parcels_scanned":   len(parcels_scanned),
        "total":             len(violations),
        "open":              open_count,
        "closed":            closed_count,
        "active_categories": active_categories,
        "category_counts":   category_counts,
        "violations":        violations,
        "fetched_at":        _utcnow(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 3 — Full scoring pipeline for a hero parcel
# ─────────────────────────────────────────────────────────────────────────────


@app.get(
    "/parcels/{parcel_id}/score",
    summary="Score a hero parcel (full pipeline)",
    tags=["Parcels"],
)
def score_hero_parcel(
    parcel_id: str = Path(
        ...,
        description="Hero parcel identifier: A, B, or C",
        examples=["A"],
    ),
) -> dict[str, Any]:
    """
    Runs the complete RISE scoring pipeline for one hero parcel:

    1. ArcGIS foot-traffic query (1-mile radius, Most Visited Locations)
    2. FEMA flood-hazard lookup via Montgomery OneView GIS
    3. Montgomery 311 community-distress density (90-day window)
    4. 8-dimension scorer (Heritage 25 %, Industrial 20 %, Activity 15 % …)
    5. Gemini 2.5 Pro land-reuse recommendations (mock fallback if no key)

    The response shape matches exactly what ``demo_3.html`` expects.
    """
    pid = parcel_id.upper()
    if pid not in PARCEL_INDEX:
        raise HTTPException(
            status_code=404,
            detail=f"Parcel '{parcel_id}' not found. Valid IDs: A, B, C",
        )
    result = _run_pipeline(HERO_PARCELS[PARCEL_INDEX[pid]])
    # Upsert live scores into ChromaDB in a background thread so the RAG
    # chatbot always reflects the latest real-time data without blocking the response.
    _upsert_live_scores(pid, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 4 — Full scoring pipeline for arbitrary coordinates
# ─────────────────────────────────────────────────────────────────────────────


@app.post(
    "/parcels/custom",
    summary="Score a custom parcel",
    tags=["Parcels"],
)
def score_custom_parcel(body: CustomParcelRequest) -> dict[str, Any]:
    """
    Runs the complete RISE pipeline for any parcel defined by lat / lon.

    The nearest anchor is auto-detected, then all 8 dimensions are scored
    identically to the hero parcels.  Useful for planners exploring
    non-hero parcels from the GIS catalogue.
    """
    nearest_anchor, min_dist = _nearest_anchor(body.lat, body.lon)

    parcel: dict[str, Any] = {
        "story": body.story,
        "label": body.label,
        "address": body.address,
        "addr_source": "api_request",
        "parcel_id": body.parcel_id,
        "owner": body.owner,
        "coords": (body.lat, body.lon),
        "acres": body.acres,
        "zip": "",
        "nearest_anchor": nearest_anchor,
        "min_dist": min_dist,
        "avg_dist": min_dist,
        "raw_attrs": {"ImpValue": 0},
        "zone_context": body.zone_context,
        "health_flags": body.health_flags,
        "grant_flags": body.grant_flags,
    }
    result = _run_pipeline(parcel)
    # Custom parcels are upserted under their parcel_id so the RAG chatbot
    # can answer questions about them immediately after scoring.
    custom_pid = body.parcel_id.upper()[:16]  # safe key length
    _upsert_live_scores(custom_pid, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 5 — RAG AI summary for a hero parcel
# ─────────────────────────────────────────────────────────────────────────────


@app.get(
    "/parcels/{parcel_id}/answer",
    summary="RAG AI summary for a hero parcel",
    tags=["RAG Chatbot"],
)
def answer_parcel(
    parcel_id: str = Path(
        ...,
        description="Hero parcel identifier: A, B, or C",
        examples=["A"],
    ),
) -> dict[str, Any]:
    """
    Uses the RAG knowledge base (ChromaDB + Groq LLaMA-3.3-70B) to
    generate a detailed narrative summary for a hero parcel, grounded
    entirely in the knowledge-base documents.

    Covers: all dimension scores, top recommendations, open grants with
    deadlines and dollar amounts, and community / health context.
    """
    pid = parcel_id.upper().strip()
    if pid not in VALID_PARCEL_IDS:
        raise HTTPException(
            status_code=404,
            detail=f"Parcel '{parcel_id}' not found. Valid IDs: A, B, C",
        )

    chatbot = _require_rag()

    try:
        response = chatbot.ask(_RAG_SUMMARY_QUESTIONS[pid], parcel_filter=pid)
    except Exception:
        logger.exception("RAG error generating parcel summary for %s", pid)
        raise HTTPException(
            status_code=500, detail="Failed to generate RAG parcel summary."
        )

    return {
        "parcel_id": pid,
        "question": _RAG_SUMMARY_QUESTIONS[pid],
        "answer": response.answer,
        "sources": [vars(s) for s in _build_sources(response.sources)],
        "num_chunks_retrieved": response.num_chunks_retrieved,
        "used_fallback": response.used_fallback,
        "generated_at": _utcnow(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 6 — Free-form RAG question (JSON)
# ─────────────────────────────────────────────────────────────────────────────


@app.post(
    "/chat/ask",
    response_model=AskResponse,
    summary="Ask a free-form question about RISE",
    tags=["RAG Chatbot"],
)
def ask_question(body: AskRequest) -> AskResponse:
    """
    Ask any question about the RISE parcels, scoring model, grants, or
    Montgomery community context.

    Set ``parcel_filter`` to ``'A'``, ``'B'``, or ``'C'`` to restrict
    retrieval to chunks for that parcel plus general knowledge-base chunks.
    """
    chatbot = _require_rag()
    parcel_filter = _parse_parcel_filter(body.parcel_filter)

    try:
        response = chatbot.ask(body.question, parcel_filter=parcel_filter)
    except Exception:
        logger.exception("RAG error answering: %s", body.question[:80])
        raise HTTPException(
            status_code=500, detail="Failed to generate a RAG response."
        )

    return AskResponse(
        question=response.question,
        answer=response.answer,
        sources=_build_sources(response.sources),
        num_chunks_retrieved=response.num_chunks_retrieved,
        used_fallback=response.used_fallback,
        parcel_filter=parcel_filter,
        generated_at=_utcnow(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 7 — Streaming RAG answer (Server-Sent Events)
# ─────────────────────────────────────────────────────────────────────────────


@app.post(
    "/chat/stream",
    summary="Streaming RAG answer (Server-Sent Events)",
    tags=["RAG Chatbot"],
)
def stream_question(body: AskRequest) -> StreamingResponse:
    """
    Identical to ``POST /chat/ask`` but streams the answer token-by-token
    using Server-Sent Events.

    * Listen for ``data: [DONE]`` to detect end-of-stream.
    * On error the stream emits ``data: [ERROR]`` and closes.
    """
    chatbot = _require_rag()
    parcel_filter = _parse_parcel_filter(body.parcel_filter)

    def _event_generator():
        try:
            for token in chatbot.stream_ask(
                body.question, parcel_filter=parcel_filter
            ):
                if token == "__DONE__":
                    yield "data: [DONE]\n\n"
                else:
                    # Escape newlines: each SSE frame must stay on one line.
                    safe_token = token.replace("\n", " ")
                    yield f"data: {safe_token}\n\n"
        except Exception:
            logger.exception("SSE stream error: %s", body.question[:80])
            yield "data: [ERROR]\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  (local development only — Railway uses the uvicorn start command)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)