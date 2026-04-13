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

Map / 311 signals
  GET  /map/vacant-parcels          City-owned surplus parcels with violation counts
  GET  /map/violations-bulk         All ~20 000+ violation records with lat/lon
  GET  /map/violations              311 violations within 0.3 mi of one parcel

Environment variables
---------------------
  GEMINI_API_KEY   Gemini 2.5 Pro for scoring-pipeline recommendations
  GROQ_API_KEY     Groq LLaMA-3.3-70B for RAG chatbot answers

Quick start
-----------
  pip install -r requirements.txt
  uvicorn api:app --reload --port 8000

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

import requests as _requests
import uvicorn
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── optional dotenv ───────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
PARCEL_INDEX: dict[str, int] = {"A": 0, "B": 1, "C": 2}

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
# ArcGIS URLs + violation category mapping
# ─────────────────────────────────────────────────────────────────────────────

_ARCGIS_VIOLATIONS_URL = (
    "https://services7.arcgis.com/xNUwUjOJqYE54USz/arcgis/rest/services/"
    "Code_Violations/FeatureServer/0/query"
)

_ARCGIS_PARCEL_URL = (
    "https://services7.arcgis.com/xNUwUjOJqYE54USz/arcgis/rest/services/"
    "SURPLUS_CITY_PROPERTIES_polygon/FeatureServer/0/query"
)

_ALL_SIGNAL_CATEGORIES = [
    "Overgrown Vegetation",
    "Illegal Dumping",
    "Abandoned Building",
    "Drug Activity",
    "Vacant Lot",
    "Noise Complaint",
    "Graffiti",
]

_CASE_TYPE_CATEGORY: dict[str, str] = {
    "NUISANCE":              "Illegal Dumping",
    "OPEN VACANT":           "Vacant Lot",
    "DEMOLITION":            "Abandoned Building",
    "REPAIR":                "Abandoned Building",
    "GENERIC":               "Abandoned Building",
    "PARKING ON FRONT LAWN": "Illegal Dumping",
}

_VIOLATIONS_RADIUS_MILES = 0.3
_VIOLATIONS_RADIUS_FEET  = _VIOLATIONS_RADIUS_MILES * 5280  # 1 584 ft


def _casetype_to_category(case_type: str | None) -> str:
    return _CASE_TYPE_CATEGORY.get((case_type or "").upper(), "Abandoned Building")


# ─────────────────────────────────────────────────────────────────────────────
# Application lifecycle
# ─────────────────────────────────────────────────────────────────────────────

_chatbot: RiseChatbot | None = None


def _auto_ingest(chatbot: RiseChatbot) -> None:
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
        logger.error("No .txt files found in %s — RAG will not work.", DATA_DIR)
        return

    try:
        chunks = load_all_chunks(DATA_DIR)
        ingested = chatbot._store.upsert_chunks(chunks)  # noqa: SLF001
        logger.info(
            "Auto-ingest complete: %d chunks from %d files.",
            ingested, len(txt_files),
        )
    except Exception:
        logger.exception("Auto-ingest failed.")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global _chatbot  # noqa: PLW0603

    logger.info("Montgomery RISE v4.0.0 — starting up")
    try:
        _chatbot = RiseChatbot()
        _auto_ingest(_chatbot)
        logger.info("RAG chatbot ready — %d chunks.", _chatbot.knowledge_base_size())
        if not is_llm_available():
            logger.warning("GROQ_API_KEY not configured. RAG will return fallback context only.")
    except Exception:
        logger.exception("RAG chatbot failed to initialise.")
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
        "parcel scoring pipeline + RAG chatbot for Montgomery, Alabama. v4.0.0."
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
    lat: float = Field(..., examples=[32.3789], description="Latitude (WGS84)")
    lon: float = Field(..., examples=[-86.3109], description="Longitude (WGS84)")
    label: str = Field("Custom Parcel", examples=["My Parcel"])
    story: str = Field("Custom", examples=["Economic Urgency"])
    address: str = Field("Montgomery, Alabama", examples=["123 Main St, Montgomery AL"])
    parcel_id: str = Field("CUSTOM-001", examples=["CUSTOM-001"])
    acres: float = Field(1.0, examples=[2.5])
    owner: str = Field("City of Montgomery", examples=["City of Montgomery"])
    zone_context: str = Field("heritage", examples=["heritage"])
    health_flags: dict[str, Any] = Field(default_factory=dict)
    grant_flags: list[dict[str, Any]] = Field(default_factory=list)


class WeightsRequest(BaseModel):
    weights: dict[str, float] = Field(
        ...,
        description="Score dimension weights. Keys: heritage, industrial, activity, proximity, economic, vacancy, flood, 311. Must sum to 1.0.",
        examples=[{
            "heritage": 0.25, "industrial": 0.20, "activity": 0.15,
            "proximity": 0.10, "economic": 0.10, "vacancy": 0.10,
            "flood": 0.05, "311": 0.05,
        }],
    )


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    parcel_filter: Optional[str] = Field(default=None, examples=["C"])


class SourceModel(BaseModel):
    document_title: str
    parcel_id: str
    topic: str
    source_file: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]
    num_chunks_retrieved: int
    used_fallback: bool
    parcel_filter: Optional[str]
    generated_at: str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — misc
# ─────────────────────────────────────────────────────────────────────────────


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nearest_anchor(lat: float, lon: float) -> tuple[str, float]:
    best_name, best_dist = "Unknown", float("inf")
    for anchor in ANCHORS:
        dist = calculate_distance(lat, lon, anchor["lat"], anchor["lon"])
        if dist < best_dist:
            best_dist, best_name = dist, anchor["name"]
    return best_name, round(best_dist, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — scoring pipeline
# ─────────────────────────────────────────────────────────────────────────────


def _run_pipeline(parcel: dict[str, Any]) -> dict[str, Any]:
    t_start = time.perf_counter()
    p = copy.deepcopy(parcel)

    live_grant_data = get_grant_data()
    p["grant_flags"] = merge_grants(p.get("grant_flags", []), live_grant_data.get("grants", []))

    foot_traffic: dict[str, Any] = get_foot_traffic(p)
    scores: dict[str, Any] = compute_score(p, foot_traffic)
    p["scores"] = scores
    ai_analysis: dict[str, Any] = analyse_with_gemini(p, foot_traffic)

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    flood_risk = {
        "score": scores["flood"],
        "zone":  scores["flood_zone"],
        "label": scores["flood_label"],
    }
    distress_311 = {
        "score":              scores["distress"],
        "density_per_sq_mi":  scores["destress_density"],
        "total_calls_90days": scores["destress_calls_90days"],
        "top_complaints":     scores["destress_top_complaints"],
        "label":              scores["destress_label"],
    }
    scores_out = {**scores, "urgency": ai_analysis.get("urgency_flag", "medium")}
    gemini_live = GEMINI_API_KEY not in ("", "YOUR_GEMINI_KEY")

    return {
        "label":          p["label"],
        "story":          p["story"],
        "address":        p["address"],
        "parcel_id":      p["parcel_id"],
        "acres":          p.get("acres"),
        "nearest_anchor": p["nearest_anchor"],
        "min_dist_miles": p["min_dist"],
        "zone_context":   p.get("zone_context", ""),
        "owner":          p.get("owner", "City of Montgomery"),
        "scores":         scores_out,
        "flood_risk":     flood_risk,
        "distress_311":   distress_311,
        "foot_traffic": {
            "score":          foot_traffic["score"],
            "location_count": foot_traffic["location_count"],
            "total_visits":   foot_traffic["total_visits"],
            "nearest_name":   foot_traffic["nearest_name"],
            "nearest_dist_mi":foot_traffic["nearest_dist_mi"],
            "source":         foot_traffic["source"],
            "top_locations":  foot_traffic["top_locations"][:5],
        },
        "grant_flags":  p.get("grant_flags", []),
        "health_flags": p.get("health_flags", {}),
        "ai_analysis": {
            **ai_analysis,
            "ai_source": "gemini-2.5-pro" if gemini_live else "mock",
        },
        "meta": {
            "generated_at": _utcnow(),
            "pipeline":     "RISE v4 — ArcGIS + Gemini 2.5 Pro + Groq RAG",
            "pipeline_ms":  elapsed_ms,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — live RAG upsert
# ─────────────────────────────────────────────────────────────────────────────


def _build_live_document(pid: str, result: dict[str, Any]) -> str:
    scores    = result.get("scores", {})
    ft        = result.get("foot_traffic", {})
    flood     = result.get("flood_risk", {})
    d311      = result.get("distress_311", {})
    ai        = result.get("ai_analysis", {})
    grants    = result.get("grant_flags", [])
    health    = result.get("health_flags", {})
    recs      = ai.get("recommendations", [])
    generated = result.get("meta", {}).get("generated_at", _utcnow())

    grant_lines = [
        f"  - {g.get('name','Unknown grant')}: {g.get('status','unknown').upper()}, "
        f"{g.get('days_remaining','?')} days remaining, covers {g.get('eligibility_pct','?')}% "
        f"of costs, {g.get('match','?')} match required."
        for g in grants
    ]
    grants_text = "\n".join(grant_lines) if grant_lines else "  No open grants found."

    rec_lines = [
        f"  {r.get('rank','?')}. {r.get('name','Unknown')} "
        f"(fit score {r.get('fit_score','?')}/100, {r.get('cost_tier','?')}): "
        f"{r.get('explanation','')}"
        for r in recs[:3]
    ]
    recs_text = "\n".join(rec_lines) if rec_lines else "  No recommendations generated."

    ft_lines = [
        f"  - {loc.get('name','?')}: {loc.get('visits','?'):,} visits "
        f"({loc.get('dist_mi','?')} miles)"
        for loc in ft.get("top_locations", [])[:5]
    ]
    ft_text = "\n".join(ft_lines) if ft_lines else "  No foot-traffic locations found."

    health_lines = [f"  - {k}: {v}" for k, v in health.items()]
    health_text  = "\n".join(health_lines) if health_lines else "  No health flags recorded."

    return f"""\
DOCUMENT: Parcel {pid} — Live Real-Time Score ({generated})
PARCEL_ID: {pid}
TOPIC: live_scores

PARCEL IDENTITY
  Label          : {result.get("label","?")}
  Address        : {result.get("address","?")}
  Parcel ID      : {result.get("parcel_id","?")}
  Acres          : {result.get("acres","?")}
  Zone context   : {result.get("zone_context","?")}
  Nearest anchor : {result.get("nearest_anchor","?")} ({result.get("min_dist_miles","?")} miles)
  Owner          : {result.get("owner","?")}

RISE SCORES (live — fetched {generated})
  Final RISE score : {scores.get("final","?")} / 100
  Urgency flag     : {scores.get("urgency","?")}
  Heritage         : {scores.get("heritage","?")} / 100
  Industrial       : {scores.get("industrial","?")} / 100
  Activity         : {scores.get("activity","?")} / 100
  Proximity        : {scores.get("proximity","?")} / 100
  Economic         : {scores.get("economic","?")} / 100
  Vacancy          : {scores.get("vacancy","?")} / 100
  Flood risk       : {scores.get("flood","?")} / 5  (zone: {flood.get("zone","?")} — {flood.get("label","?")})
  311 distress     : {scores.get("distress","?")} / 10 — {d311.get("label","?")}
    Density        : {d311.get("density_per_sq_mi","?")} calls/sq mi (90-day window)
    Total calls    : {d311.get("total_calls_90days","?")}
    Top complaints : {", ".join(str(c.get("type",c)) if isinstance(c,dict) else str(c) for c in d311.get("top_complaints",[])) or "none"}

FOOT TRAFFIC
  Score           : {ft.get("score","?")} / 100
  Locations found : {ft.get("location_count","?")} within 1 mile
  Total visits    : {ft.get("total_visits","?"):,} (weighted)
  Nearest venue   : {ft.get("nearest_name","?")} ({ft.get("nearest_dist_mi","?")} miles)
  Source          : {ft.get("source","?")}
  Top locations:
{ft_text}

AI RECOMMENDATIONS (generated by {ai.get("ai_source","Gemini")} at {generated})
  One-line summary : {ai.get("one_line_summary","?")}
{recs_text}

OPEN GRANT WINDOWS
{grants_text}

COMMUNITY HEALTH FLAGS
{health_text}
"""


def _upsert_live_scores(pid: str, result: dict[str, Any]) -> None:
    if _chatbot is None:
        return

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
                    "source_file":    f"live_scores_parcel_{pid.lower()}",
                    "document_title": f"Parcel {pid} — Live Real-Time Score ({generated})",
                    "parcel_id":      pid,
                    "topic":          "live_scores",
                    "block_index":    0,
                    "generated_at":   generated,
                },
            )
            _chatbot._store.upsert_chunks([chunk])  # noqa: SLF001
            logger.info("Live scores for Parcel %s upserted into ChromaDB.", pid)
        except Exception:
            logger.exception("Background RAG upsert failed for Parcel %s.", pid)

    threading.Thread(target=_worker, daemon=True, name=f"rag-upsert-{pid}").start()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — RAG chatbot
# ─────────────────────────────────────────────────────────────────────────────


def _require_rag() -> RiseChatbot:
    if _chatbot is None:
        raise HTTPException(status_code=503, detail="RAG chatbot is not available.")
    if not _chatbot.is_ready():
        raise HTTPException(status_code=503, detail="RAG knowledge base is empty.")
    return _chatbot


def _parse_parcel_filter(raw: Optional[str]) -> Optional[str]:
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
    return [
        SourceModel(
            document_title=s.get("document_title", "Unknown"),
            parcel_id=s.get("parcel_id", "general"),
            topic=s.get("topic", "general"),
            source_file=s.get("source_file", "unknown"),
        )
        for s in raw
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — score weights persistence
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
    if _WEIGHTS_FILE.exists():
        try:
            data = json.loads(_WEIGHTS_FILE.read_text(encoding="utf-8"))
            if city_id in data:
                return data[city_id]["weights"]
        except Exception:
            pass
    return dict(_DEFAULT_WEIGHTS)


def _save_weights(city_id: str, weights: dict[str, float]) -> None:
    data: dict[str, Any] = {}
    if _WEIGHTS_FILE.exists():
        try:
            data = json.loads(_WEIGHTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    data[city_id] = {"weights": weights, "updated_at": _utcnow()}
    _WEIGHTS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — ArcGIS queries
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_violations_bulk(parcel_ids: list[str] | None = None) -> dict[str, dict]:
    """
    Fetch violation summary counts for a list of parcel IDs (used by
    /map/vacant-parcels to join counts onto each surplus parcel).
    Returns a dict keyed by parcel_id.
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

    summaries: dict[str, dict] = {}
    for feat in data.get("features", []):
        a   = feat.get("attributes", {})
        pid = a.get("ParcelNo")
        if not pid:
            continue
        if pid not in summaries:
            summaries[pid] = {
                "violation_count":  0,
                "open_violations":  0,
                "signal_categories": {cat: 0 for cat in _ALL_SIGNAL_CATEGORIES},
            }
        summaries[pid]["violation_count"] += 1
        if (a.get("CaseStatus") or "").upper() == "OPEN":
            summaries[pid]["open_violations"] += 1
        summaries[pid]["signal_categories"][_casetype_to_category(a.get("CaseType"))] += 1

    return summaries


def _fetch_vacant_city_parcels() -> list[dict[str, Any]]:
    """
    Query the Montgomery ArcGIS surplus city properties layer (DISPLAY = 'YES').
    Joins violation summary counts onto each parcel.
    """
    params = {
        "where": "DISPLAY = 'YES'",
        "outFields": (
            "FID,TAX_MAP,PARCEL_NUM,STREET_NUM,STREET_NAM,"
            "LOCATION,NOTES,STRATEGY,District,CALC_ACRE,SQ_FT"
        ),
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

    features: list[dict[str, Any]] = []
    parcel_ids: list[str] = []

    for feat in data.get("features", []):
        attrs = feat.get("attributes", {})
        geom  = feat.get("geometry", {})

        lat = lon = None
        if "rings" in geom:
            ring = geom["rings"][0]
            if ring:
                lon = sum(pt[0] for pt in ring) / len(ring)
                lat = sum(pt[1] for pt in ring) / len(ring)

        street_num = (attrs.get("STREET_NUM") or "").strip()
        street_nam = (attrs.get("STREET_NAM") or "").strip()
        address    = f"{street_num} {street_nam}".strip() if street_num else street_nam
        tax_map    = attrs.get("TAX_MAP")
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

    # Join violation counts
    violations_by_parcel = _fetch_violations_bulk(parcel_ids)
    empty_signals = {cat: 0 for cat in _ALL_SIGNAL_CATEGORIES}

    for feature in features:
        v = violations_by_parcel.get(feature["parcel_id"], {})
        feature["violation_count"]   = v.get("violation_count", 0)
        feature["open_violations"]   = v.get("open_violations", 0)
        feature["signal_categories"] = v.get("signal_categories", dict(empty_signals))

    return features


def _fetch_all_violations_with_coords() -> list[dict[str, Any]]:
    """
    Paginate through the ArcGIS Code Violations layer and return every record
    with its point geometry converted to WGS84 lat/lon.
    ~20 000+ records paged at 2 000 per request.
    """
    all_violations: list[dict[str, Any]] = []
    offset    = 0
    page_size = 2000

    while True:
        params = {
            "where": "1=1",
            "outFields": (
                "ParcelNo,OffenceNum,CaseDate,CaseType,CaseStatus,"
                "LienStatus,ComplaintRem,Year,Address"
            ),
            "returnGeometry":  "true",
            "outSR":           "4326",
            "f":               "json",
            "resultRecordCount": page_size,
            "resultOffset":    offset,
            "orderByFields":   "CaseDate DESC",
        }
        try:
            resp = _requests.get(_ARCGIS_VIOLATIONS_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Violations page fetch failed at offset %d: %s", offset, exc)
            break

        features = data.get("features", [])
        if not features:
            break

        for feat in features:
            a    = feat.get("attributes", {})
            geom = feat.get("geometry") or {}
            lat  = geom.get("y")
            lon  = geom.get("x")

            if lat is None or lon is None:
                continue

            case_date_ms  = a.get("CaseDate")
            case_date_str = None
            if case_date_ms:
                try:
                    case_date_str = datetime.fromtimestamp(
                        case_date_ms / 1000, tz=timezone.utc
                    ).strftime("%Y-%m-%d")
                except Exception:
                    case_date_str = str(case_date_ms)

            all_violations.append({
                "offence_num": a.get("OffenceNum"),
                "parcel_no":   a.get("ParcelNo"),
                "address":     (a.get("Address") or "").strip(),
                "case_date":   case_date_str,
                "case_type":   (a.get("CaseType") or "").strip(),
                "case_status": (a.get("CaseStatus") or "").strip(),
                "lien_status": (a.get("LienStatus") or "").strip(),
                "complaint":   (a.get("ComplaintRem") or "").strip(),
                "year":        a.get("Year"),
                "category":    _casetype_to_category(a.get("CaseType")),
                "lat":         lat,
                "lon":         lon,
            })

        if len(features) < page_size:
            break  # last page

        offset += page_size
        logger.info("Violations fetched so far: %d", len(all_violations))

    logger.info("Total violations fetched: %d", len(all_violations))
    return all_violations


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint — Health check
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/", summary="Health check", tags=["System"])
def health_check() -> dict[str, Any]:
    gemini_status = (
        "configured" if GEMINI_API_KEY not in ("", "YOUR_GEMINI_KEY") else "mock"
    )
    llm_info = get_llm_info()
    kb_size  = _chatbot.knowledge_base_size() if _chatbot else 0
    kb_ready = _chatbot.is_ready() if _chatbot else False

    return {
        "status":                    "ok",
        "service":                   "Montgomery RISE Unified API",
        "version":                   "4.0.0",
        "scoring_pipeline":          "rise_selector_v3",
        "gemini":                    gemini_status,
        "hero_parcels":              len(HERO_PARCELS),
        "rag_knowledge_base_chunks": kb_size,
        "rag_knowledge_base_ready":  kb_ready,
        "rag_llm_provider":          llm_info["provider"],
        "rag_llm_model":             llm_info["model"],
        "rag_llm_configured":        llm_info["configured"],
        "timestamp":                 _utcnow(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — Score weights config
# ─────────────────────────────────────────────────────────────────────────────

_VALID_WEIGHT_KEYS = frozenset(_DEFAULT_WEIGHTS.keys())


@app.get("/config/{city_id}/weights", summary="Get score weights", tags=["Config"])
def get_weights(city_id: str = Path(..., examples=["montgomery"])) -> dict[str, Any]:
    return {"city_id": city_id, "weights": _load_weights(city_id)}


@app.post("/config/{city_id}/weights", summary="Save score weights", tags=["Config"])
def save_weights(
    city_id: str = Path(..., examples=["montgomery"]),
    body: WeightsRequest = ...,
) -> dict[str, Any]:
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
    logger.info("Score weights saved for city '%s'.", city_id)
    return {"city_id": city_id, "weights": body.weights, "saved": True, "updated_at": _utcnow()}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint — List hero parcels
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/parcels", summary="List hero parcels", tags=["Parcels"])
def list_parcels() -> dict[str, Any]:
    parcels = []
    for key, idx in PARCEL_INDEX.items():
        p = HERO_PARCELS[idx]
        open_grants = sum(1 for g in p.get("grant_flags", []) if g.get("status") == "open")
        parcels.append({
            "id":             key,
            "label":          p["label"],
            "story":          p["story"],
            "address":        p["address"],
            "parcel_id":      p["parcel_id"],
            "acres":          p.get("acres"),
            "zone_context":   p.get("zone_context"),
            "nearest_anchor": p.get("nearest_anchor"),
            "min_dist_miles": p.get("min_dist"),
            "open_grants":    open_grants,
            "coords":         {"lat": p["coords"][0], "lon": p["coords"][1]},
        })
    return {"parcels": parcels}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint — City-owned vacant parcels (map layer)
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/map/vacant-parcels", summary="City-owned surplus parcels", tags=["Map"])
def vacant_parcels() -> dict[str, Any]:
    """
    Returns all 79 city-owned surplus parcels with centroid lat/lon, address,
    acreage, and a joined violation count summary per parcel.
    """
    features = _fetch_vacant_city_parcels()
    return {"count": len(features), "parcels": features, "fetched_at": _utcnow()}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint — All violations with lat/lon (bulk, for client-side radius filter)
# ─────────────────────────────────────────────────────────────────────────────


@app.get(
    "/map/violations-bulk",
    summary="All code violations with lat/lon (client-side radius filtering)",
    tags=["Map"],
)
def all_violations_with_coords() -> dict[str, Any]:
    """
    Returns every code violation record (~20 000+) from the Montgomery ArcGIS
    layer, each with its WGS84 lat/lon point so the frontend can Haversine-
    filter to any radius without a per-parcel API call.

    Paginates ArcGIS automatically (2 000 records per page).
    Cache this client-side — it changes daily at most.
    """
    violations = _fetch_all_violations_with_coords()
    return {"count": len(violations), "violations": violations, "fetched_at": _utcnow()}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint — Violations within 0.3 mi of one surplus parcel (detail view)
# ─────────────────────────────────────────────────────────────────────────────


@app.get(
    "/map/violations",
    summary="311 distress signals within 0.3 mi of a surplus parcel",
    tags=["Map"],
)
def parcel_violations(
    parcel_id: str = Query(
        ...,
        description="TAX_MAP parcel ID from /map/vacant-parcels",
        examples=["04 09 32 1 019 008.000"],
    ),
) -> dict[str, Any]:
    """
    Returns code enforcement violations within 0.3 miles of the named parcel's
    centroid, fetched via a spatial query against the ArcGIS violations layer.
    """
    # Look up the parcel centroid
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
    lat = lon = None
    if "rings" in geom:
        ring = geom["rings"][0]
        if ring:
            lon = sum(pt[0] for pt in ring) / len(ring)
            lat = sum(pt[1] for pt in ring) / len(ring)

    if lat is None or lon is None:
        raise HTTPException(status_code=422, detail="Could not compute parcel centroid.")

    # Spatial query — violations within radius
    params = {
        "where": "1=1",
        "outFields": (
            "ParcelNo,OffenceNum,CaseDate,CaseType,CaseStatus,"
            "LienStatus,ComplaintRem,Year"
        ),
        "geometry":     f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR":         "4326",
        "spatialRel":   "esriSpatialRelWithin",
        "distance":     _VIOLATIONS_RADIUS_FEET,
        "units":        "esriSRUnit_Foot",
        "orderByFields":"CaseDate DESC",
        "f":            "json",
        "resultRecordCount": 1000,
    }
    try:
        resp = _requests.get(_ARCGIS_VIOLATIONS_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ArcGIS query failed: {exc}") from exc

    violations      = []
    parcels_scanned: set[str] = set()
    category_counts = {cat: 0 for cat in _ALL_SIGNAL_CATEGORIES}

    for feat in data.get("features", []):
        a        = feat.get("attributes", {})
        category = _casetype_to_category(a.get("CaseType"))
        category_counts[category] += 1
        if a.get("ParcelNo"):
            parcels_scanned.add(a["ParcelNo"])

        case_date_ms  = a.get("CaseDate")
        case_date_str = None
        if case_date_ms:
            try:
                case_date_str = datetime.fromtimestamp(
                    case_date_ms / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d")
            except Exception:
                case_date_str = str(case_date_ms)

        violations.append({
            "offence_num": a.get("OffenceNum"),
            "case_date":   case_date_str,
            "case_type":   (a.get("CaseType") or "").strip(),
            "category":    category,
            "case_status": (a.get("CaseStatus") or "").strip(),
            "lien_status": (a.get("LienStatus") or "").strip(),
            "complaint":   (a.get("ComplaintRem") or "").strip(),
            "year":        a.get("Year"),
            "parcel_no":   a.get("ParcelNo"),
        })

    open_count       = sum(1 for v in violations if v["case_status"].upper() == "OPEN")
    active_categories = [cat for cat, cnt in category_counts.items() if cnt > 0]

    return {
        "parcel_id":         parcel_id,
        "radius_miles":      _VIOLATIONS_RADIUS_MILES,
        "parcels_scanned":   len(parcels_scanned),
        "total":             len(violations),
        "open":              open_count,
        "closed":            len(violations) - open_count,
        "active_categories": active_categories,
        "category_counts":   category_counts,
        "violations":        violations,
        "fetched_at":        _utcnow(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint — Score a hero parcel
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/parcels/{parcel_id}/score", summary="Score a hero parcel", tags=["Parcels"])
def score_hero_parcel(
    parcel_id: str = Path(..., description="Hero parcel: A, B, or C", examples=["A"]),
) -> dict[str, Any]:
    pid = parcel_id.upper()
    if pid not in PARCEL_INDEX:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found. Valid: A, B, C")
    result = _run_pipeline(HERO_PARCELS[PARCEL_INDEX[pid]])
    _upsert_live_scores(pid, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint — Score a custom parcel
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/parcels/custom", summary="Score a custom parcel", tags=["Parcels"])
def score_custom_parcel(body: CustomParcelRequest) -> dict[str, Any]:
    nearest_anchor, min_dist = _nearest_anchor(body.lat, body.lon)
    parcel: dict[str, Any] = {
        "story":          body.story,
        "label":          body.label,
        "address":        body.address,
        "addr_source":    "api_request",
        "parcel_id":      body.parcel_id,
        "owner":          body.owner,
        "coords":         (body.lat, body.lon),
        "acres":          body.acres,
        "zip":            "",
        "nearest_anchor": nearest_anchor,
        "min_dist":       min_dist,
        "avg_dist":       min_dist,
        "raw_attrs":      {"ImpValue": 0},
        "zone_context":   body.zone_context,
        "health_flags":   body.health_flags,
        "grant_flags":    body.grant_flags,
    }
    result     = _run_pipeline(parcel)
    custom_pid = body.parcel_id.upper()[:16]
    _upsert_live_scores(custom_pid, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint — RAG AI summary for a hero parcel
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/parcels/{parcel_id}/answer", summary="RAG AI summary for a hero parcel", tags=["RAG Chatbot"])
def answer_parcel(
    parcel_id: str = Path(..., description="Hero parcel: A, B, or C", examples=["A"]),
) -> dict[str, Any]:
    pid = parcel_id.upper().strip()
    if pid not in VALID_PARCEL_IDS:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found.")
    chatbot = _require_rag()
    try:
        response = chatbot.ask(_RAG_SUMMARY_QUESTIONS[pid], parcel_filter=pid)
    except Exception:
        logger.exception("RAG error for parcel %s", pid)
        raise HTTPException(status_code=500, detail="Failed to generate RAG summary.")
    return {
        "parcel_id":           pid,
        "question":            _RAG_SUMMARY_QUESTIONS[pid],
        "answer":              response.answer,
        "sources":             [vars(s) for s in _build_sources(response.sources)],
        "num_chunks_retrieved":response.num_chunks_retrieved,
        "used_fallback":       response.used_fallback,
        "generated_at":        _utcnow(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint — Free-form RAG question (JSON)
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/chat/ask", response_model=AskResponse, summary="Ask a free-form RAG question", tags=["RAG Chatbot"])
def ask_question(body: AskRequest) -> AskResponse:
    chatbot       = _require_rag()
    parcel_filter = _parse_parcel_filter(body.parcel_filter)
    try:
        response = chatbot.ask(body.question, parcel_filter=parcel_filter)
    except Exception:
        logger.exception("RAG error: %s", body.question[:80])
        raise HTTPException(status_code=500, detail="Failed to generate a RAG response.")
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
# Endpoint — Streaming RAG answer (SSE)
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/chat/stream", summary="Streaming RAG answer (SSE)", tags=["RAG Chatbot"])
def stream_question(body: AskRequest) -> StreamingResponse:
    chatbot       = _require_rag()
    parcel_filter = _parse_parcel_filter(body.parcel_filter)

    def _event_generator():
        try:
            for token in chatbot.stream_ask(body.question, parcel_filter=parcel_filter):
                if token == "__DONE__":
                    yield "data: [DONE]\n\n"
                else:
                    yield f"data: {token.replace(chr(10), ' ')}\n\n"
        except Exception:
            logger.exception("SSE stream error: %s", body.question[:80])
            yield "data: [ERROR]\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)