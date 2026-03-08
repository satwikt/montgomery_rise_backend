"""
Montgomery RISE — FastAPI Server
=================================
Wraps rise_selector_v3.py pipeline and exposes 4 endpoints:

  GET  /                          Health check + Gemini status
  GET  /parcels                   List all 3 hero parcels (metadata only)
  GET  /parcels/{id}/score        Full pipeline for a hero parcel (A | B | C)
  POST /parcels/custom            Full pipeline for arbitrary coords

Setup:
  pip install fastapi uvicorn python-dotenv requests

Run:
  uvicorn api:app --reload --port 8000

Environment:
  GEMINI_API_KEY=your_key_here   (set in .env or environment)
"""

import copy
import time
import os
import uvicorn
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── load .env if present (python-dotenv is optional) ─────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — rely on shell environment

# ── import the entire scoring pipeline from rise_selector_v3 ─────────────────
# All heavy logic lives there; this file is pure HTTP routing.
from rise_selector_v3 import (
    HERO_PARCELS,
    GEMINI_API_KEY,
    get_foot_traffic,
    compute_score,
    analyse_with_gemini,
    calculate_distance,
    ANCHOR_META,
    ANCHORS,
)

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Montgomery RISE API",
    description=(
        "Revitalization Intelligence & Smart Empowerment — "
        "parcel scoring pipeline for Montgomery, Alabama."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the HTML demo (served from any origin) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# PARCEL ID → INDEX MAP
# ─────────────────────────────────────────────────────────────────────────────

PARCEL_MAP: dict[str, int] = {"A": 0, "B": 1, "C": 2}

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class CustomParcelRequest(BaseModel):
    """Body for POST /parcels/custom"""

    lat: float = Field(..., example=32.3789, description="Latitude (WGS84)")
    lon: float = Field(..., example=-86.3109, description="Longitude (WGS84)")
    label: str = Field("Custom Parcel", example="My Parcel")
    story: str = Field("Custom", example="Economic Urgency")
    address: str = Field("Montgomery, Alabama", example="123 Main St, Montgomery AL")
    parcel_id: str = Field("CUSTOM-001", example="CUSTOM-001")
    acres: float = Field(1.0, example=2.5, description="Parcel size in acres")
    owner: str = Field("City of Montgomery", example="City of Montgomery")
    zone_context: str = Field(
        "heritage",
        example="heritage",
        description="One of: heritage | ix_hub | food_desert",
    )
    health_flags: dict = Field(
        default_factory=dict,
        example={"food_insecurity_pct": 35, "asthma_rate_multiplier": 1.8},
    )
    grant_flags: list = Field(
        default_factory=list,
        example=[
            {
                "name": "USDA Rural Economic Development Q3",
                "status": "open",
                "days_remaining": 26,
                "eligibility_pct": 70,
                "match": "20%",
            }
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_anchor(lat: float, lon: float) -> tuple[str, float]:
    """Return (anchor_name, distance_miles) for the closest anchor."""
    best_name, best_dist = "Unknown", float("inf")
    for anchor in ANCHORS:
        d = calculate_distance(lat, lon, anchor["lat"], anchor["lon"])
        if d < best_dist:
            best_dist, best_name = d, anchor["name"]
    return best_name, round(best_dist, 3)


def _run_pipeline(parcel: dict) -> dict:
    """
    Execute the full RISE pipeline for any parcel dict and return a
    serialisable response payload that matches the shape demo_3.html expects.
    """
    t0 = time.perf_counter()

    # Deep-copy so we never mutate the in-memory HERO_PARCELS list
    p = copy.deepcopy(parcel)

    # Step 1 — ArcGIS foot traffic
    foot_traffic = get_foot_traffic(p)

    # Step 2 — 8-dimension scorer
    scores = compute_score(p, foot_traffic)
    p["scores"] = scores

    # Step 3 — Gemini AI (or mock fallback)
    ai_analysis = analyse_with_gemini(p, foot_traffic)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # ── reshape scores into the flood_risk / distress_311 sub-objects
    # that demo_3.html destructures
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

    # ── scores sub-object (add urgency from AI for the urgency pill)
    scores_out = {k: v for k, v in scores.items()}
    scores_out["urgency"] = ai_analysis.get("urgency_flag", "medium")

    return {
        # Parcel identity
        "label":            p["label"],
        "story":            p["story"],
        "address":          p["address"],
        "parcel_id":        p["parcel_id"],
        "acres":            p.get("acres"),
        "nearest_anchor":   p["nearest_anchor"],
        "min_dist_miles":   p["min_dist"],
        "zone_context":     p.get("zone_context", ""),
        "owner":            p.get("owner", "City of Montgomery"),
        # Scoring
        "scores":           scores_out,
        # Signal sub-panels
        "flood_risk":       flood_risk,
        "distress_311":     distress_311,
        "foot_traffic": {
            "score":           foot_traffic["score"],
            "location_count":  foot_traffic["location_count"],
            "total_visits":    foot_traffic["total_visits"],
            "nearest_name":    foot_traffic["nearest_name"],
            "nearest_dist_mi": foot_traffic["nearest_dist_mi"],
            "source":          foot_traffic["source"],
            "top_locations":   foot_traffic["top_locations"][:5],
        },
        # Right-rail panels
        "grant_flags":      p.get("grant_flags", []),
        "health_flags":     p.get("health_flags", {}),
        # AI
        "ai_analysis": {
            **ai_analysis,
            "ai_source": (
                "gemini-2.5-pro"
                if GEMINI_API_KEY != "YOUR_GEMINI_KEY"
                else "mock"
            ),
        },
        # Metadata
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "pipeline":     "RISE v3 — ArcGIS + Gemini 2.5 Pro",
            "pipeline_ms":  elapsed_ms,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check", tags=["System"])
def health_check():
    """
    Returns API status and whether Gemini is configured.
    demo_3.html polls this endpoint to decide 'LIVE' vs 'OFFLINE'.
    """
    gemini_status = (
        "configured" if GEMINI_API_KEY not in ("", "YOUR_GEMINI_KEY") else "mock"
    )
    return {
        "status":  "ok",
        "service": "Montgomery RISE API",
        "version": "3.0.0",
        "gemini":  gemini_status,
        "parcels": len(HERO_PARCELS),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — List hero parcels (metadata only, no scoring)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/parcels", summary="List hero parcels", tags=["Parcels"])
def list_parcels():
    """
    Returns lightweight metadata for all 3 pre-selected hero parcels.
    Use GET /parcels/{id}/score to run the full scoring pipeline.
    """
    result = []
    for key, idx in PARCEL_MAP.items():
        p = HERO_PARCELS[idx]
        open_grants = [g for g in p.get("grant_flags", []) if g.get("status") == "open"]
        result.append(
            {
                "id":             key,
                "label":          p["label"],
                "story":          p["story"],
                "address":        p["address"],
                "parcel_id":      p["parcel_id"],
                "acres":          p.get("acres"),
                "zone_context":   p.get("zone_context"),
                "nearest_anchor": p.get("nearest_anchor"),
                "min_dist_miles": p.get("min_dist"),
                "open_grants":    len(open_grants),
                "coords": {
                    "lat": p["coords"][0],
                    "lon": p["coords"][1],
                },
            }
        )
    return {"parcels": result}


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — Full scoring pipeline for a hero parcel
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/parcels/{parcel_id}/score",
    summary="Score a hero parcel",
    tags=["Parcels"],
)
def score_hero_parcel(
    parcel_id: str = Path(
        ...,
        description="Hero parcel identifier: A, B, or C",
        example="A",
    )
):
    """
    Runs the complete RISE pipeline for one of the 3 hero parcels:

    1. ArcGIS foot traffic (Most Visited Locations, 1-mile radius)
    2. Montgomery OneView flood hazard lookup
    3. Montgomery 311 service-request density (90-day window)
    4. 8-dimension scorer (weights sum to 100%)
    5. Gemini 2.5 Pro recommendations (or mock if no key)

    Response shape matches exactly what `demo_3.html` expects.
    """
    pid = parcel_id.upper()
    if pid not in PARCEL_MAP:
        raise HTTPException(
            status_code=404,
            detail=f"Parcel '{parcel_id}' not found. Valid IDs: A, B, C",
        )

    parcel = HERO_PARCELS[PARCEL_MAP[pid]]
    return _run_pipeline(parcel)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4 — Full scoring pipeline for arbitrary coordinates
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/parcels/custom",
    summary="Score a custom parcel",
    tags=["Parcels"],
)
def score_custom_parcel(body: CustomParcelRequest):
    """
    Runs the complete RISE pipeline for any parcel defined by lat/lon.

    The pipeline auto-detects the nearest anchor, then scores all
    8 dimensions exactly as it does for the hero parcels.

    Useful for planners exploring non-hero parcels from the GIS catalogue.
    """
    nearest_anchor, min_dist = _nearest_anchor(body.lat, body.lon)

    # Build a parcel dict compatible with rise_selector_v3 internals
    parcel = {
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

    return _run_pipeline(parcel)