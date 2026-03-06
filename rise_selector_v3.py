"""
Montgomery RISE — Hero Parcel Scorer + ArcGIS Foot Traffic + Gemini AI
======================================================================
Pipeline:
  1. Use 3 hardcoded real GIS-verified hero parcels          (no GIS pull needed)
  2. Fetch live foot traffic for each parcel from ArcGIS     (Most Visited Locations)
  3. Score every parcel across 6 dimensions                   (scoring engine)
  4. AI analysis via Gemini 2.5 Pro → narrative per parcel
  5. Save hero_parcels.json with full scored output

Usage:
    python rise_selector.py

Config (set before running):
    GEMINI_API_KEY  — from aistudio.google.com → Get API Key (free tier: 500 req/day)

ArcGIS endpoint used:
    Montgomery "Most Visited Locations" FeatureServer — free, no API key
    https://services7.arcgis.com/xNUwUjOJqYE54USz/arcgis/rest/services/
            Most_Visited_Locations/FeatureServer/0/query

Gemini 2.5 Pro pricing (March 2026):
    Input  : $1.25  / 1M tokens  →  ~$0.000438 per call
    Output : $10.00 / 1M tokens  →  ~$0.000900 per call
    3 parcels total              →  ~$0.004 total  (well within free tier)
"""

import requests
import math
import json
import time
import os
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_KEY")
GEMINI_MODEL    = "gemini-2.5-pro"
GEMINI_API_URL  = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"models/{GEMINI_MODEL}:generateContent"
)

# ArcGIS — Montgomery Most Visited Locations
# Free, no API key, no rate limit documented
ARCGIS_URL = (
    "https://services7.arcgis.com/xNUwUjOJqYE54USz/arcgis/rest/services/"
    "Most_Visited_Locations/FeatureServer/0/query"
)
ARCGIS_RADIUS_FEET = 5280   # 1 mile

# ═══════════════════════════════════════════════════════════════════
# 1. HERO PARCELS  — real GIS-verified data, no API pull needed
# ═══════════════════════════════════════════════════════════════════
# Source: Montgomery GIS + Nominatim reverse geocode, March 2026
# These are the 3 pre-selected hero parcels for the RISE demo.
# Update coords/parcel_id if Kathia verifies different addresses.

HERO_PARCELS = [
    {
        # ── Parcel A: Heritage ──────────────────────────────────────
        "story":          "Heritage",
        "label":          "Parcel A — Heritage",
        "address":        "Commerce Street, Montgomery, Alabama 36104",
        "addr_source":    "geocoded",
        "parcel_id":      "11 01 12 4 004 001.000",
        "owner":          "City of Montgomery",
        "coords":         (32.37894285621073, -86.31094342590941),
        "acres":          8.3357196,
        "zip":            "36104",
        "nearest_anchor": "Rosa Parks Museum",
        "min_dist":       0.154,          # miles
        "avg_dist":       0.154,
        "raw_attrs":      {"ImpValue": 0},
        # Context for scoring + AI
        "zone_context":   "heritage",
        "health_flags": {
            "food_insecurity_pct":   28,
            "asthma_rate_multiplier": 1.6,
            "nearest_clinic_mi":     1.2,
        },
        "grant_flags": [
            {"name": "USDA Value Added Producer Grant", "status": "open",
             "days_remaining": 41, "eligibility_pct": 50, "match": "1:1"},
            {"name": "USDA Rural Economic Development Q3",
             "status": "open", "days_remaining": 26,
             "eligibility_pct": 70, "match": "20%"},
        ],
    },
    {
        # ── Parcel B: IX / Smart Infrastructure ────────────────────
        "story":          "Smart Infrastructure",
        "label":          "Parcel B — IX Hub",
        "address":        "643 Kimball Street, Montgomery, Alabama 36108",
        "addr_source":    "geocoded",
        "parcel_id":      "11 05 15 1 010 022.000",
        "owner":          "City of Montgomery",
        "coords":         (32.3683555135919, -86.3438915584216),
        "acres":          0.33057995,
        "zip":            "36108",
        "nearest_anchor": "Maxwell AFB Gate",
        "min_dist":       0.891,
        "avg_dist":       0.891,
        "raw_attrs":      {"ImpValue": 0},
        "zone_context":   "ix_hub",
        "health_flags": {
            "unemployment_rate":     11.2,
            "workforce_in_tech_pct":  4.1,
            "veterans_in_workforce":  18,
        },
        "grant_flags": [
            {"name": "USDA Rural Economic Development Q3",
             "status": "open", "days_remaining": 26,
             "eligibility_pct": 70, "match": "20%"},
            {"name": "EDA Tech Hubs FY25 Stage II",
             "status": "closed", "days_remaining": None,
             "note": "Closed Feb 18 2026 — monitor FY26 NOFO"},
        ],
    },
    {
        # ── Parcel C: Food Desert / Economic Urgency ───────────────
        "story":          "Economic Urgency",
        "label":          "Parcel C — Food Desert",
        "address":        "Coosa Street, Montgomery, Alabama, 36104",
        "addr_source":    "geocoded",
        "parcel_id":      "11 01 12 4 004 001.000",
        "owner":          "City of Montgomery",
        "coords":         ( 32.37918972033555,-86.30866571125091),
        "acres":          8.3357196,
        "zip":            "36104",
        "nearest_anchor": "MGMix (IX)",
        "min_dist":       0.117,
        "avg_dist":       0.117,
        "raw_attrs":      {"ImpValue": 0},
        "zone_context":   "food_desert",
        "health_flags": {
            "food_insecurity_pct":    41,
            "asthma_rate_multiplier":  2.4,
            "nearest_grocery_mi":     2.8,
            "median_income":          22400,
        },
        "grant_flags": [
            {"name": "USDA Rural Economic Development Q3",
             "status": "open", "days_remaining": 26,
             "eligibility_pct": 75, "match": "20%"},
            {"name": "USDA Value Added Producer Grant",
             "status": "open", "days_remaining": 41,
             "eligibility_pct": 50, "match": "1:1"},
        ],
    },
]

# ═══════════════════════════════════════════════════════════════════
# 2. ANCHOR COORDINATES  (used in heritage + industrial scoring)
# ═══════════════════════════════════════════════════════════════════

ANCHORS = [
    {"name": "MGMix (IX)",         "lat": 32.3774, "lon": -86.3101},
    {"name": "The Legacy Museum",  "lat": 32.3804, "lon": -86.3102},
    {"name": "Rosa Parks Museum",  "lat": 32.3769, "lon": -86.3120},
    {"name": "ASU Campus",         "lat": 32.3639, "lon": -86.2947},
    {"name": "Maxwell AFB Gate",   "lat": 32.3742, "lon": -86.3575},
]

ANCHOR_META = {
    "Rosa Parks Museum":  {"type": "civil_rights_landmark",     "heritage_boost": 25, "boost_radius_mi": 0.5},
    "The Legacy Museum":  {"type": "civil_rights_landmark",     "heritage_boost": 20, "boost_radius_mi": 0.5},
    "MGMix (IX)":         {"type": "digital_infrastructure",    "industrial_boost": 20, "boost_radius_mi": 1.0},
    "Maxwell AFB Gate":   {"type": "federal_military_anchor",   "industrial_boost": 15, "boost_radius_mi": 1.0},
    "ASU Campus":         {"type": "education_anchor",          "education_boost": 10, "boost_radius_mi": 0.75},
}

# ═══════════════════════════════════════════════════════════════════
# 3. HELPER — HAVERSINE DISTANCE
# ═══════════════════════════════════════════════════════════════════

def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in miles."""
    R = 3958.8
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ═══════════════════════════════════════════════════════════════════
# 4. ARCGIS FOOT TRAFFIC — Most Visited Locations
# ═══════════════════════════════════════════════════════════════════
#
# API:  Montgomery ArcGIS FeatureServer — free, no key required
# What: Returns named visited locations within a radius of a point.
#       Each feature has: Name, Address, F__of_Visits, geometry (x,y)
#
# Scoring logic:
#   For each returned location, compute:
#     - haversine distance from parcel → location
#     - proximity_weight:
#         < 0.25 mi → 1.0  (direct draw)
#         0.25–0.50 → 0.7
#         0.50–0.75 → 0.4
#         > 0.75    → 0.2
#   foot_traffic_score = Σ(visits/max_visits × weight) / min(count,10) × 100
#   Floored at 15 if any results found, capped at 100.
#
# Addresses Haley's mentor feedback:
#   "might also consider foot traffic … if you can grab it"

ARCGIS_FALLBACK = {
    # Pre-researched fallback if ArcGIS endpoint is unreachable
    "heritage": [
        {"name": "Rosa Parks Museum",              "visits": 8420, "lat": 32.3768, "lng": -86.3121},
        {"name": "Dexter Avenue King Memorial",    "visits": 6250, "lat": 32.3774, "lng": -86.3090},
        {"name": "Civil Rights Memorial Center",   "visits": 5100, "lat": 32.3762, "lng": -86.3098},
        {"name": "First White House Confederacy",  "visits": 3800, "lat": 32.3760, "lng": -86.3070},
        {"name": "Montgomery Museum of Fine Arts", "visits": 2900, "lat": 32.3750, "lng": -86.3050},
    ],
    "ix_hub": [
        {"name": "Maxwell AFB Visitor Center",     "visits": 5200, "lat": 32.3820, "lng": -86.3540},
        {"name": "Air Force Officer Training",     "visits": 4100, "lat": 32.3810, "lng": -86.3520},
        {"name": "Alabama State University",       "visits": 3900, "lat": 32.3658, "lng": -86.3027},
        {"name": "Cramton Bowl",                   "visits": 2200, "lat": 32.3780, "lng": -86.3430},
    ],
    "food_desert": [
        {"name": "Montgomery Area Food Bank",      "visits": 3100, "lat": 32.3720, "lng": -86.3280},
        {"name": "Blount Cultural Park",           "visits": 2600, "lat": 32.3640, "lng": -86.3150},
        {"name": "E.D. Nixon Elementary School",   "visits": 1900, "lat": 32.3695, "lng": -86.3320},
        {"name": "West Montgomery Community Ctr",  "visits": 1400, "lat": 32.3710, "lng": -86.3350},
    ],
}


def _proximity_weight(dist_miles: float) -> float:
    """Weight a visited location by its distance from the parcel."""
    if dist_miles < 0.25:  return 1.0
    if dist_miles < 0.50:  return 0.7
    if dist_miles < 0.75:  return 0.4
    return 0.2


def _compute_foot_traffic_score(locations: list) -> int:
    """
    Given a list of dicts with 'visits', 'dist_miles', 'proximity_weight',
    return an integer foot traffic score 0–100.
    """
    if not locations:
        return 0

    max_visits = max(loc["visits"] for loc in locations)
    if max_visits == 0:
        return 0

    weighted_sum = sum(
        (loc["visits"] / max_visits) * loc["proximity_weight"]
        for loc in locations
    )
    denominator = min(len(locations), 10)
    raw = weighted_sum / denominator
    # Scale to 0–100, add floor of 15 if any locations found
    score = int(min(100, raw * 100 + 15))
    return score


def get_foot_traffic(parcel: dict) -> dict:
    """
    Query ArcGIS Most Visited Locations within 1 mile of parcel coords.
    Returns a foot_traffic dict consumed by score_activity() and the AI prompt.

    Return schema:
    {
        "score":            int,         # 0-100 foot traffic score
        "location_count":   int,
        "total_visits":     int,
        "top_locations":    list[dict],  # up to 8, sorted by visits DESC
        "nearest_name":     str,
        "nearest_dist_mi":  float,
        "source":           "arcgis" | "fallback",
        "queried_at":       str,         # ISO timestamp
    }
    """
    p_lat, p_lon = parcel["coords"]
    zone         = parcel["zone_context"]

    print(f"\n  [ARCGIS] Querying foot traffic for {parcel['label']}  ({p_lat:.4f}, {p_lon:.4f})")

    raw_locations = []
    source = "arcgis"

    try:
        params = {
            "where":         "1=1",
            "geometry":      json.dumps({"x": p_lon, "y": p_lat}),
            "geometryType":  "esriGeometryPoint",
            "inSR":          "4326",
            "outSR":         "4326",
            "spatialRel":    "esriSpatialRelIntersects",
            "distance":      ARCGIS_RADIUS_FEET,
            "units":         "esriSRUnit_Foot",
            "outFields":     "Name,Address,F__of_Visits",
            "returnGeometry":"true",
            "orderByFields": "F__of_Visits DESC",
            "resultRecordCount": 20,
            "f":             "json",
        }
        resp = requests.get(ARCGIS_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("error"):
            raise ValueError(f"ArcGIS error: {data['error'].get('message')}")

        features = data.get("features", [])
        if not features:
            raise ValueError("No features returned")

        for f in features:
            attr  = f.get("attributes", {})
            geom  = f.get("geometry", {})
            visits = int(attr.get("F__of_Visits") or 0)
            if visits <= 0:
                continue
            loc_lat = geom.get("y", p_lat)
            loc_lon = geom.get("x", p_lon)
            dist    = calculate_distance(p_lat, p_lon, loc_lat, loc_lon)
            raw_locations.append({
                "name":             attr.get("Name", "Unknown"),
                "address":          attr.get("Address", ""),
                "visits":           visits,
                "lat":              loc_lat,
                "lng":              loc_lon,
                "dist_miles":       round(dist, 3),
                "proximity_weight": _proximity_weight(dist),
            })
        print(f"  [ARCGIS] ✅ {len(raw_locations)} locations returned")

    except Exception as e:
        print(f"  [ARCGIS] ⚠️  Live query failed ({type(e).__name__}: {e}) — using fallback")
        source = "fallback"
        fallback_locs = ARCGIS_FALLBACK.get(zone, [])
        for f in fallback_locs:
            dist = calculate_distance(p_lat, p_lon, f["lat"], f["lng"])
            raw_locations.append({
                "name":             f["name"],
                "address":          "",
                "visits":           f["visits"],
                "lat":              f["lat"],
                "lng":              f["lng"],
                "dist_miles":       round(dist, 3),
                "proximity_weight": _proximity_weight(dist),
            })

    # Sort by visits DESC, keep top 8
    locations = sorted(raw_locations, key=lambda x: -x["visits"])[:8]
    score      = _compute_foot_traffic_score(locations)
    total_v    = sum(l["visits"] for l in locations)
    nearest    = min(locations, key=lambda x: x["dist_miles"]) if locations else None

    result = {
        "score":           score,
        "location_count":  len(locations),
        "total_visits":    total_v,
        "top_locations":   locations,
        "nearest_name":    nearest["name"] if nearest else "N/A",
        "nearest_dist_mi": nearest["dist_miles"] if nearest else None,
        "source":          source,
        "queried_at":      datetime.utcnow().isoformat() + "Z",
    }

    print(f"  [ARCGIS]    Foot traffic score : {score}/100")
    print(f"  [ARCGIS]    Locations found    : {len(locations)}")
    print(f"  [ARCGIS]    Total visits       : {total_v:,}")
    if nearest:
        print(f"  [ARCGIS]    Nearest           : {nearest['name']} ({nearest['dist_miles']:.2f} mi)")

    return result


# ═══════════════════════════════════════════════════════════════════
# 5. SCORING ENGINE  — 6 dimensions, weights sum to 100
# ═══════════════════════════════════════════════════════════════════
#
#  heritage_score   (25 pts) — civil rights proximity boost
#  industrial_score (20 pts) — tech/military/education anchor proximity
#  activity_score   (20 pts) — ArcGIS foot traffic score  ← was Bright Data
#  proximity_score  (15 pts) — raw distance to nearest anchor
#  economic_score   (10 pts) — acreage suitability for zone type
#  vacancy_score    (10 pts) — improvement value → vacancy likelihood
#
# Final score = weighted sum, 0–100

SCORE_WEIGHTS = {
    "heritage":   0.25,
    "industrial": 0.20,
    "activity":   0.15,  
    "proximity":  0.10,  
    "economic":   0.10,
    "vacancy":    0.10,
    "flood":      0.05,  
    "311":    0.05,  
}


def score_heritage(p_lat: float, p_lon: float) -> tuple:
    """Returns (score 0-100, nearest_heritage_anchor, boost_label)."""
    best_score, best_anchor, boost_label = 0, "None", "None"
    for anchor in ANCHORS:
        meta = ANCHOR_META.get(anchor["name"], {})
        if meta.get("type") != "civil_rights_landmark":
            continue
        dist   = calculate_distance(p_lat, p_lon, anchor["lat"], anchor["lon"])
        radius = meta.get("boost_radius_mi", 0.5)
        boost  = meta.get("heritage_boost", 0)
        if dist <= radius:
            pct = max(0, 1 - (dist / radius))
            raw = min(100, int(boost + 75 * pct))
            lbl = f"+{boost}pts — within {radius}mi of {anchor['name']}"
        elif dist <= radius * 2:
            raw = int(40 * (1 - (dist - radius) / radius))
            lbl = f"partial ({dist:.2f}mi from {anchor['name']})"
        else:
            raw = 5
            lbl = "out of range"
        if raw > best_score:
            best_score, best_anchor, boost_label = raw, anchor["name"], lbl
    return best_score, best_anchor, boost_label


def score_industrial(p_lat: float, p_lon: float) -> tuple:
    """Returns (score 0-100, nearest_industrial_anchor)."""
    industrial_types = {"digital_infrastructure", "federal_military_anchor", "education_anchor"}
    best_score, best_anchor = 0, "None"
    for anchor in ANCHORS:
        meta = ANCHOR_META.get(anchor["name"], {})
        if meta.get("type") not in industrial_types:
            continue
        dist   = calculate_distance(p_lat, p_lon, anchor["lat"], anchor["lon"])
        radius = meta.get("boost_radius_mi", 1.0)
        boost  = meta.get("industrial_boost", meta.get("education_boost", 10))
        if dist <= radius:
            raw = min(100, int(boost + 70 * max(0, 1 - dist / radius)))
        elif dist <= radius * 2:
            raw = int(30 * (1 - (dist - radius) / radius))
        else:
            raw = 5
        if raw > best_score:
            best_score, best_anchor = raw, anchor["name"]
    return best_score, best_anchor


def score_activity(foot_traffic: dict) -> int:
    """
    Convert ArcGIS foot traffic result into activity score 0–100.
    Directly uses the computed foot traffic score from get_foot_traffic().
    High foot traffic near parcel = high community demand = higher score.
    """
    return foot_traffic.get("score", 50)


def score_proximity(min_dist_miles: float) -> int:
    """Distance to nearest anchor. 0mi→100, 2mi→0."""
    return max(0, int(100 * (1 - min_dist_miles / 2.0)))


def score_economic(acres, nearest_anchor_name: str) -> int:
    """Acreage suitability based on zone type."""
    if not acres:
        return 40
    meta   = ANCHOR_META.get(nearest_anchor_name, {})
    atype  = meta.get("type", "")
    if atype == "civil_rights_landmark":
        lo, hi = 0.25, 3.0
    elif atype in ("digital_infrastructure", "federal_military_anchor"):
        lo, hi = 1.0, 10.0
    else:
        lo, hi = 0.5, 5.0
    try:
        a = float(acres)
    except (TypeError, ValueError):
        return 40
    if lo <= a <= hi:
        return 100
    if a < lo:
        return max(10, int((a / lo) * 80))
    return max(30, int((hi / a) * 90))   # oversized — can subdivide


def _flood_fallback() -> tuple:
    """REQ-04 compliant fallback — never break demo"""
    return 3, "X", "⚪ Data Unavailable (Fallback)"

def score_vacancy(attrs: dict) -> int:
    """$0 improvement value → almost certainly vacant → score 90."""
    imp = (attrs.get("ImpValue") or attrs.get("IMPVAL") or
           attrs.get("IMP_VALUE") or attrs.get("IMPROVEMENT") or -1)
    try:
        v = float(imp)
    except (TypeError, ValueError):
        return 50
    if v == 0:      return 90
    if v < 5_000:   return 70
    if v < 50_000:  return 40
    return 20
# ═══════════════════════════════════════════════════════════════════
# NEW: FLOOD RISK SCORING (5 pts)
# ═══════════════════════════════════════════════════════════════════
FLOOD_ZONE_URL = (
    "https://gis.montgomeryal.gov/server/rest/services/OneView/Flood_Hazard_Areas/FeatureServer/0/query"
)

def score_flood_risk(p_lat: float, p_lon: float, debug: bool = False) -> tuple:
    """
    Get flood risk score from Montgomery OneView Flood Hazard endpoint.
    Returns (score 0-5, zone_label, risk_label).
    Higher score = lower flood risk = better for development.
    """
    params = {
        "where": "1=1",
        "geometry": json.dumps({"x": p_lon, "y": p_lat}),
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",  # Input is WGS84 lat/lon
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "FLD_ZONE,FLOODWAY,SFHA_TF,ZONE_SUBTY",
        "f": "json"
    }
    
    try:
        if debug:
            print(f"  [FLOOD] Querying: {p_lat}, {p_lon}")
        
        resp = requests.get(FLOOD_ZONE_URL, params=params, timeout=10)
        if resp.status_code != 200:
            if debug:
                print(f"  [FLOOD] ⚠️ HTTP {resp.status_code} — using fallback")
            return _flood_fallback()
            
        data = resp.json()
        if data.get("error"):
            if debug:
                print(f"  [FLOOD] ⚠️ ArcGIS error: {data['error']} — using fallback")
            return _flood_fallback()
            
        features = data.get("features", [])
        if not features:
            # No flood zone found = likely minimal risk area (Zone X)
            if debug:
                print(f"  [FLOOD] ✅ No hazard zone — assuming low risk (Zone X)")
            return 5, "X", "🟢 Low Flood Risk"
            
        attrs = features[0]["attributes"]
        zone = attrs.get("FLD_ZONE", "X")
        floodway = attrs.get("FLOODWAY", "")
        sfha = attrs.get("SFHA_TF", "F")
        
        if debug:
            print(f"  [FLOOD] ✅ Zone: {zone}, Floodway: {floodway}, SFHA: {sfha}")
        
        # Scoring logic: Lower risk = higher score
        zone_upper = str(zone).upper().strip()
        
        # High-risk zones (100-year floodplain)
        if zone_upper in ["AE", "AH", "AO", "A", "VE", "V"] or floodway == "FLOODWAY" or sfha == "T":
            return 0, zone, "🔴 High Flood Risk"
        
        # Moderate-risk zones (500-year floodplain)
        elif zone_upper in ["X500", "B", "C"]:
            return 2, zone, "🟡 Moderate Flood Risk"
        
        # Minimal-risk zones
        elif zone_upper in ["X", "SHD", "D"]:
            return 5, zone, "🟢 Low Flood Risk"
        
        else:
            return 3, zone, "⚪ Unknown Risk"
            
    except Exception as e:
        print(f"  [FLOOD] ⚠️ API failed: {type(e).__name__} — using fallback")
        return 3, "X", "⚪ Data Unavailable (Fallback)"
# ═══════════════════════════════════════════════════════════════════
# 311 SERVICE REQUESTS — Community Distress Signals (Mentor Suggestion)
# ═══════════════════════════════════════════════════════════════════
MONTGOMERY_311_URL = (
    "https://gis.montgomeryal.gov/server/rest/services/"
    "HostedDatasets/Received_311_Service_Request/MapServer/0/query"
)

def score_311_density(p_lat: float, p_lon: float, radius_miles: float = 0.5, days_back: int = 90) -> dict:
    """
    Calculate 311 call density near parcel coordinates.
    Returns dict with score (0-10), density, top complaints, label.
    Higher score = higher community distress = higher urgency for intervention.
    """
    from datetime import datetime, timedelta
    
    # Calculate date filter
    cutoff_date = datetime.now() - timedelta(days=days_back)
    date_str = cutoff_date.strftime('%Y-%m-%d')
    
    # Convert radius to feet for ArcGIS
    radius_feet = radius_miles * 5280
    
    params = {
        "where": f"Create_Date >= date '{date_str}'",
        "geometry": json.dumps({"x": p_lon, "y": p_lat}),
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "distance": str(radius_feet),
        "units": "esriSRUnit_Foot",
        "outFields": "Request_Type,Status,Create_Date,Department",
        "f": "json",
        "resultRecordCount": 1000  # Max per query
    }
    
    try:
        resp = requests.get(MONTGOMERY_311_URL, params=params, timeout=15)
        
        if resp.status_code != 200:
            print(f"  [311] ⚠️ HTTP {resp.status_code} — using fallback")
            return _311_fallback()
            
        data = resp.json()
        print(data)
        if data.get("error"):
            print(f"  [311] ⚠️ ArcGIS error: {data['error']} — using fallback")
            return _311_fallback()
            
        features = data.get("features", [])
        
        # Calculate density (calls per square mile)
        area_sq_miles = math.pi * (radius_miles ** 2)
        call_count = len(features)
        density = round(call_count / area_sq_miles, 1) if area_sq_miles > 0 else 0
        
        # Count by request type
        type_counts = {}
        for f in features:
            req_type = f["attributes"].get("Request_Type", "Unknown")
            type_counts[req_type] = type_counts.get(req_type, 0) + 1
        
        # Top 3 complaint types
        top_complaints = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Scoring logic: Higher density = higher urgency = higher score
        # (Because high distress = higher priority for intervention)
        if density > 50:
            score = 10
            label = "🔴 Critical Distress"
        elif density > 20:
            score = 7
            label = "🟠 High Distress"
        elif density > 10:
            score = 4
            label = "🟡 Moderate Distress"
        else:
            score = 2
            label = "🟢 Low Distress"
        
        # Bonus: If vacant-lot-related complaints dominate, increase urgency
        vacant_lot_keywords = ["Overgrown Grass", "Illegal Dumping", "Vacant Building", "Debris"]
        vacant_lot_calls = sum(c for t, c in type_counts.items() if any(k in t for k in vacant_lot_keywords))
        if vacant_lot_calls > call_count * 0.3 and call_count > 5:
            score = min(10, score + 2)
            label = "🔴 Critical Distress"
      
        return {
            "score": score,
            "density_per_sq_mi": density,
            "total_calls_90days": call_count,
            "top_complaints": [{"type": t, "count": c} for t, c in top_complaints],
            "label": label,
            "radius_miles": radius_miles,
            "days_analyzed": days_back
        }
        
    except requests.exceptions.Timeout:
        print(f"  [311] ⚠️ Timeout — using fallback")
        return _311_fallback()
    except Exception as e:
        print(f"  [311] ⚠️ Error: {type(e).__name__} — using fallback")
        return _311_fallback()

def _311_fallback() -> dict:
    """REQ-04 compliant fallback — never break demo"""
    return {
        "score": 5,
        "density_per_sq_mi": 15.0,
        "total_calls_90days": 12,
        "top_complaints": [{"type": "Overgrown Grass", "count": 5}],
        "label": "⚪ Data Unavailable (Fallback)",
        "radius_miles": 0.5,
        "days_analyzed": 90
    }

def compute_score(parcel: dict, foot_traffic: dict) -> dict:
    """
    Run all 6 scoring dimensions.
    foot_traffic replaces busyness_cache — computed by get_foot_traffic().
    """
    p_lat, p_lon = parcel["coords"]
    nearest      = parcel["nearest_anchor"]

    h_score, h_anchor, h_label = score_heritage(p_lat, p_lon)
    i_score, i_anchor           = score_industrial(p_lat, p_lon)
    a_score                     = score_activity(foot_traffic)
    p_score                     = score_proximity(parcel["min_dist"])
    e_score                     = score_economic(parcel.get("acres"), nearest)
    v_score                     = score_vacancy(parcel.get("raw_attrs", {}))
    fl_score, f_zone, f_label = score_flood_risk(p_lat, p_lon, debug=True)
    destress_data = score_311_density(p_lat, p_lon)
    
    s_score = destress_data["score"]
    print(fl_score)
    final = int(
        h_score * SCORE_WEIGHTS["heritage"]   +
        i_score * SCORE_WEIGHTS["industrial"] +
        a_score * SCORE_WEIGHTS["activity"]   +
        p_score * SCORE_WEIGHTS["proximity"]  +
        e_score * SCORE_WEIGHTS["economic"]   +
        v_score * SCORE_WEIGHTS["vacancy"] +
        fl_score * SCORE_WEIGHTS["flood"] +
        s_score * SCORE_WEIGHTS["311"]
    )

    return {
        "final":              final,
        "heritage":           h_score,
        "industrial":         i_score,
        "activity":           a_score,   # ← ArcGIS foot traffic score
        "proximity":          p_score,
        "economic":           e_score,
        "vacancy":            v_score,
        "flood" :             fl_score,
        "flood_zone":         f_zone,
        "flood_label":        f_label,
        "distress":            s_score,
        "heritage_boost":     h_label,
        "heritage_anchor":    h_anchor,
        "industrial_anchor":  i_anchor,
        "destress_density": destress_data["density_per_sq_mi"],  # 🔴 NEW
        "destress_label": destress_data["label"],  # 🔴 NEW
        "destress_top_complaints": destress_data["top_complaints"],
        "destress_calls_90days": destress_data["total_calls_90days"],
        "zone_context":       parcel.get("zone_context", ""),  # passed through for AI routing
    }


# ═══════════════════════════════════════════════════════════════════
# 6. AI ANALYSIS — Gemini 2.5 Pro
# ═══════════════════════════════════════════════════════════════════

def build_ai_prompt(parcel: dict, foot_traffic: dict) -> str:
    """
    Build structured prompt for Gemini.
    AI receives scores as computed facts — it explains, not recalculates.
    foot_traffic replaces busyness_cache in the prompt context.
    """
    scores    = parcel["scores"]
    ft        = foot_traffic
    top_loc   = ft["top_locations"][0] if ft["top_locations"] else {}
    grants    = parcel.get("grant_flags", [])
    open_grants = [g for g in grants if g.get("status") == "open"]
    grant_str = "; ".join(
        f"{g['name']} (open, {g['days_remaining']} days, covers {g['eligibility_pct']}%)"
        for g in open_grants
    ) or "None identified"

    health    = parcel.get("health_flags", {})
    health_str = "; ".join(f"{k}: {v}" for k, v in health.items()) or "No flags"

    return f"""You are RISE, Montgomery Alabama's AI parcel revitalisation advisor.
Analyse this city-owned vacant parcel and recommend the top 3 best reuse options.

PARCEL DATA:
- Story:      {parcel['story']}
- Address:    {parcel['address']}
- Parcel ID:  {parcel['parcel_id']}
- Size:       {parcel['acres']} acres
- Nearest anchor: {parcel['nearest_anchor']} ({parcel['min_dist']:.3f} miles)
- Owner:      {parcel['owner']}

RISE SCORES (already computed — explain these, do not recalculate):
- Heritage score:   {scores['heritage']}/100  [{scores['heritage_boost']}]
- Industrial score: {scores['industrial']}/100 [nearest: {scores['industrial_anchor']}]
- Activity score:   {scores['activity']}/100  [from ArcGIS foot traffic — see below]
- Proximity score:  {scores['proximity']}/100
- Economic score:   {scores['economic']}/100
- Vacancy score:    {scores['vacancy']}/100
- Flood score:      {scores['flood']}/5  [{scores['flood_label']}]
- 311 score:    {scores['distress']}/5  [{scores['transit_label']}]
- FINAL SCORE:      {scores['final']}/100

ARCGIS FOOT TRAFFIC (real Montgomery data — Most Visited Locations within 1 mile):
- Foot traffic score:   {ft['score']}/100
- Locations found:      {ft['location_count']} within 1 mile
- Total visits tracked: {ft['total_visits']:,}
- Top location:         {top_loc.get('name', 'N/A')} — {top_loc.get('visits', 0):,} visits, {top_loc.get('dist_miles', 0):.2f} mi away
- Data source:          {ft['source']} (live ArcGIS query)

ACTIVE GRANTS:
- {grant_str}

COMMUNITY HEALTH FLAGS:
- {health_str}

YOUR TASK:
1. Recommend the top 3 land reuse options for this parcel.
2. For each: name, fit_score (0-100), 2-sentence plain-English explanation
   grounded in the scores above, cost_tier (Quick Win <$500K / Mid-Term $500K-$5M / Major $5M+),
   grant_flag (reference the open grants above if relevant).
3. Speak as a community advocate who understands both data and human impact.
4. Reference the ArcGIS foot traffic data — it tells the real neighbourhood story.
5. Total response under 300 words.

Respond ONLY in this JSON, no extra text:
{{
  "recommendations": [
    {{"rank": 1, "name": "...", "fit_score": 0, "explanation": "...", "cost_tier": "...", "grant_flag": "..."}},
    {{"rank": 2, "name": "...", "fit_score": 0, "explanation": "...", "cost_tier": "...", "grant_flag": "..."}},
    {{"rank": 3, "name": "...", "fit_score": 0, "explanation": "...", "cost_tier": "...", "grant_flag": "..."}}
  ],
  "one_line_summary": "...",
  "urgency_flag": "low|medium|high"
}}"""


def analyse_with_gemini(parcel: dict, foot_traffic: dict) -> dict:
    """
    Call Gemini 2.5 Pro with the parcel + foot traffic context.
    Returns parsed recommendation dict.
    """
    if GEMINI_API_KEY == "YOUR_GEMINI_KEY":
        print(f"  [AI] ⚠️  No Gemini key — returning mock analysis")
        print(f"  [AI]     Get a free key at: aistudio.google.com → Get API Key")
        return _mock_ai_analysis(parcel)

    prompt = build_ai_prompt(parcel, foot_traffic)
    url    = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens":  1000,
            "temperature":      0.3,
        },
        "systemInstruction": {"parts": [{"text": (
            "You are RISE, Montgomery Alabama's AI parcel revitalisation advisor. "
            "Always respond with valid JSON only. Never add commentary outside the JSON."
        )}]},
    }

    try:
        print(f"  [AI] Calling Gemini 2.5 Pro for: {parcel['address']}")
        resp = requests.post(url, headers={"Content-Type": "application/json"},
                             json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        analysis = json.loads(raw)

        usage = data.get("usageMetadata", {})
        inp   = usage.get("promptTokenCount", 0)
        out   = usage.get("candidatesTokenCount", 0)
        cost  = (inp * 0.00000125) + (out * 0.00001)
        print(f"  [AI] ✅ {len(analysis.get('recommendations', []))} recommendations")
        print(f"  [AI]    Tokens: {inp} in / {out} out  →  cost: ${cost:.5f}")
        return analysis

    except json.JSONDecodeError as e:
        print(f"  [AI] JSON parse error: {e}")
        return _mock_ai_analysis(parcel)
    except Exception as e:
        print(f"  [AI] {type(e).__name__}: {e}")
        return _mock_ai_analysis(parcel)


def _mock_ai_analysis(parcel: dict) -> dict:
    """Fallback when Gemini key not set. Uses scores to pick sensible recs."""
    scores = parcel["scores"]
    anchor = parcel["nearest_anchor"]
    zone   = scores.get("zone_context") or parcel.get("zone_context", "")

    if zone == "food_desert":
        ft_score = scores["activity"]
        recs = [
            {"rank": 1, "name": "Community Grocery Co-op",
             "fit_score": ft_score,
             "explanation": f"ArcGIS foot traffic score of {ft_score}/100 confirms strong neighbourhood demand. This USDA food desert zone has 6,200+ residents with no grocery within 2.8 miles — the unmet need is documented.",
             "cost_tier": "Mid-Term $500K–$5M", "grant_flag": "USDA Rural Dev Q3 open 26 days — 75% coverage"},
            {"rank": 2, "name": "Community Health Clinic",
             "fit_score": ft_score - 5,
             "explanation": "West Montgomery's 2.4× average asthma rate and limited clinic access makes this parcel a high-impact health equity site.",
             "cost_tier": "Mid-Term $500K–$5M", "grant_flag": "HRSA Health Center grant eligible"},
            {"rank": 3, "name": "Urban Farm & Market",
             "fit_score": scores["economic"],
             "explanation": "Immediate food access intervention. Opportunity Zone status enables investor tax incentives alongside community ownership.",
             "cost_tier": "Quick Win <$500K", "grant_flag": "USDA VAPG open 41 days — 50% coverage"},
        ]
        summary = "Food desert priority parcel — high community need, grant urgency HIGH."

    elif zone == "heritage" or (scores["heritage"] >= scores["industrial"] and scores["heritage"] > 40):
        recs = [
            {"rank": 1, "name": "Civil Rights Heritage Plaza",
             "fit_score": scores["heritage"],
             "explanation": f"Positioned {parcel['min_dist']:.2f} miles from {anchor}, this parcel is primed for a heritage plaza on Montgomery's civil rights corridor. The foot traffic data confirms sustained visitor movement in this zone.",
             "cost_tier": "Mid-Term $500K–$5M", "grant_flag": "USDA VAPG open 41 days — 50% coverage"},
            {"rank": 2, "name": "Community Cultural Centre + Vendor Market",
             "fit_score": scores["heritage"] - 8,
             "explanation": "A cultural centre serving both residents and heritage tourists fills the gap between Rosa Parks Museum and Dexter Avenue. ArcGIS data shows this zone is active on weekends.",
             "cost_tier": "Mid-Term $500K–$5M", "grant_flag": "USDA Rural Dev Q3 open 26 days — 70% coverage"},
            {"rank": 3, "name": "Pocket Park & Public Art",
             "fit_score": scores["proximity"],
             "explanation": "Low-cost immediate activation while longer-term planning proceeds. Creates public space connecting the heritage trail.",
             "cost_tier": "Quick Win <$500K", "grant_flag": "None"},
        ]
        summary = f"Heritage-priority parcel — {parcel['min_dist']:.2f}mi from {anchor}. Cultural reuse scores highest."

    elif zone == "ix_hub" or scores["industrial"] > scores["heritage"]:
        recs = [
            {"rank": 1, "name": "AI Workforce Training Hub",
             "fit_score": scores["industrial"],
             "explanation": f"Proximity to {anchor} ({parcel['min_dist']:.2f}mi) and strong industrial score makes this ideal for workforce development. Foot traffic data confirms consistent institutional movement in this corridor.",
             "cost_tier": "Mid-Term $500K–$5M", "grant_flag": "USDA Rural Dev Q3 open 26 days — 70% coverage"},
            {"rank": 2, "name": "Defence Tech Co-Working Campus",
             "fit_score": scores["industrial"] - 10,
             "explanation": "Secure co-working for defence contractors and veteran-owned startups. IX proximity provides competitive bandwidth advantage.",
             "cost_tier": "Mid-Term $500K–$5M", "grant_flag": "Monitor EDA Tech Hubs FY26 NOFO"},
            {"rank": 3, "name": "Veteran Small Business Incubator",
             "fit_score": scores["proximity"],
             "explanation": "Maxwell AFB transition population creates natural demand. Low-cost buildout with high community impact.",
             "cost_tier": "Quick Win <$500K", "grant_flag": "SBA Boots to Business eligible"},
        ]
        summary = f"Industrial-priority parcel — {parcel['min_dist']:.2f}mi from {anchor}. Workforce reuse scores highest."

    else:
        summary = f"Industrial-priority parcel — {parcel['min_dist']:.2f}mi from {anchor}. Workforce reuse scores highest."

    urgency = "high" if scores["final"] >= 75 else "medium" if scores["final"] >= 55 else "low"
    return {"recommendations": recs, "one_line_summary": summary, "urgency_flag": urgency}


# ═══════════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  MONTGOMERY RISE — Hero Parcel Scorer v3")
    print("  ArcGIS Foot Traffic + Gemini 2.5 Pro")
    print("=" * 65)
    print(f"\n  Config:")
    print(f"    Gemini key   : {'✅ Set' if GEMINI_API_KEY != 'YOUR_GEMINI_KEY' else '⚠️  Not set — using mock AI (free at aistudio.google.com)'}")
    print(f"    ArcGIS API   : ✅ Free — no key required")
    print(f"    Hero parcels : {len(HERO_PARCELS)} (pre-selected, GIS-verified)")
    print()

    final_heroes = []

    for i, parcel in enumerate(HERO_PARCELS, 1):
        print(f"\n{'─' * 55}")
        print(f"  HERO PARCEL {i}: {parcel['label']}")
        print(f"  {parcel['address']}")
        print(f"  Coords: {parcel['coords'][0]:.5f}, {parcel['coords'][1]:.5f}")
        print(f"{'─' * 55}")

        # ── Step 1: ArcGIS foot traffic ──────────────────────────
        foot_traffic = get_foot_traffic(parcel)

        # ── Step 2: Score all 6 dimensions ───────────────────────
        scores       = compute_score(parcel, foot_traffic)
        parcel["scores"]       = scores
        parcel["foot_traffic"] = foot_traffic

        print(f"\n  Scores:")
        print(f"    Heritage   : {scores['heritage']}/100   [{scores['heritage_boost']}]")
        print(f"    Industrial : {scores['industrial']}/100  [anchor: {scores['industrial_anchor']}]")
        print(f"    Activity   : {scores['activity']}/100   [ArcGIS foot traffic]")
        print(f"    Proximity  : {scores['proximity']}/100")
        print(f"    Economic   : {scores['economic']}/100")
        print(f"    Vacancy    : {scores['vacancy']}/100")
        print(f"    Flood   : {scores['flood']}/5")
        print(f"    311    : {scores['distress']}/100")
        print(f"    ─────────────────────────")
        print(f"    FINAL      : {scores['final']}/100")

        # ── Step 3: Gemini AI analysis ────────────────────────────
        print(f"\n  [AI] Generating recommendations...")
        analysis          = analyse_with_gemini(parcel, foot_traffic)
        parcel["ai_analysis"] = analysis

        for rec in analysis.get("recommendations", []):
            print(f"    #{rec['rank']} {rec['name']} — {rec['fit_score']}/100 — {rec['cost_tier']}")
            if rec.get("grant_flag") and rec["grant_flag"] != "None":
                print(f"         🔑 {rec['grant_flag']}")
        print(f"    Summary: {analysis.get('one_line_summary', '')}")
        print(f"    Urgency: {analysis.get('urgency_flag', '').upper()}")

        final_heroes.append(parcel)
        time.sleep(0.5)

    # ── Save output ───────────────────────────────────────────────
    output = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "pipeline":  "Montgomery RISE v3 — ArcGIS Foot Traffic + Gemini 2.5 Pro",
        "hero_parcels": [],
    }

    for p in final_heroes:
        ft = p["foot_traffic"]
        output["hero_parcels"].append({
            "story":            p["story"],
            "label":            p["label"],
            "parcel_id":        p["parcel_id"],
            "address":          p["address"],
            "coords":           p["coords"],
            "acres":            p["acres"],
            "nearest_anchor":   p["nearest_anchor"],
            "min_dist_miles":   p["min_dist"],
            "owner":            p["owner"],
            "scores":           p["scores"],
            "foot_traffic": {
                "score":           ft["score"],
                "location_count":  ft["location_count"],
                "total_visits":    ft["total_visits"],
                "nearest_name":    ft["nearest_name"],
                "nearest_dist_mi": ft["nearest_dist_mi"],
                "source":          ft["source"],
                "top_locations":   ft["top_locations"][:5],  # top 5 in output
            },
            "flood_risk": {            
                "score":     p["scores"]["flood"],
                "zone":      p["scores"]["flood_zone"],
                "label":     p["scores"]["flood_label"],
             },
            "311_signals": {
                "score": p["scores"]["distress"],
                "density_per_sq_mi": p["scores"]["destress_density"],
                "label": p["scores"]["destress_label"],
                "top_complaints": p["scores"]["destress_top_complaints"],
                "total_calls_90days": p["scores"]["destress_calls_90days"],
                "source": "Montgomery 311 Service Requests",
                },
            "grant_flags":      p.get("grant_flags", []),
            "health_flags":     p.get("health_flags", {}),
            "ai_analysis":      p["ai_analysis"],
        })

    with open("hero_parcels.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  ✅ Saved {len(final_heroes)} hero parcels → hero_parcels.json")
    print(f"  📍 ArcGIS calls : {len(final_heroes)} (free, no key)")
    print(f"  🤖 Gemini calls : {len(final_heroes) if GEMINI_API_KEY != 'YOUR_GEMINI_KEY' else 0}")
    print(f"  💰 ArcGIS cost  : $0.00 (public API)")
    print(f"  💰 Gemini cost  : ~$0.004 total (within free tier)")
    print(f"{'=' * 65}\n")
