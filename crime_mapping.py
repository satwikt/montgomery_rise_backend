"""
Montgomery RISE — Crime Mapping
================================
Standalone script — does NOT modify rise_selector.py.

Pipeline:
  1. Import parcel coordinates directly from rise_selector.py (read-only)
  2. Query Montgomery Crime Incidents ArcGIS endpoint for each parcel
  3. Aggregate crime counts by type, severity, and time
  4. Output crime_mapping.json with full results per parcel

Usage:
    python crime_mapping.py

API endpoint (free, no key):
    https://services3.arcgis.com/dty2kHktVXHrqO8i/ArcGIS/rest/services/Crime_Incidents/FeatureServer/0/query

Output:
    crime_mapping.json
"""

import requests
import math
import json
import os
import sys
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════════════
# CONFIG — tune these without touching rise_selector.py
# ═══════════════════════════════════════════════════════════════════

CRIME_API_URL  = (
    "https://services3.arcgis.com/dty2kHktVXHrqO8i/ArcGIS/rest/services/"
    "Crime_Incidents/FeatureServer/0/query"
)
SEARCH_RADIUS_MILES = 2   # radius around each parcel to search for crimes
DAYS_BACK           = 180    # how many days of crime data to pull
MAX_RECORDS         = 1000  # max records per parcel query
OUTPUT_FILE         = "crime_mapping.json"

# ═══════════════════════════════════════════════════════════════════
# PARCEL COORDINATES — read directly from rise_selector.py
# These are never modified here, just referenced
# ═══════════════════════════════════════════════════════════════════

# Attempt to import from rise_selector if it's in the same folder.
# If not found, falls back to inline coords so this file always runs standalone.
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rise_selector import HERO_PARCELS
    print("✅ Loaded parcel coordinates from rise_selector.py")
except ImportError:
    print("⚠️  rise_selector.py not found — using inline parcel coordinates")
    HERO_PARCELS = [
        {
            "story":   "Heritage",
            "label":   "Parcel A — Heritage",
            "address": "Commerce Street, Montgomery, Alabama 36104",
            "coords":  (32.37894285621073, -86.31094342590941),
            "zip":     "36104",
        },
        {
            "story":   "Smart Infrastructure",
            "label":   "Parcel B — IX Hub",
            "address": "643 Kimball Street, Montgomery, Alabama 36108",
            "coords":  (32.3683555135919, -86.3438915584216),
            "zip":     "36108",
        },
        {
            "story":   "Economic Urgency",
            "label":   "Parcel C — Food Desert",
            "address": "Coosa Street, Montgomery, Alabama 36104",
            "coords":  (32.37918972033555, -86.30866571125091),
            "zip":     "36104",
        },
    ]

# ═══════════════════════════════════════════════════════════════════
# SEVERITY MAP — classify crime types into High / Medium / Low
# ═══════════════════════════════════════════════════════════════════

SEVERITY_MAP = {
    "high": [
        "HOMICIDE", "MURDER", "RAPE", "ROBBERY", "AGGRAVATED ASSAULT",
        "ARSON", "KIDNAPPING", "SHOOTING", "CARJACKING",
    ],
    "medium": [
        "BURGLARY", "BREAKING AND ENTERING", "ASSAULT", "BATTERY",
        "MOTOR VEHICLE THEFT", "AUTO THEFT", "DRUG", "WEAPONS",
        "DOMESTIC", "INTIMIDATION",
    ],
    "low": [
        "LARCENY", "THEFT", "VANDALISM", "TRESPASSING", "DISORDERLY",
        "FRAUD", "FORGERY", "SHOPLIFTING", "TRAFFIC",
    ],
}

def get_severity(crime_type: str) -> str:
    if not crime_type:
        return "unknown"
    ct = crime_type.upper()
    for level, keywords in SEVERITY_MAP.items():
        if any(k in ct for k in keywords):
            return level
    return "other"

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance between two points in miles."""
    R = 3958.8
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def crime_score(counts: dict, radius_miles: float) -> int:
    """
    Convert raw crime counts into a 0-100 risk score.
    Higher score = higher crime risk.
    Weighted: high x3, medium x2, low x1.
    Normalized per square mile then capped at 100.
    """
    area = math.pi * (radius_miles ** 2)
    weighted = (
        counts.get("high", 0)   * 3 +
        counts.get("medium", 0) * 2 +
        counts.get("low", 0)    * 1
    )
    density = weighted / area if area > 0 else 0
    # Scale: density of 200 weighted crimes/sqmi = score 100
    score = int(min(100, (density / 200) * 100))
    return score


def crime_label(score: int) -> str:
    if score >= 75: return "🔴 High Crime Risk"
    if score >= 45: return "🟠 Elevated Crime Risk"
    if score >= 20: return "🟡 Moderate Crime Risk"
    return "🟢 Low Crime Risk"

# ═══════════════════════════════════════════════════════════════════
# MAIN FETCH FUNCTION
# ═══════════════════════════════════════════════════════════════════

def fetch_crimes_near(parcel: dict) -> dict:
    """
    Query crime incidents within SEARCH_RADIUS_MILES of the parcel.
    Returns aggregated crime data dict.
    """
    p_lat, p_lon = parcel["coords"]
    label        = parcel["label"]
    radius_feet  = SEARCH_RADIUS_MILES * 5280

    cutoff       = datetime.now() - timedelta(days=DAYS_BACK)
    # ArcGIS date filter — try common field names
    date_str     = cutoff.strftime("%Y-%m-%d")

    print(f"\n  [CRIME] Querying: {label}")
    print(f"          Coords : {p_lat:.5f}, {p_lon:.5f}")
    print(f"          Radius : {SEARCH_RADIUS_MILES} miles | Last {DAYS_BACK} days")

    params = {
        "where":            f"1=1",
        "geometry":         json.dumps({"x": p_lon, "y": p_lat}),
        "geometryType":     "esriGeometryPoint",
        "inSR":             "4326",
        "outSR":            "4326",
        "spatialRel":       "esriSpatialRelIntersects",
        "distance":         radius_feet,
        "units":            "esriSRUnit_Foot",
        "outFields":        "*",
        "returnGeometry":   "true",
        "resultRecordCount": MAX_RECORDS,
        "f":                "json",
    }

    try:
        resp = requests.get(CRIME_API_URL, params=params, timeout=20)
        if resp.status_code != 200:
            print(f"  [CRIME] ⚠️  HTTP {resp.status_code}")
            return _crime_fallback(parcel)

        data = resp.json()
        if data.get("error"):
            print(f"  [CRIME] ⚠️  API error: {data['error']}")
            return _crime_fallback(parcel)

        features = data.get("features", [])
        print(f"  [CRIME] ✅ {len(features)} incidents returned")

        if not features:
            return {
                "label":            label,
                "address":          parcel["address"],
                "coords":           {"lat": p_lat, "lon": p_lon},
                "radius_miles":     SEARCH_RADIUS_MILES,
                "days_analyzed":    DAYS_BACK,
                "total_incidents":  0,
                "severity_counts":  {"high": 0, "medium": 0, "low": 0, "other": 0, "unknown": 0},
                "crime_score":      0,
                "crime_label":      "🟢 Low Crime Risk",
                "top_crime_types":  [],
                "incidents":        [],
                "source":           "live",
                "queried_at":       datetime.utcnow().isoformat() + "Z",
            }

        # ── Aggregate ──────────────────────────────────────────
        severity_counts = {"high": 0, "medium": 0, "low": 0, "other": 0, "unknown": 0}
        type_counts     = {}
        incidents       = []

        # Print available fields from first feature for debugging
        if features:
            sample_attrs = features[0].get("attributes", {})
            print(f"  [CRIME]    Sample fields: {list(sample_attrs.keys())[:10]}")

        for f in features:
            attrs    = f.get("attributes", {})
            geom     = f.get("geometry", {})

            # Try common field name variations for crime type
            crime_type = (
                attrs.get("OFFENSE_DESCRIPTION") or
                attrs.get("Offense_Description") or
                attrs.get("offense_description") or
                attrs.get("CRIME_TYPE") or
                attrs.get("Crime_Type") or
                attrs.get("OFFENSE") or
                attrs.get("Offense") or
                attrs.get("TYPE") or
                attrs.get("Type") or
                attrs.get("NIBRS_Description") or
                "Unknown"
            )

            # Try common date field names
            incident_date = (
                attrs.get("REPORT_DATE") or
                attrs.get("Report_Date") or
                attrs.get("DATE_OCCURRED") or
                attrs.get("Date_Occurred") or
                attrs.get("INCIDENT_DATE") or
                attrs.get("Incident_Date") or
                attrs.get("OccurredFromDate") or
                None
            )

            # Convert epoch ms to readable date if needed
            if isinstance(incident_date, (int, float)):
                try:
                    incident_date = datetime.fromtimestamp(incident_date/1000).strftime("%Y-%m-%d")
                except:
                    incident_date = str(incident_date)

            severity = get_severity(crime_type)
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            type_counts[crime_type]   = type_counts.get(crime_type, 0) + 1

            # Compute distance from parcel
            loc_lat = geom.get("y", p_lat)
            loc_lon = geom.get("x", p_lon)
            dist    = haversine(p_lat, p_lon, loc_lat, loc_lon)

            incidents.append({
                "crime_type":    crime_type,
                "severity":      severity,
                "date":          incident_date,
                "dist_miles":    round(dist, 3),
                "coords":        {"lat": loc_lat, "lon": loc_lon},
                "raw_attrs":     attrs,
            })

        # Top 5 crime types by count
        top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        score = crime_score(severity_counts, SEARCH_RADIUS_MILES)
        label_str = crime_label(score)

        print(f"  [CRIME]    Score     : {score}/100  {label_str}")
        print(f"  [CRIME]    High      : {severity_counts['high']}")
        print(f"  [CRIME]    Medium    : {severity_counts['medium']}")
        print(f"  [CRIME]    Low       : {severity_counts['low']}")
        print(f"  [CRIME]    Top type  : {top_types[0][0] if top_types else 'N/A'} ({top_types[0][1] if top_types else 0})")

        return {
            "label":            label,
            "address":          parcel["address"],
            "coords":           {"lat": p_lat, "lon": p_lon},
            "radius_miles":     SEARCH_RADIUS_MILES,
            "days_analyzed":    DAYS_BACK,
            "total_incidents":  len(incidents),
            "severity_counts":  severity_counts,
            "crime_score":      score,
            "crime_label":      label_str,
            "top_crime_types":  [{"type": t, "count": c} for t, c in top_types],
            "incidents":        incidents,
            "source":           "live",
            "queried_at":       datetime.utcnow().isoformat() + "Z",
        }

    except requests.exceptions.Timeout:
        print(f"  [CRIME] ⚠️  Timeout — using fallback")
        return _crime_fallback(parcel)
    except Exception as e:
        print(f"  [CRIME] ⚠️  {type(e).__name__}: {e} — using fallback")
        return _crime_fallback(parcel)


def _crime_fallback(parcel: dict) -> dict:
    """Returns a safe fallback if the API is unreachable."""
    p_lat, p_lon = parcel["coords"]
    return {
        "label":           parcel["label"],
        "address":         parcel["address"],
        "coords":          {"lat": p_lat, "lon": p_lon},
        "radius_miles":    SEARCH_RADIUS_MILES,
        "days_analyzed":   DAYS_BACK,
        "total_incidents": 0,
        "severity_counts": {"high": 0, "medium": 0, "low": 0, "other": 0, "unknown": 0},
        "crime_score":     0,
        "crime_label":     "⚪ Data Unavailable (Fallback)",
        "top_crime_types": [],
        "incidents":       [],
        "source":          "fallback",
        "queried_at":      datetime.utcnow().isoformat() + "Z",
    }

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  MONTGOMERY RISE — Crime Mapping")
    print(f"  Radius: {SEARCH_RADIUS_MILES} miles | Last {DAYS_BACK} days")
    print("=" * 65)

    results = []

    for parcel in HERO_PARCELS:
        crime_data = fetch_crimes_near(parcel)
        results.append(crime_data)

    output = {
        "generated":    datetime.utcnow().isoformat() + "Z",
        "pipeline":     "Montgomery RISE — Crime Mapping",
        "radius_miles": SEARCH_RADIUS_MILES,
        "days_back":    DAYS_BACK,
        "api_endpoint": CRIME_API_URL,
        "parcels":      results,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  ✅ Saved {len(results)} parcels → {OUTPUT_FILE}")
    for r in results:
        print(f"  {r['label']:<30} Score: {r['crime_score']}/100  {r['crime_label']}  ({r['total_incidents']} incidents)")
    print(f"{'=' * 65}\n")
