"""
Foot Traffic Data via Google Popular Times
------------------------------------------
Strategy:
  1. Geocode each address → lat/lon (via Nominatim, free)
  2. Find nearby businesses within SEARCH_RADIUS meters (via Google Places API)
  3. Fetch popular times for each business (via populartimes library)
  4. Average the busyness scores across all nearby businesses
  5. Output averaged peaks, lows, and hourly breakdown per address

Install dependencies:
    pip install requests
    pip install git+https://github.com/m-wrzr/populartimes.git
"""

import json
import time
import requests
import populartimes
import os
from dotenv import load_dotenv

load_dotenv()
# ------------------------------------------------------------------ #
# CONFIG
# ------------------------------------------------------------------ #

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
OUTPUT_FILE     = "foot_traffic_output.json"
SEARCH_RADIUS   = 200   # meters radius around each address to find businesses
MAX_BUSINESSES  = 10    # max nearby businesses to average (keeps API costs low)

ADDRESSES = [
    "Commerce Street, Montgomery, AL",
    "643 Kimball Street, Montgomery, AL",
    "Coosa Street, Montgomery, AL",
]

# ------------------------------------------------------------------ #
# STEP 1 — Geocode address → (lat, lon)
# ------------------------------------------------------------------ #

def geocode_address(address: str) -> tuple[float, float] | None:
    """Use free Nominatim geocoder to get lat/lon from an address string."""
    url     = "https://nominatim.openstreetmap.org/search"
    params  = {"q": address, "format": "json", "limit": 1}
    headers = {
        "Accept-Language": "en",
        "User-Agent": "MontgomeryFootTraffic/1.0 (hackathon project)"
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        print(f"  Geocode status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"  Geocode error: {resp.text[:200]}")
            return None
        data = resp.json()
        if data:
            return (float(data[0]["lat"]), float(data[0]["lon"]))
        print(f"  No geocode results for: {address}")
        return None
    except Exception as e:
        print(f"  Geocode failed: {e}")
        return None

# ------------------------------------------------------------------ #
# STEP 2 — Find nearby businesses via Google Places Nearby Search
# ------------------------------------------------------------------ #

def get_nearby_place_ids(lat: float, lon: float, api_key: str, radius: int = SEARCH_RADIUS) -> list[dict]:
    """Return list of {place_id, name} for businesses near the given coordinates."""
    url    = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lon}",
        "radius":   radius,
        "type":     "establishment",
        "key":      api_key,
    }
    resp    = requests.get(url, params=params)
    results = resp.json().get("results", [])
    return [{"place_id": r["place_id"], "name": r["name"]} for r in results[:MAX_BUSINESSES]]

# ------------------------------------------------------------------ #
# STEP 3 — Fetch popular times per place
# ------------------------------------------------------------------ #

def get_popular_times(place_id: str, api_key: str) -> list[dict]:
    """Fetch raw populartimes data for a single place_id."""
    try:
        data = populartimes.get_id(api_key=api_key, place_id=place_id)
        return data.get("populartimes", [])
    except Exception:
        return []

# ------------------------------------------------------------------ #
# STEP 4 — Parse + Average popular times across businesses
# ------------------------------------------------------------------ #

def parse_popular_times(raw: list[dict]) -> dict:
    """Convert raw populartimes list → {day: {hour: score}} dict."""
    result = {}
    for day_data in raw:
        day  = day_data.get("name", "Unknown")
        data = day_data.get("data", [])
        result[day] = {f"{i:02d}:00": score for i, score in enumerate(data)}
    return result


def average_popular_times(all_parsed: list[dict]) -> dict:
    """
    Given a list of parsed popular_times dicts (one per business),
    return a single dict with averaged scores per day/hour.
    """
    totals = {}  # {day: {hour: [scores]}}

    for parsed in all_parsed:
        for day, hours in parsed.items():
            if day not in totals:
                totals[day] = {}
            for hour, score in hours.items():
                if score == 0:
                    continue  # skip closed hours
                totals[day].setdefault(hour, []).append(score)

    averaged = {}
    for day, hours in totals.items():
        averaged[day] = {
            hour: round(sum(scores) / len(scores))
            for hour, scores in hours.items()
        }

    return averaged


def get_peak_hours(averaged_days: dict) -> dict:
    peaks = {}
    for day, hours in averaged_days.items():
        if not hours:
            peaks[day] = None
            continue
        max_score  = max(hours.values())
        peak_times = [t for t, s in hours.items() if s == max_score]
        peaks[day] = {"time": peak_times, "score": max_score}
    return peaks


def get_low_hours(averaged_days: dict) -> dict:
    lows = {}
    for day, hours in averaged_days.items():
        if not hours:
            lows[day] = None
            continue
        min_score = min(hours.values())
        low_times = [t for t, s in hours.items() if s == min_score]
        lows[day] = {"time": low_times, "score": min_score}
    return lows

# ------------------------------------------------------------------ #
# MAIN — tie it all together
# ------------------------------------------------------------------ #

def process_address(address: str, api_key: str) -> dict:
    print(f"\nProcessing: {address}")

    # 1. Geocode
    coords = geocode_address(address)
    if not coords:
        return {"address": address, "error": "Could not geocode address"}
    lat, lon = coords
    print(f"  Coords: {lat}, {lon}")

    # 2. Find nearby businesses
    nearby = get_nearby_place_ids(lat, lon, api_key)
    if not nearby:
        return {"address": address, "error": "No nearby businesses found", "coords": {"lat": lat, "lon": lon}}
    print(f"  Found {len(nearby)} nearby businesses")

    # 3. Fetch popular times for each
    all_parsed      = []
    businesses_used = []
    for place in nearby:
        print(f"  Fetching: {place['name']}")
        raw    = get_popular_times(place["place_id"], api_key)
        parsed = parse_popular_times(raw)
        if parsed:
            all_parsed.append(parsed)
            businesses_used.append(place["name"])
        time.sleep(0.3)

    if not all_parsed:
        return {
            "address": address,
            "coords":  {"lat": lat, "lon": lon},
            "error":   "No popular times data found for nearby businesses",
        }

    # 4. Average across all businesses
    averaged = average_popular_times(all_parsed)

    return {
        "address":                address,
        "coords":                 {"lat": lat, "lon": lon},
        "businesses_used":        businesses_used,
        "business_count":         len(businesses_used),
        "averaged_popular_times": averaged,
        "peaks":                  get_peak_hours(averaged),
        "lows":                   get_low_hours(averaged),
    }


if __name__ == "__main__":
    address = "Commerce Street, Montgomery, AL"
    coords = geocode_address(address)
    lat, lon = coords

    nearby = get_nearby_place_ids(lat, lon, GOOGLE_API_KEY)

    debug_results = []
    for place in nearby:
        raw = get_popular_times(place["place_id"], GOOGLE_API_KEY)
        if raw:
            print(f"HAS DATA: {place['name']}")
            debug_results.append({
                "name": place["name"],
                "place_id": place["place_id"],
                "has_data": True,
                "popular_times": raw
            })
        else:
            print(f"No data: {place['name']}")
            debug_results.append({
                "name": place["name"],
                "place_id": place["place_id"],
                "has_data": False,
                "popular_times": []
            })
        time.sleep(0.3)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(debug_results, f, indent=2)

    print(f"\nDone. Results saved to {OUTPUT_FILE}")