# Montgomery RISE
**Revitalization Intelligence & Smart Empowerment**

> *"Every other tool tells you what a parcel was worth. RISE tells you what it's worth right now."*

Built at **WWVibes Hackathon** · Alabama State University · City of Montgomery  
Track: **Workforce & Economic Growth** (cross-coverage: Smart Cities, Civic Access)

---

## What It Does

RISE is a single-page web application that takes city-owned vacant parcels in Montgomery, Alabama and tells planners what each one should become — scored across 8 dimensions using real live data and AI.

A planner clicks a parcel. They get:
- **Top 3 land reuse recommendations** — math-scored, Gemini AI-explained
- **Live foot traffic signals** — real visit counts from Montgomery's Most Visited Locations (ArcGIS)
- **Flood risk flag** — live FEMA zone lookup from Montgomery OneView
- **311 distress signals** — community complaint density from the last 90 days
- **Open grant windows** — verified deadlines and eligibility % from Grants.gov
- **Civil rights heritage boost** — Haversine distance to Rosa Parks Museum, auto-applied
- **Community health panel** — food insecurity, asthma rates, clinic proximity
- **RAG chatbot** — ask anything about any parcel in plain English

---

## The 3 Hero Parcels

| | Parcel A | Parcel B | Parcel C |
|---|---|---|---|
| **Story** | Heritage | Smart Infrastructure | Economic Urgency |
| **Address** | Commerce St, 36104 | 643 Kimball St, 36108 | Coosa St, 36104 |
| **Parcel ID** | 11 01 12 4 004 001 | 11 05 15 1 010 022 | 11 01 12 4 004 001 |
| **Acres** | 8.34 | 0.33 | 8.34 |
| **Nearest Anchor** | Rosa Parks Museum (0.15 mi) | Maxwell AFB Gate (0.89 mi) | MGMix IX Node (0.12 mi) |
| **Zone** | `heritage` | `ix_hub` | `food_desert` |
| **Open Grants** | USDA VAPG · 41 days | USDA Rural Dev · 26 days | USDA Rural Dev · 26 days |

---

## Project Structure

```
montgomery-rise/
│
├── rise_selector_v3.py            # Core pipeline — 8-dimension scorer + ArcGIS + Flood + 311 + Gemini
├── parcel_finder.py               # GIS candidate finder (used pre-hackathon to select hero parcels)
├── parcel_candidates.json         # Output from parcel_finder — top GIS candidates
│
├── montgomery-rise-demo-v2.html   # Main demo UI — parcel cards + recommendations + chatbot
├── rise_footprint.html            # Standalone foot traffic module (ArcGIS visualiser)
├── parcel_review.html             # Parcel review UI (used during candidate selection)
│
├── requirements.txt               # Python dependencies (just: requests)
├── .env.example                   # Environment variable template
├── .gitignore
└── README.md
```

---

## Scoring Engine — 8 Dimensions

| Dimension | Weight | Data Source | Notes |
|---|---|---|---|
| Heritage score | 25% | Haversine to Rosa Parks Museum / Legacy Museum | +25pt boost if within 0.5 mi |
| Industrial score | 20% | Proximity to MGMix IX, Maxwell AFB, ASU Campus | Stacks independently of heritage |
| Activity score | 15% | **ArcGIS Most Visited Locations** | Weighted visit count × proximity |
| Proximity score | 10% | Distance to nearest anchor | 0 mi → 100pts, 2 mi → 0pts |
| Economic score | 10% | Acreage suitability for zone type | Heritage: 0.25–3 ac ideal; IX: 1–10 ac |
| Vacancy score | 10% | GIS improvement value | $0 improvement → 90pts (likely vacant) |
| Flood risk | 5% | **Montgomery OneView Flood Hazard** (ArcGIS) | Zone AE/A → 0pts; Zone X → 5pts |
| 311 distress | 5% | **Montgomery 311 Service Requests** (ArcGIS) | 90-day call density; higher distress = higher urgency |

**Weights sum to exactly 100%.**

### Heritage Boost
```
dist ≤ 0.5 mi  →  score = boost_pts + 75 × (1 - dist/radius)   [max 100]
dist ≤ 1.0 mi  →  score = 40 × (1 - (dist - radius)/radius)    [partial]
dist > 1.0 mi  →  score = 5                                     [out of range]

Rosa Parks Museum boost: +25pts within 0.5 mi
The Legacy Museum boost: +20pts within 0.5 mi
```

### Foot Traffic
```
score = Σ(visits/max_visits × proximity_weight) / min(count, 10) × 100

proximity_weight:
  < 0.25 mi  →  1.0   (direct draw)
  0.25–0.50  →  0.7
  0.50–0.75  →  0.4
  > 0.75     →  0.2
```

### Flood Risk
```
Zone AE / A / VE / Floodway  →  0 pts  🔴 High Flood Risk
Zone X500 / B / C            →  2 pts  🟡 Moderate Flood Risk
Zone X / D                   →  5 pts  🟢 Low Flood Risk
No zone found                →  5 pts  🟢 Low (assumed Zone X)
```

### 311 Distress
```
Density > 50 calls/sq mi (90 days)  →  10 pts  🔴 Critical
Density > 20                        →   7 pts  🟠 High
Density > 10                        →   4 pts  🟡 Moderate
Density ≤ 10                        →   2 pts  🟢 Low

+2 bonus if >30% of calls are vacant-lot-related
(Overgrown Grass, Illegal Dumping, Vacant Building, Debris)
```

---

## APIs Used

All APIs except Gemini are **free with no key required**.

### ArcGIS — Most Visited Locations
```
GET https://services7.arcgis.com/xNUwUjOJqYE54USz/arcgis/rest/services/
    Most_Visited_Locations/FeatureServer/0/query
    ?geometry={"x":-86.3109,"y":32.3789}
    &geometryType=esriGeometryPoint
    &distance=5280&units=esriSRUnit_Foot
    &outFields=Name,Address,F__of_Visits
    &orderByFields=F__of_Visits DESC
    &f=json
```

### ArcGIS — Montgomery Flood Hazard Areas
```
GET https://gis.montgomeryal.gov/server/rest/services/OneView/
    Flood_Hazard_Areas/FeatureServer/0/query
    ?geometry={"x":-86.3109,"y":32.3789}
    &geometryType=esriGeometryPoint
    &spatialRel=esriSpatialRelIntersects
    &outFields=FLD_ZONE,FLOODWAY,SFHA_TF,ZONE_SUBTY
    &f=json
```

### ArcGIS — Montgomery 311 Service Requests
```
GET https://gis.montgomeryal.gov/server/rest/services/HostedDatasets/
    Received_311_Service_Request/MapServer/0/query
    ?where=Create_Date >= date '2025-12-06'
    &geometry={"x":-86.3109,"y":32.3789}
    &distance=2640&units=esriSRUnit_Foot    (0.5 mile radius)
    &outFields=Request_Type,Status,Create_Date,Department
    &f=json
```

### Gemini 2.5 Pro
```
POST https://generativelanguage.googleapis.com/v1beta/
     models/gemini-2.5-pro:generateContent?key={KEY}
Body: {
  "generationConfig": { "responseMimeType": "application/json", "temperature": 0.3 }
}
Cost: ~$0.004 total for 3 parcels (well within free tier: 500 req/day)
```

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| AI / Recommendations | Google Gemini 2.5 Pro | JSON output mode. Free tier at aistudio.google.com |
| Foot Traffic | Montgomery ArcGIS · Most Visited Locations | Free, no key |
| Flood Risk | Montgomery OneView · Flood Hazard Areas | Free, no key |
| 311 Distress | Montgomery ArcGIS · 311 Service Requests | Free, no key. 90-day window |
| Grant Data | Grants.gov API | Free, no key |
| Parcel Data | Montgomery GIS + Nominatim | City-owned parcel geometries |
| RAG Chatbot | ChromaDB + Sentence Transformers | Montgomery parcel knowledge base |
| Backend | FastAPI / Flask | Python — serves scoring engine + API calls |
| Frontend | HTML + CSS + Vanilla JS | No framework — prototype ready |
| Deployment | Render / Railway | Free tier — one-command deploy from GitHub |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/your-team/montgomery-rise.git
cd montgomery-rise
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env:
# GEMINI_API_KEY=your_key_here
```

Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com) — no credit card, 500 requests/day.

### 3. Run the scoring pipeline

```bash
python rise_selector_v3.py
```

**Sample output:**
```
══════════════════════════════════════════════════════════════
  MONTGOMERY RISE — Hero Parcel Scorer v3
══════════════════════════════════════════════════════════════
  HERO PARCEL 1: Parcel A — Heritage
  Commerce Street, Montgomery, Alabama 36104
  ────────────────────────────────────────────────────────────
  [ARCGIS] ✅ 5 locations returned
  [ARCGIS]    Foot traffic score : 73/100
  [FLOOD]  ✅ Zone: X  →  🟢 Low Flood Risk
  [311]       Density: 12.4/sq mi  →  🟡 Moderate Distress

  Scores:
    Heritage   : 78/100   [+25pts — within 0.5mi of Rosa Parks Museum]
    Industrial : 81/100   [anchor: MGMix (IX)]
    Activity   : 73/100   [ArcGIS foot traffic]
    Proximity  : 92/100
    Economic   : 32/100
    Vacancy    : 70/100
    Flood      : 5/5
    311        : 4/10
    ────────────────────────────────────────────────────────────
    FINAL      : 74/100

  [AI] ✅ 3 recommendations
    #1 Civil Rights Heritage Plaza   — 78/100 — Mid-Term $500K–$5M
       🔑 USDA VAPG open 41 days — 50% coverage
    #2 Community Cultural Centre     — 70/100 — Mid-Term $500K–$5M
       🔑 USDA Rural Dev Q3 open 26 days — 70% coverage
    #3 Pocket Park & Public Art      — 55/100 — Quick Win <$500K
```

Output saved to `hero_parcels.json`.

---

## Active Grants — Verified March 2026

| Grant | Status | Deadline | Coverage | Match |
|---|---|---|---|---|
| USDA Rural Economic Dev Q3 | 🟢 OPEN | Mar 31 · 26 days | Up to 70% | 20% |
| USDA Value Added Producer | 🟢 OPEN | Apr 15 · 41 days | Up to 50% | 1:1 |
| HUD CDBG (Montgomery) | Formula | Annual entitlement | Direct | N/A |
| EDA Tech Hubs FY25 Stage II | 🔴 CLOSED | Feb 18 — passed | — | Monitor FY26 |

West Montgomery median income ($22,400) qualifies for the **70–75% coverage tier** under USDA programmes. City out-of-pocket on a $500K project: ~$100–150K.

---

## Fallback Strategy

Every live API call has a silent pre-written fallback — the demo never crashes or shows a blank panel.

| Signal | Live Source | Fallback |
|---|---|---|
| Foot traffic | ArcGIS Most Visited Locations | Pre-researched visit counts per zone |
| Flood risk | Montgomery OneView Flood Hazard | Score 3, Zone X, "Data Unavailable" |
| 311 distress | Montgomery 311 Service Requests | Score 5, density 15/sq mi, "Data Unavailable" |
| AI analysis | Gemini 2.5 Pro | Mock recommendations built from score dimensions |

---

## Environment Variables

```env
# .env  — never commit this file

# Required
GEMINI_API_KEY=your_gemini_key_here

# All other APIs are free public endpoints — no key needed
# ArcGIS foot traffic, flood risk, 311 distress, Grants.gov
```

---

## Mentor Feedback Incorporated

| Mentor | Suggestion | Implemented |
|---|---|---|
| Haley (ASU) | Add foot traffic signals | ✅ ArcGIS Most Visited Locations |
| Haley (ASU) | Flood risk flags | ✅ Montgomery OneView Flood Hazard |
| Haley (ASU) | 311 call density | ✅ Montgomery 311 Service Requests · 90-day window |
| Antoine Edwards | Plan thoroughly before building | ✅ Full scope doc locked before implementation |
