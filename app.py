import os
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import requests
import swisseph as swe
from dateutil.relativedelta import relativedelta
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# ============================
# Config
# ============================

# Swiss Ephemeris flags
# MOSEPH = no external ephemeris files needed (works well on Render)
FLAGS = swe.FLG_MOSEPH | swe.FLG_SPEED

# Secrets / keys (set these in Render Environment Variables)
API_KEY = os.getenv("API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# Planet ids
PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mercury": swe.MERCURY,
    "Venus": swe.VENUS,
    "Mars": swe.MARS,
    "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN,
    "Uranus": swe.URANUS,
    "Neptune": swe.NEPTUNE,
    "Pluto": swe.PLUTO,
}

# Major aspects
ASPECTS = [
    ("☌", 0),
    ("⚹", 60),
    ("□", 90),
    ("△", 120),
    ("☍", 180),
]

# Simple orbs by transiting planet (starter defaults)
ORB_BY_TRANSIT = {
    "Mercury": 1.5,
    "Venus": 1.5,
    "Mars": 2.0,
    "Jupiter": 2.5,
    "Saturn": 2.5,
    "Uranus": 1.5,
    "Neptune": 1.5,
    "Pluto": 1.5,
}

# ============================
# App
# ============================

app = FastAPI(title="Madam Dudu Ephemeris API", version="1.1.1")


# ============================
# Helpers
# ============================

def auth(x_api_key: str | None):
    """If API_KEY is set, require matching x-api-key header."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def require_google_key():
    if not GOOGLE_MAPS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Missing GOOGLE_MAPS_API_KEY in environment variables",
        )


def to_jd_ut(dt_utc: datetime) -> float:
    hour = dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, hour)


def planet_lon(jd_ut: float, planet_id: int) -> float:
    """
    Swiss Ephemeris calc_ut return format varies by flags/version.
    Safest: read longitude from index 0 of the returned position array.
    """
    res = swe.calc_ut(jd_ut, planet_id, FLAGS)[0]  # array-like
    lon = float(res[0])
    return lon % 360.0


def houses_placidus(jd_ut: float, lat: float, lon: float):
    """Placidus houses + angles."""
    cusps, ascmc = swe.houses_ex(jd_ut, lat, lon, b"P")
    asc = float(ascmc[0]) % 360.0
    mc = float(ascmc[1]) % 360.0
    return asc, mc, [float(c) % 360.0 for c in cusps[1:]]  # 12 cusps


def sign_index(lon: float) -> int:
    return int((lon % 360.0) // 30)


def aspect_hit(trans_lon: float, natal_lon: float, orb: float):
    """
    Checks whether trans_lon forms a major aspect to natal_lon within orb.
    Returns (symbol, exact_angle, deviation) or None.
    """
    d = (trans_lon - natal_lon) % 360.0
    if d > 180:
        d = 360 - d
    for sym, ang in ASPECTS:
        dev = abs(d - ang)
        if dev <= orb:
            return sym, ang, dev
    return None


def add_months_minus_one_day(start: date, months: int) -> date:
    return (start + relativedelta(months=months)) - timedelta(days=1)


# ============================
# Google Cloud helpers
# ============================

def google_geocode(place: str):
    """
    City, Country -> (lat, lon, formatted_address)
    Requires Geocoding API enabled.
    """
    require_google_key()
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": place, "key": GOOGLE_MAPS_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    status = data.get("status")
    if status != "OK" or not data.get("results"):
        raise HTTPException(status_code=400, detail=f"Geocoding failed: {status}")

    res = data["results"][0]
    loc = res["geometry"]["location"]
    lat = float(loc["lat"])
    lon = float(loc["lng"])
    formatted = res.get("formatted_address", place)
    return lat, lon, formatted


def google_timezone(lat: float, lon: float, when_utc: datetime) -> str:
    """
    lat/lon + timestamp -> timeZoneId (e.g., Europe/Istanbul)
    Requires Time Zone API enabled.
    """
    require_google_key()
    url = "https://maps.googleapis.com/maps/api/timezone/json"
    ts = int(when_utc.timestamp())
    params = {"location": f"{lat},{lon}", "timestamp": ts, "key": GOOGLE_MAPS_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    status = data.get("status")
    if status != "OK":
        raise HTTPException(status_code=400, detail=f"TimeZone failed: {status}")

    tz_id = data.get("timeZoneId")
    if not tz_id:
        raise HTTPException(status_code=400, detail="TimeZone missing timeZoneId")
    return tz_id


# ============================
# Schemas
# ============================

class ComputeRequest(BaseModel):
    name: str = Field(..., description="Name label")
    birth_date: str = Field(..., description="YYYY-MM-DD")
    birth_time: str | None = Field(None, description="HH:MM or ~HH:MM or omitted")
    birth_place: str = Field(..., description="City, Country")
    start_date: str = Field(..., description="YYYY-MM-DD")
    months: int = Field(..., description="6 or 12")
    detail_level: str = Field("clean", description="Output verbosity: clean|full")


# ============================
# Routes

def _signed_delta_deg(prev_lon: float, curr_lon: float) -> float:
    """Signed smallest difference curr-prev in degrees, in (-180, 180]."""
    return ((curr_lon - prev_lon + 540.0) % 360.0) - 180.0


def compute_retrogrades(sd: date, ed: date, tzname: str) -> list[dict]:
    """
    Compute retrograde windows (day-level) for common retrograde planets within [sd, ed].
    Uses daily geocentric ecliptic longitude at 12:00 local time (converted to UTC) as sampling.
    """
    retro_planets = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]
    windows: list[dict] = []

    # Need a previous day sample to detect start on sd
    prev_day = sd - timedelta(days=1)
    prev_dt = to_utc(prev_day, "12:00", tzname)

    prev_lons = {p: planet_lon(p, prev_dt) for p in retro_planets}
    in_retro = {p: False for p in retro_planets}
    start_date = {p: None for p in retro_planets}  # type: ignore

    d = sd
    while d <= ed:
        dt = to_utc(d, "12:00", tzname)
        for p in retro_planets:
            curr = planet_lon(p, dt)
            delta = _signed_delta_deg(prev_lons[p], curr)

            is_retro = delta < 0  # daily motion backwards

            # Enter retrograde
            if is_retro and not in_retro[p]:
                in_retro[p] = True
                start_date[p] = d

            # Exit retrograde
            if (not is_retro) and in_retro[p]:
                in_retro[p] = False
                s = start_date[p] or sd
                windows.append({
                    "planet": p,
                    "start_date": s.isoformat(),
                    "end_date": d.isoformat(),  # station direct day (day-level)
                })
                start_date[p] = None

            prev_lons[p] = curr

        d += timedelta(days=1)

    # If still retro at end, close at ed
    for p in retro_planets:
        if in_retro[p]:
            s = start_date[p] or sd
            windows.append({
                "planet": p,
                "start_date": s.isoformat(),
                "end_date": ed.isoformat(),
            })

    # Sort by start date then planet
    windows.sort(key=lambda w: (w["start_date"], w["planet"]))
    return windows


ZODIAC = [
    ("♈", "Aries"),
    ("♉", "Taurus"),
    ("♊", "Gemini"),
    ("♋", "Cancer"),
    ("♌", "Leo"),
    ("♍", "Virgo"),
    ("♎", "Libra"),
    ("♏", "Scorpio"),
    ("♐", "Sagittarius"),
    ("♑", "Capricorn"),
    ("♒", "Aquarius"),
    ("♓", "Pisces"),
]


def fmt_zodiac(lon: float) -> str:
    """Format longitude as '♉ Taurus 09°54′'."""
    lon = lon % 360.0
    sign = int(lon // 30)
    deg = lon - sign * 30
    d = int(deg)
    m = int(round((deg - d) * 60))
    if m == 60:
        m = 0
        d += 1
        if d == 30:
            d = 0
            sign = (sign + 1) % 12
    sym, name = ZODIAC[sign]
    return f"{sym} {name} {d:02d}°{m:02d}′"


def fmt_aspect_symbol(a: str) -> str:
    return ASPECTS.get(a, a)


def format_compute_gpt(result: dict, payload: ComputeRequest) -> str:
    sd = payload.start_date
    ed = sd + relativedelta(months=payload.months) - timedelta(days=1)

    lines: list[str] = []
    lines.append(f"FORECAST WINDOW: {sd.isoformat()} – {ed.isoformat()}  ({payload.months} months)")
    lines.append(f"NATAL NAME: {payload.name}")
    loc = result.get("location", {})
    lines.append(f"LOCATION: {payload.birth_place} (lat {loc.get('lat')}, lon {loc.get('lon')}) | TZ: {loc.get('tzname')}")
    bt = payload.birth_time or ""
    lines.append(f"BIRTH TIME: {bt} | missing={result.get('birth_time_missing')} | uncertain={result.get('birth_time_uncertain')}")
    lines.append(f"SYSTEM: Tropical; Houses: {payload.house_system.capitalize()}")
    lines.append("")
    lines.append("NATAL SNAPSHOT (formatted):")
    snap = result["natal_snapshot"]
    lines.append(f"• Sun: {fmt_zodiac(snap['sun'])}")
    lines.append(f"• Moon: {fmt_zodiac(snap['moon'])}")
    lines.append(f"• Mercury: {fmt_zodiac(snap['mercury'])}")
    lines.append(f"• Venus: {fmt_zodiac(snap['venus'])}")
    lines.append(f"• Mars: {fmt_zodiac(snap['mars'])}")
    lines.append(f"• ASC: {fmt_zodiac(snap['asc'])}")
    lines.append(f"• MC: {fmt_zodiac(snap['mc'])}")
    lines.append("")

    # Transits
    detail = (payload.detail_level or "clean").lower()
    lines.append(f"TRANSITS ({detail}):")
    transits = result.get("transits", [])
    for ev in transits:
        lines.append(f"• {ev['label']} — {ev['start_date']} to {ev['end_date']} (peak: {ev['peak_date']})")
    lines.append("")

    # Retrogrades
    retros = result.get("retrogrades", [])
    if retros:
        lines.append("RETROGRADES (day-level):")
        # Capitalize planet names
        for r in retros:
            pname = r["planet"].capitalize()
            lines.append(f"• {pname} Retrograde — {r['start_date']} to {r['end_date']}")
    else:
        lines.append("RETROGRADES: none detected in this window.")
    return "\n".join(lines)


def run_compute(payload: ComputeRequest) -> dict:
    # Inputs and time window
    sd = payload.start_date
    ed = sd + relativedelta(months=payload.months) - timedelta(days=1)

    # Resolve location / timezone (Google geocode)
    geo = geocode(payload.birth_place)
    lat = geo["lat"]
    lon = geo["lon"]
    tzname = geo["tzname"]

    # Birth time handling
    birth_time_missing = payload.birth_time is None
    birth_time_uncertain = False

    if payload.birth_time is None:
        # Default noon local if missing
        bt = "12:00"
        birth_time_uncertain = True
    else:
        bt = payload.birth_time

    # Natal snapshot
    natal_dt_utc = to_utc(payload.birth_date, bt, tzname)
    snapshot = {k: planet_lon(k, natal_dt_utc) for k in PLANET_KEYS}
    snapshot["asc"], snapshot["mc"] = asc_mc(payload.birth_date, bt, tzname, lat, lon, payload.house_system)

    # Transit selection
    include = set(payload.include_aspects or ["conjunction", "opposition", "square", "trine"])
    orb = float(payload.orb_deg or 2.5)

    # Scan day by day for transits
    events: list[dict] = []
    active: dict[tuple[str, str, str], dict] = {}

    d = sd
    while d <= ed:
        dt = to_utc(d, "12:00", tzname)
        for tp in TRANSIT_PLANETS:
            t_lon = planet_lon(tp, dt)

            for natal_key in NATAL_POINTS:
                n_lon = snapshot[natal_key]
                for asp_name, asp_deg in ASPECT_ANGLES.items():
                    if asp_name not in include:
                        continue

                    diff = abs(((t_lon - n_lon + 180) % 360) - 180)
                    delta = abs(diff - asp_deg)
                    key = (tp, natal_key, asp_name)

                    if delta <= orb:
                        if key not in active:
                            active[key] = {
                                "label": f"{tp.capitalize()} {fmt_aspect_symbol(asp_name)} {natal_key.capitalize()}",
                                "start_date": d.isoformat(),
                                "end_date": d.isoformat(),
                                "peak_date": d.isoformat(),
                                "peak_delta": delta,
                                "exact_date": None,
                            }
                        else:
                            active[key]["end_date"] = d.isoformat()
                            if delta < active[key]["peak_delta"]:
                                active[key]["peak_delta"] = delta
                                active[key]["peak_date"] = d.isoformat()
                    else:
                        if key in active:
                            events.append(active.pop(key))
        d += timedelta(days=1)

    events.extend(active.values())
    # Sort events by peak date
    events.sort(key=lambda e: (e["peak_date"], e["label"]))

    # Retrogrades (computed, not interpreted)
    retros = compute_retrogrades(sd, ed, tzname)

    return {
        "location": geo,
        "birth_time_missing": birth_time_missing,
        "birth_time_uncertain": birth_time_uncertain,
        "natal_snapshot": snapshot,
        "transits": events,
        "retrogrades": retros,
        "meta": {"sd": sd.isoformat(), "ed": ed.isoformat(), "months": payload.months},
    }


@app.post("/compute")
def compute(payload: ComputeRequest, x_api_key: str = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return run_compute(payload)


@app.post("/compute_gpt")
def compute_gpt(payload: ComputeRequest, x_api_key: str = Header(default=None)):
    """
    Same inputs as /compute, but returns a GPT-ready text block.
    """
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    data = run_compute(payload)
    return {"text": format_compute_gpt(data, payload)}
