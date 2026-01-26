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


# ============================
# Routes
# ============================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/compute")
def compute(payload: ComputeRequest, x_api_key: str | None = Header(default=None)):
    auth(x_api_key)

    # Parse dates
    try:
        bd = date.fromisoformat(payload.birth_date)
        sd = date.fromisoformat(payload.start_date)
    except Exception:
        raise HTTPException(status_code=400, detail="birth_date/start_date must be YYYY-MM-DD")

    if payload.months not in (6, 12):
        raise HTTPException(status_code=400, detail="months must be 6 or 12")

    # Google: place -> lat/lon
    place_in = payload.birth_place.strip()
    lat, lon, place_display = google_geocode(place_in)

    # Birth time handling
    time_raw = (payload.birth_time or "").strip()
    time_uncertain = False
    birth_time_missing = False

    if not time_raw:
        birth_time_missing = True
        bt = None
    else:
        if time_raw.startswith("~"):
            time_uncertain = True
            time_raw = time_raw[1:].strip()
        try:
            bt = datetime.strptime(time_raw, "%H:%M").time()
        except Exception:
            raise HTTPException(status_code=400, detail="birth_time must be HH:MM, ~HH:MM, or omitted")

    # Forecast window
    ed = add_months_minus_one_day(sd, payload.months)

    # Month labels
    months_list = []
    cur = date(sd.year, sd.month, 1)
    while cur <= ed:
        months_list.append(cur.strftime("%B %Y"))
        cur = (cur + relativedelta(months=1))

    # Timezone for birth place (use birth date noon UTC as a stable reference)
    reference_utc = datetime(bd.year, bd.month, bd.day, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
    tzname = google_timezone(lat, lon, reference_utc)

    # Natal positions
    natal = {}
    asc = None
    mc = None
    cusps = None
    asc_stable = None

    if not birth_time_missing:
        local = datetime(bd.year, bd.month, bd.day, bt.hour, bt.minute, 0, tzinfo=ZoneInfo(tzname))
        utc = local.astimezone(ZoneInfo("UTC"))
        jd = to_jd_ut(utc)

        for pname in ("Sun", "Moon", "Mercury", "Venus", "Mars"):
            natal[pname] = planet_lon(jd, PLANETS[pname])

        asc0, mc0, cusps0 = houses_placidus(jd, lat, lon)
        asc, mc, cusps = asc0, mc0, cusps0

        if time_uncertain:
            def asc_sign_at(min_delta: int):
                local2 = local + timedelta(minutes=min_delta)
                utc2 = local2.astimezone(ZoneInfo("UTC"))
                jd2 = to_jd_ut(utc2)
                a, m, c = houses_placidus(jd2, lat, lon)
                return sign_index(a)

            s1 = asc_sign_at(-20)
            s2 = asc_sign_at(0)
            s3 = asc_sign_at(+20)
            asc_stable = (s1 == s2 == s3)

    else:
        # No birth time: compute planets at local noon for sign/aspect-level (houses not reliable)
        local_noon = datetime(bd.year, bd.month, bd.day, 12, 0, 0, tzinfo=ZoneInfo(tzname))
        utc_noon = local_noon.astimezone(ZoneInfo("UTC"))
        jd_noon = to_jd_ut(utc_noon)

        for pname in ("Sun", "Moon", "Mercury", "Venus", "Mars"):
            natal[pname] = planet_lon(jd_noon, PLANETS[pname])

    # Natal points for aspect scanning
    natal_points = {
        "Sun": natal.get("Sun"),
        "Moon": natal.get("Moon"),
        "Mercury": natal.get("Mercury"),
        "Venus": natal.get("Venus"),
        "Mars": natal.get("Mars"),
    }

    # Include Asc/MC only if time is present and Asc is stable (or time not uncertain)
    if asc is not None and (not time_uncertain or asc_stable):
        natal_points["Asc"] = asc
        natal_points["MC"] = mc

    # Transit scanning (daily at 12:00 UTC)
    transit_planets = ["Saturn", "Jupiter", "Mars", "Venus", "Mercury", "Uranus", "Neptune", "Pluto"]
    events = []

    @dataclass
    class ActiveWindow:
        tplanet: str
        npoint: str
        sym: str
        ang: int
        start: date
        end: date
        peak_day: date
        peak_dev: float

    active: dict[str, ActiveWindow] = {}

    def key(tp, np, sym, ang):
        return f"{tp}|{np}|{sym}|{ang}"

    d = sd
    while d <= ed:
        dt_utc = datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        jd = to_jd_ut(dt_utc)

        seen = set()

        for tp in transit_planets:
            t_lon = planet_lon(jd, PLANETS[tp])
            orb = ORB_BY_TRANSIT.get(tp, 1.5)

            for np, n_lon in natal_points.items():
                hit = aspect_hit(t_lon, n_lon, orb)
                if hit:
                    sym, ang, dev = hit
                    k = key(tp, np, sym, ang)
                    seen.add(k)

                    if k in active:
                        aw = active[k]
                        aw.end = d
                        if dev < aw.peak_dev:
                            aw.peak_dev = dev
                            aw.peak_day = d
                    else:
                        active[k] = ActiveWindow(tp, np, sym, ang, d, d, d, dev)

        # Close windows not seen today
        to_close = [k for k in list(active.keys()) if k not in seen]
        for k in to_close:
            aw = active.pop(k)
            if (aw.end - aw.start).days >= 1:
                events.append({
                    "label": f"{aw.tplanet} {aw.sym} {aw.npoint}",
                    "start": aw.start.isoformat(),
                    "end": aw.end.isoformat(),
                    "peak": aw.peak_day.isoformat(),
                    "exact": None
                })

        d += timedelta(days=1)

    # Close remaining active windows
    for aw in active.values():
        if (aw.end - aw.start).days >= 1:
            events.append({
                "label": f"{aw.tplanet} {aw.sym} {aw.npoint}",
                "start": aw.start.isoformat(),
                "end": aw.end.isoformat(),
                "peak": aw.peak_day.isoformat(),
                "exact": None
            })

    return {
        "forecast_window": {"start": sd.isoformat(), "end": ed.isoformat(), "months": months_list},
        "location": {"input": payload.birth_place, "resolved": place_display, "lat": lat, "lon": lon, "timezone": tzname},
        "birth_time": {
            "raw": payload.birth_time,
            "missing": birth_time_missing,
            "uncertain": time_uncertain,
            "asc_stable_within_20min": asc_stable
        },
        "system": {"zodiac": "Tropical", "houses_default": "Placidus"},
        "natal_snapshot": {
            "Sun_lon": natal.get("Sun"),
            "Moon_lon": natal.get("Moon"),
            "Mercury_lon": natal.get("Mercury"),
            "Venus_lon": natal.get("Venus"),
            "Mars_lon": natal.get("Mars"),
            "Asc_lon": asc,
            "MC_lon": mc
        },
        "transits": events
    }
