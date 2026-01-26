import os
import math
import requests
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import swisseph as swe
from dateutil.relativedelta import relativedelta
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from timezonefinder import TimezoneFinder

# Swiss Ephemeris flags (MOSEPH = dosyasız çalışır, Render için pratik)
FLAGS = swe.FLG_MOSEPH | swe.FLG_SPEED

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

ASPECTS = [
    ("☌", 0),
    ("⚹", 60),
    ("□", 90),
    ("△", 120),
    ("☍", 180),
]

# Orblar (basit/başlangıç)
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

tf = TimezoneFinder()

app = FastAPI(title="Madam Dudu Ephemeris API", version="1.0.0")
API_KEY = os.getenv("API_KEY", "")

def auth(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def norm_place(place: str) -> str:
    return place.strip().replace("  ", " ")

def geocode_place(place: str):
    # OSM Nominatim (Google Cloud gerekmez)
    # Not: Üretimde cache + rate-limit önerilir.
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1}
    headers = {"User-Agent": "MadamDuduEphemeris/1.0 (contact: you@example.com)"}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise HTTPException(status_code=400, detail=f"Place not found: {place}")
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    display = data[0].get("display_name", place)
    return lat, lon, display

def tz_from_latlon(lat: float, lon: float) -> str:
    tz = tf.timezone_at(lat=lat, lng=lon)
    if not tz:
        raise HTTPException(status_code=400, detail="Timezone not found for coordinates")
    return tz

def to_jd_ut(dt_utc: datetime) -> float:
    hour = dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, hour)

def planet_lon(jd_ut: float, planet_id: int) -> float:
    lon, lat, dist, speed_lon = swe.calc_ut(jd_ut, planet_id, FLAGS)[0]
    return lon % 360.0

def houses_placidus(jd_ut: float, lat: float, lon: float):
    # 'P' = Placidus
    cusps, ascmc = swe.houses_ex(jd_ut, lat, lon, b'P')
    asc = ascmc[0] % 360.0
    mc = ascmc[1] % 360.0
    return asc, mc, [c % 360.0 for c in cusps[1:]]  # 12 cusp

def sign_index(lon: float) -> int:
    return int((lon % 360.0) // 30)

def ang_diff(a: float, b: float) -> float:
    # minimum absolute difference in degrees
    d = (a - b) % 360.0
    if d > 180:
        d = 360 - d
    return abs(d)

def aspect_hit(trans_lon: float, natal_lon: float, orb: float):
    # returns (symbol, angle, deviation) if within orb else None
    d = (trans_lon - natal_lon) % 360.0
    if d > 180:
        d = 360 - d
    for sym, ang in ASPECTS:
        dev = abs(d - ang)
        if dev <= orb:
            return sym, ang, dev
    return None

def add_months_minus_one_day(start: date, months: int) -> date:
    end = (start + relativedelta(months=months)) - timedelta(days=1)
    return end

class ComputeRequest(BaseModel):
    name: str = Field(..., description="Name label")
    birth_date: str = Field(..., description="YYYY-MM-DD")
    birth_time: str | None = Field(None, description="HH:MM or ~HH:MM or None")
    birth_place: str = Field(..., description="City, Country")
    rising_sign: str | None = Field(None, description="Optional (if birth_time missing)")
    start_date: str = Field(..., description="YYYY-MM-DD")
    months: int = Field(..., description="6 or 12")

class TransitEvent(BaseModel):
    label: str
    start: str
    end: str
    peak: str | None = None
    exact: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/compute")
def compute(payload: ComputeRequest, x_api_key: str | None = Header(default=None)):
    auth(x_api_key)

    # parse dates
    try:
        bd = date.fromisoformat(payload.birth_date)
        sd = date.fromisoformat(payload.start_date)
    except Exception:
        raise HTTPException(status_code=400, detail="birth_date/start_date must be YYYY-MM-DD")

    if payload.months not in (6, 12):
        raise HTTPException(status_code=400, detail="months must be 6 or 12")

    # geocode
    place_q = norm_place(payload.birth_place)
    lat, lon, place_display = geocode_place(place_q)
    tzname = tz_from_latlon(lat, lon)

    # birth time handling
    time_raw = (payload.birth_time or "").strip()
    time_uncertain = False
    birth_time_missing = False
    if not time_raw:
        birth_time_missing = True
    else:
        if time_raw.startswith("~"):
            time_uncertain = True
            time_raw = time_raw[1:].strip()
        try:
            bt = datetime.strptime(time_raw, "%H:%M").time()
        except Exception:
            raise HTTPException(status_code=400, detail="birth_time must be HH:MM, ~HH:MM, or omitted")

    # forecast window
    ed = add_months_minus_one_day(sd, payload.months)

    # month labels
    months_list = []
    cur = date(sd.year, sd.month, 1)
    while cur <= ed:
        months_list.append(cur.strftime("%B %Y"))
        cur = (cur + relativedelta(months=1))

    # natal longitudes
    natal = {}

    # Decide if we can compute ASC/MC (needs time)
    asc = None
    mc = None
    cusps = None
    asc_stable = None

    if not birth_time_missing:
        local = datetime(bd.year, bd.month, bd.day, bt.hour, bt.minute, 0, tzinfo=ZoneInfo(tzname))
        utc = local.astimezone(ZoneInfo("UTC"))
        jd = to_jd_ut(utc)

        # natal planets
        for pname in ("Sun", "Moon", "Mercury", "Venus", "Mars"):
            natal[pname] = planet_lon(jd, PLANETS[pname])

        # houses/angles (Placidus)
        asc0, mc0, cusps0 = houses_placidus(jd, lat, lon)
        asc = asc0
        mc = mc0
        cusps = cusps0

        # ~time stability check for Ascendant sign
        if time_uncertain:
            def asc_sign_at(min_delta: int):
                local2 = (local + timedelta(minutes=min_delta))
                utc2 = local2.astimezone(ZoneInfo("UTC"))
                jd2 = to_jd_ut(utc2)
                a, m, c = houses_placidus(jd2, lat, lon)
                return sign_index(a)
            s1 = asc_sign_at(-20)
            s2 = asc_sign_at(0)
            s3 = asc_sign_at(+20)
            asc_stable = (s1 == s2 == s3)
    else:
        # no birth time: compute at noon just to get Sun/Moon etc (safe for sign/aspect-level)
        local_noon = datetime(bd.year, bd.month, bd.day, 12, 0, 0, tzinfo=ZoneInfo(tzname))
        utc_noon = local_noon.astimezone(ZoneInfo("UTC"))
        jd_noon = to_jd_ut(utc_noon)
        for pname in ("Sun", "Moon", "Mercury", "Venus", "Mars"):
            natal[pname] = planet_lon(jd_noon, PLANETS[pname])

    # Build natal points list (ASC/MC only if reliable & available)
    natal_points = {
        "Sun": natal.get("Sun"),
        "Moon": natal.get("Moon"),
        "Mercury": natal.get("Mercury"),
        "Venus": natal.get("Venus"),
        "Mars": natal.get("Mars"),
    }
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

        for tp in transit_planets:
            t_lon = planet_lon(jd, PLANETS[tp])
            orb = ORB_BY_TRANSIT.get(tp, 1.5)

            for np, n_lon in natal_points.items():
                hit = aspect_hit(t_lon, n_lon, orb)
                # close previous windows when hit changes/ends
                # We'll open/extend by exact aspect identity
                # For simplicity: only one aspect per planet/point per day (closest)
                if hit:
                    sym, ang, dev = hit
                    k = key(tp, np, sym, ang)
                    if k in active:
                        aw = active[k]
                        aw.end = d
                        if dev < aw.peak_dev:
                            aw.peak_dev = dev
                            aw.peak_day = d
                    else:
                        active[k] = ActiveWindow(tp, np, sym, ang, d, d, d, dev)

        # any active window that did not get extended today should be closed
        # We detect by marking which keys were seen today
        # For brevity, we re-run a seen set:
        seen = set()
        for tp in transit_planets:
            t_lon = planet_lon(jd, PLANETS[tp])
            orb = ORB_BY_TRANSIT.get(tp, 1.5)
            for np, n_lon in natal_points.items():
                hit = aspect_hit(t_lon, n_lon, orb)
                if hit:
                    sym, ang, dev = hit
                    seen.add(key(tp, np, sym, ang))

        to_close = [k for k in active.keys() if k not in seen]
        for k in to_close:
            aw = active.pop(k)
            # keep only windows >= 2 days to reduce noise
            if (aw.end - aw.start).days >= 1:
                label = f"{aw.tplanet} {aw.sym} {aw.npoint}"
                events.append({
                    "label": label,
                    "start": aw.start.isoformat(),
                    "end": aw.end.isoformat(),
                    "peak": aw.peak_day.isoformat(),
                    "exact": None  # refinement later (optional)
                })

        d += timedelta(days=1)

    # close remaining at end
    for aw in active.values():
        if (aw.end - aw.start).days >= 1:
            label = f"{aw.tplanet} {aw.sym} {aw.npoint}"
            events.append({
                "label": label,
                "start": aw.start.isoformat(),
                "end": aw.end.isoformat(),
                "peak": aw.peak_day.isoformat(),
                "exact": None
            })

    # response
    return {
        "forecast_window": {
            "start": sd.isoformat(),
            "end": ed.isoformat(),
            "months": months_list
        },
        "location": {
            "input": payload.birth_place,
            "resolved": place_display,
            "lat": lat,
            "lon": lon,
            "timezone": tzname
        },
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
