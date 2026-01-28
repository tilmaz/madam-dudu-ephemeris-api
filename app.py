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

FLAGS = swe.FLG_MOSEPH | swe.FLG_SPEED

API_KEY = os.getenv("API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

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

# Orbs (you can tune later)
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

# ---------- Default "clean" filtering (reduces noise) ----------
CLEAN_TRANSITING_PLANETS = {"Saturn", "Jupiter", "Uranus", "Neptune", "Pluto", "Mars"}
CLEAN_NATAL_POINTS = {"Sun", "Moon", "Asc", "MC"}
CLEAN_ASPECTS = {"☌", "☍", "□", "△"}  # drop ⚹ by default (milder)
MAX_EVENTS_PER_MONTH_CLEAN = 12

ASPECT_WEIGHT = {"☌": 5, "☍": 4, "□": 4, "△": 3, "⚹": 2}
PLANET_WEIGHT = {"Pluto": 5, "Saturn": 5, "Neptune": 4, "Uranus": 4, "Jupiter": 4, "Mars": 3, "Venus": 2, "Mercury": 2}

SIGNS = [
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

# ============================
# App
# ============================

app = FastAPI(title="Madam Dudu Ephemeris API", version="1.3.1-retro")


# ============================
# Helpers
# ============================

def auth(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def require_google_key():
    if not GOOGLE_MAPS_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GOOGLE_MAPS_API_KEY in environment variables")


def to_jd_ut(dt_utc: datetime) -> float:
    hour = dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, hour)


def planet_lon(jd_ut: float, planet_id: int) -> float:
    # Swiss Ephemeris return format varies; longitude is always index 0
    res = swe.calc_ut(jd_ut, planet_id, FLAGS)[0]
    lon = float(res[0])
    return lon % 360.0


def planet_lon_speed(jd_ut: float, planet_id: int) -> tuple[float, float]:
    """Return (ecliptic longitude, longitudinal speed) from Swiss Ephemeris."""
    res = swe.calc_ut(jd_ut, planet_id, FLAGS)[0]
    lon = float(res[0]) % 360.0
    spd = float(res[3])  # deg/day (negative = retrograde)
    return lon, spd


def houses_placidus(jd_ut: float, lat: float, lon: float):
    cusps, ascmc = swe.houses_ex(jd_ut, lat, lon, b"P")
    asc = float(ascmc[0]) % 360.0
    mc = float(ascmc[1]) % 360.0
    return asc, mc, [float(c) % 360.0 for c in cusps[1:]]


def sign_index(lon: float) -> int:
    return int((lon % 360.0) // 30)


def format_lon(lon: float | None) -> str | None:
    if lon is None:
        return None
    lon = lon % 360.0
    si = sign_index(lon)
    glyph, name = SIGNS[si]
    deg_in = lon - si * 30.0
    deg = int(deg_in)
    minutes = int(round((deg_in - deg) * 60))
    if minutes == 60:
        minutes = 0
        deg += 1
        if deg == 30:
            deg = 0
            si = (si + 1) % 12
            glyph, name = SIGNS[si]
    return f"{glyph} {name} {deg:02d}°{minutes:02d}′"


def aspect_hit(trans_lon: float, natal_lon: float, orb: float):
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


def parse_label(label: str):
    parts = label.split()
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return None, None, None


def filter_events_clean(events: list[dict]) -> list[dict]:
    # Hard filter
    filtered = []
    for e in events:
        tp, asp, np = parse_label(e.get("label", ""))
        if tp in CLEAN_TRANSITING_PLANETS and np in CLEAN_NATAL_POINTS and asp in CLEAN_ASPECTS:
            filtered.append(e)

    # Group by month and cap
    by_month: dict[str, list[dict]] = {}
    for e in filtered:
        m = e["start"][:7]  # YYYY-MM
        by_month.setdefault(m, []).append(e)

    def score(e: dict) -> int:
        tp, asp, _ = parse_label(e.get("label", ""))
        w = PLANET_WEIGHT.get(tp, 1) + ASPECT_WEIGHT.get(asp, 1)
        try:
            sd = date.fromisoformat(e["start"])
            ed = date.fromisoformat(e["end"])
            days = (ed - sd).days + 1
        except Exception:
            days = 1
        return w * 10 + min(days, 30)

    out = []
    for _, lst in by_month.items():
        lst_sorted = sorted(lst, key=score, reverse=True)
        out.extend(lst_sorted[:MAX_EVENTS_PER_MONTH_CLEAN])

    return sorted(out, key=lambda x: (x["start"], x["label"]))


# ============================
# Google Cloud helpers
# ============================

def google_geocode(place: str):
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
    detail_level: str | None = Field("clean", description="clean (default) or full")


# ============================
# Core compute (shared)
# ============================

def compute_core(payload: ComputeRequest) -> dict:
    # Parse dates
    try:
        bd = date.fromisoformat(payload.birth_date)
        sd = date.fromisoformat(payload.start_date)
    except Exception:
        raise HTTPException(status_code=400, detail="birth_date/start_date must be YYYY-MM-DD")

    if payload.months not in (6, 12):
        raise HTTPException(status_code=400, detail="months must be 6 or 12")

    lat, lon, place_display = google_geocode(payload.birth_place.strip())

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

    # Timezone for birth place (stable reference)
    reference_utc = datetime(bd.year, bd.month, bd.day, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
    tzname = google_timezone(lat, lon, reference_utc)

    # Natal positions
    natal = {}
    asc = None
    mc = None
    asc_stable = None

    if not birth_time_missing:
        local = datetime(bd.year, bd.month, bd.day, bt.hour, bt.minute, 0, tzinfo=ZoneInfo(tzname))
        utc = local.astimezone(ZoneInfo("UTC"))
        jd = to_jd_ut(utc)

        for pname in ("Sun", "Moon", "Mercury", "Venus", "Mars"):
            natal[pname] = planet_lon(jd, PLANETS[pname])

        asc0, mc0, _ = houses_placidus(jd, lat, lon)
        asc, mc = asc0, mc0

        if time_uncertain:
            def asc_sign_at(min_delta: int):
                local2 = local + timedelta(minutes=min_delta)
                utc2 = local2.astimezone(ZoneInfo("UTC"))
                jd2 = to_jd_ut(utc2)
                a, _, _ = houses_placidus(jd2, lat, lon)
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
    if asc is not None and (not time_uncertain or asc_stable):
        natal_points["Asc"] = asc
        natal_points["MC"] = mc

    # Retrogrades (daily at 12:00 UTC; uses speed < 0)
    retro_planets = ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
    retrogrades: list[dict] = []
    for rp in retro_planets:
        in_rx = False
        rx_start: date | None = None
        d_rx = sd
        while d_rx <= ed:
            dt_utc = datetime(d_rx.year, d_rx.month, d_rx.day, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
            jd_rx = to_jd_ut(dt_utc)
            _, spd = planet_lon_speed(jd_rx, PLANETS[rp])
            is_rx = spd < 0

            if is_rx and not in_rx:
                in_rx = True
                rx_start = d_rx
            elif (not is_rx) and in_rx:
                # ended yesterday
                rx_end = d_rx - timedelta(days=1)
                if rx_start is not None and rx_end >= rx_start:
                    retrogrades.append({
                        "planet": rp,
                        "start": rx_start.isoformat(),
                        "end": rx_end.isoformat(),
                    })
                in_rx = False
                rx_start = None

            d_rx += timedelta(days=1)

        # still retrograde at window end
        if in_rx and rx_start is not None:
            retrogrades.append({
                "planet": rp,
                "start": rx_start.isoformat(),
                "end": ed.isoformat(),
            })

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

    def kkey(tp, np, sym, ang):
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
                    kk = kkey(tp, np, sym, ang)
                    seen.add(kk)

                    if kk in active:
                        aw = active[kk]
                        aw.end = d
                        if dev < aw.peak_dev:
                            aw.peak_dev = dev
                            aw.peak_day = d
                    else:
                        active[kk] = ActiveWindow(tp, np, sym, ang, d, d, d, dev)

        # Close windows not seen today
        to_close = [kk for kk in list(active.keys()) if kk not in seen]
        for kk in to_close:
            aw = active.pop(kk)
            if (aw.end - aw.start).days >= 1:
                events.append({
                    "label": f"{aw.tplanet} {aw.sym} {aw.npoint}",
                    "start": aw.start.isoformat(),
                    "end": aw.end.isoformat(),
                    "peak": aw.peak_day.isoformat(),
                    "exact": None
                })

        d += timedelta(days=1)

    # Close remaining
    for aw in active.values():
        if (aw.end - aw.start).days >= 1:
            events.append({
                "label": f"{aw.tplanet} {aw.sym} {aw.npoint}",
                "start": aw.start.isoformat(),
                "end": aw.end.isoformat(),
                "peak": aw.peak_day.isoformat(),
                "exact": None
            })

    detail = (payload.detail_level or "clean").lower().strip()
    transits_out = events if detail == "full" else filter_events_clean(events)

    return {
        # months_labels holds the month anchors used by the writer for month-by-month sections.
        # A 12-month window can span parts of 13 calendar months (e.g., 2026-01-30 → 2027-01-29).
        # We therefore keep both the requested duration and the label list.
        "forecast_window": {
            "start": sd.isoformat(),
            "end": ed.isoformat(),
            "months_requested": payload.months,
            "months_labels": months_list,
        },
        "location": {"input": payload.birth_place, "resolved": place_display, "lat": lat, "lon": lon, "timezone": tzname},
        "birth_time": {"raw": payload.birth_time, "missing": birth_time_missing, "uncertain": time_uncertain, "asc_stable_within_20min": asc_stable},
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
        "retrogrades": retrogrades,
        "detail_level": detail,
        "transits": sorted(transits_out, key=lambda x: (x["start"], x["label"])),
    }


def to_gpt_text(result: dict) -> str:
    fw = result["forecast_window"]
    loc = result["location"]
    bt = result["birth_time"]
    sys = result["system"]
    ns = result["natal_snapshot"]
    transits = result["transits"]

    lines = []
    # Display the user-requested duration, not the count of month labels (which can differ).
    lines.append(f'FORECAST WINDOW: {fw["start"]} – {fw["end"]}  ({fw["months_requested"]} months)')
    lines.append(f'NATAL NAME: {result.get("name","")}')
    lines.append(f'LOCATION: {loc["resolved"]} (lat {loc["lat"]}, lon {loc["lon"]}) | TZ: {loc["timezone"]}')
    lines.append(f'BIRTH TIME: {bt["raw"] or "N/A"} | missing={bt["missing"]} | uncertain={bt["uncertain"]}')
    lines.append(f'SYSTEM: {sys["zodiac"]}; Houses: {sys["houses_default"]}')
    lines.append("")

    # Natal snapshot formatted
    lines.append("NATAL SNAPSHOT (formatted):")
    for key in ("Sun_lon", "Moon_lon", "Mercury_lon", "Venus_lon", "Mars_lon", "Asc_lon", "MC_lon"):
        val = ns.get(key)
        fmt = format_lon(val)
        if fmt is not None:
            label = key.replace("_lon", "").replace("Asc", "ASC").replace("MC", "MC")
            lines.append(f"• {label}: {fmt}")
    lines.append("")

    lines.append(f"TRANSITS ({result['detail_level']}):")
    if not transits:
        lines.append("• (none found for this window with current filters/orbs)")
    else:
        for t in transits:
            lines.append(f"• {t['label']} — {t['start']} to {t['end']} (peak: {t['peak']})")

    # Retrogrades section (for the GPT prompt layer to interpret)
    rxs = result.get("retrogrades", []) or []
    lines.append("")
    lines.append("RETROGRADES:")
    if not rxs:
        lines.append("• No retrogrades detected within this window.")
    else:
        for rx in rxs:
            lines.append(f"• {rx['planet']} Retrograde — {rx['start']} to {rx['end']}")

    return "\n".join(lines)


# ============================
# Routes
# ============================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/compute")
def compute(payload: ComputeRequest, x_api_key: str | None = Header(default=None)):
    auth(x_api_key)
    result = compute_core(payload)
    result["name"] = payload.name
    return result


@app.post("/compute_gpt")
def compute_gpt(payload: ComputeRequest, x_api_key: str | None = Header(default=None)):
    """
    Same inputs as /compute, but returns a GPT-ready text block.
    """
    auth(x_api_key)
    result = compute_core(payload)
    result["name"] = payload.name
    text = to_gpt_text(result)
    return {"text": text}
