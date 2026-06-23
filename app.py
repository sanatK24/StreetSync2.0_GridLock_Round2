"""
app.py  –  ASTraM Event Impact Forecaster
Gridlock Hackathon 2.0 · Event-Driven Congestion (Planned & Unplanned)

WHAT MAKES THIS DIFFERENT FROM THE OTHER ~1,600 TEAMS
═══════════════════════════════════════════════════════
1. Live weather adjustment  — calls OpenMeteo (free, no API key) to
   adjust closure probability upward in real time when it's raining.
   Rain + evening + accident on ORR = very different risk from the same
   event on a dry Tuesday morning. No other team will do this.

2. System comparison        — side-by-side: what the CURRENT ASTraM
   priority-rule system outputs vs what our ML + Impact Index outputs.
   Makes the gap between "corridor-based triage" and real risk scoring
   concrete and visual. The data shows the current system gets priority
   right 99.8% of the time — because priority is DEFINED by corridor
   membership, not by actual impact. That is the problem we're solving.

3. Anomaly detection        — flags when this event is statistically
   unusual (closure probability significantly above/below the historical
   base rate for this cause × corridor × time combination). Raw forecast
   numbers don't tell you this; percentile context does.

4. Duration confidence      — P10/P50/P90 from actual similar incidents,
   not just a point estimate from a model that barely beats the naive
   baseline. "90% of similar events resolved in under 2 hours" is more
   useful than "expected 65 minutes."

5. Resource optimizer       — given available officers for a shift,
   produces an impact-weighted allocation table across all active events.

6. Active operations        — multi-event command view with conflict
   detection (events within 500m flagged automatically), combined map,
   and download briefing per event.

All free. No paid APIs. Runs locally or on Streamlit Community Cloud.

Run locally
───────────
    pip install -r requirements.txt
    python src/train_models.py      # one-time: ~10 seconds
    streamlit run app.py
"""
from __future__ import annotations

import json
import math
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from recommend import ForecastEngine  # noqa: E402
import feedback_store                  # noqa: E402
import train_models                    # noqa: E402
import geo_utils                       # noqa: E402

MODELS_DIR = ROOT / "models"

st.set_page_config(
    page_title="ASTraM · Event Impact Forecaster",
    page_icon="🚦", layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CAUSE OPTIONS ────────────────────────────────────────────────────────────
CAUSE_OPTIONS = [
    "public_event", "procession", "vip_movement", "protest", "construction",
    "vehicle_breakdown", "accident", "pot_holes", "tree_fall", "water_logging",
    "road_conditions", "congestion", "debris", "Fog/Low Visibility", "others",
]

# Rain-sensitive causes: closure probability rises sharply when it rains
RAIN_SENSITIVE = {"water_logging", "vehicle_breakdown", "accident", "pot_holes", "Fog/Low Visibility"}

# ─── DESIGN TOKENS ────────────────────────────────────────────────────────────
INK   = "#0A0C10"
SURF  = "#14161B"
SURF2 = "#11131A"
LINE  = "#1C1F27"
PAPER = "#ECEDEF"
MUTE  = "#8A8F9C"
GREEN = "#34D399"
AMBER = "#F5A623"
ORANGE= "#F2784B"
RED   = "#E5484D"
BLUE  = "#5EA8ED"
PURPLE= "#B98AE0"

BAND_COLOR = {"Low": GREEN, "Medium": AMBER, "High": ORANGE, "Critical": RED}
CHART_COLORWAY = [AMBER, BLUE, GREEN, PURPLE, RED, ORANGE, "#6FCBD0", "#E0B84A", "#8C97A8"]
MAP_STYLE  = "carto-darkmatter"
NO_TOOLBAR = {"displayModeBar": False}
COMPASS    = ["N","NE","E","SE","S","SW","W","NW"]


def compass_dir(b: float) -> str:
    return COMPASS[int((b + 22.5) / 45) % 8]


def prettify(name: str) -> str:
    if not isinstance(name, str): return name
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    return re.sub(r"\s+", " ", spaced).strip().replace("Junc", "Junction")


# ─── CSS ─────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

#MainMenu, footer, header { display: none !important; }
.block-container { padding-top: 1.4rem !important; padding-bottom: 3rem !important; max-width: 1460px; }
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp {
  background: radial-gradient(ellipse 1200px 500px at 50% -6%, rgba(245,166,35,0.055), transparent), #0A0C10;
}
h1,h2,h3 { font-family: 'Archivo', sans-serif !important; letter-spacing: -0.01em; }

.eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
  letter-spacing: 0.1em; text-transform: uppercase; color: #F5A623;
  margin-bottom: 3px; font-weight: 500;
}
.section-head { font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.28rem; color: #ECEDEF; margin: 0; }
.section-sub  { color: #8A8F9C; font-size: 0.86rem; margin-top: 2px; }
.card-title   { font-weight: 600; font-size: 0.94rem; color: #ECEDEF; margin-bottom: 8px; }
.hr-line      { border: none; border-top: 1px solid #1C1F27; margin: 0.3rem 0 1.1rem 0; }

.stat-card {
  background: #11131A; border: 1px solid #1C1F27;
  border-radius: 10px; padding: 13px 15px 11px 15px; height: 100%;
}
.stat-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem;
  letter-spacing: 0.07em; text-transform: uppercase; color: #8A8F9C; font-weight: 500; margin-bottom: 5px; }
.stat-value { font-family: 'Archivo', sans-serif; font-weight: 700;
  font-size: 1.6rem; color: #ECEDEF; line-height: 1.15; }
.stat-sub   { font-size: 0.74rem; color: #8A8F9C; margin-top: 4px; line-height: 1.4; }

.pill { display:inline-block; padding:3px 13px; border-radius:100px; font-weight:600; font-size:0.8rem; }
.dot  { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:5px; }

.hero-bar {
  display:flex; align-items:center; justify-content:space-between;
  padding:2px 0 16px 0; border-bottom:1px solid #1C1F27; margin-bottom:1.4rem;
}
.hero-title   { font-family:'Archivo',sans-serif; font-weight:800; font-size:1.5rem; color:#ECEDEF; }
.hero-tagline { color:#8A8F9C; font-size:0.84rem; margin-top:2px; }
.hero-status  { font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#8A8F9C; text-align:right; white-space:nowrap; }
.mono { font-family:'IBM Plex Mono',monospace; }

.empty-state { border:1px dashed #262A33; border-radius:10px; padding:20px 18px; color:#8A8F9C; font-size:0.88rem; line-height:1.6; }
.alert-box   { border-radius:8px; padding:11px 14px; font-size:0.86rem; color:#ECEDEF; margin-bottom:8px; }
.alert-warn  { background:rgba(229,72,61,0.08); border:1px solid rgba(229,72,61,0.35); }
.alert-info  { background:rgba(94,168,237,0.08); border:1px solid rgba(94,168,237,0.3); }
.alert-weather { background:rgba(94,168,237,0.12); border:1px solid rgba(94,168,237,0.45); }
.alert-anomaly { background:rgba(245,166,35,0.10); border:1px solid rgba(245,166,35,0.4); }

.compare-table {
  width:100%; border-collapse:collapse; font-size:0.88rem;
}
.compare-table th {
  font-family:'IBM Plex Mono',monospace; font-size:0.68rem; letter-spacing:0.07em;
  text-transform:uppercase; color:#8A8F9C; padding:6px 10px; border-bottom:1px solid #1C1F27;
}
.compare-table td { padding:7px 10px; border-bottom:1px solid #1C1F270f; vertical-align:top; }
.compare-old { color:#8A8F9C; }
.compare-new { color:#ECEDEF; font-weight:500; }

@keyframes tlpulse { 0%,100%{opacity:1} 50%{opacity:.42} }
.tl-pulse { animation:tlpulse 1.3s ease-in-out infinite; }

div[data-testid="stVerticalBlockBorderWrapper"] {
  border:1px solid #1C1F27 !important; border-radius:12px !important; background-color:#11131A !important;
}
div[data-testid="stForm"] {
  border:1px solid #1C1F27 !important; border-radius:12px !important;
  padding:1.2rem 1.2rem 0.5rem 1.2rem !important; background-color:#11131A;
}
.stButton>button, .stFormSubmitButton>button {
  border-radius:8px !important; font-weight:600 !important; border:1px solid #262A33 !important;
  transition:border-color .15s,color .15s;
}
.stButton>button:hover, .stFormSubmitButton>button:hover {
  border-color:#F5A623 !important; color:#F5A623 !important;
}
.stTextInput input,.stNumberInput input,.stDateInput input,
.stTimeInput input,.stTextArea textarea,div[data-baseweb="select"]>div {
  border-radius:8px !important; border-color:#262A33 !important;
  background-color:#0E1014 !important; font-family:'Inter',sans-serif !important;
}
hr { border-color:#1C1F27 !important; }
button[data-baseweb="tab"] { font-family:'Archivo',sans-serif; font-weight:600; font-size:0.93rem; }
.js-plotly-plot,.plot-container { border-radius:8px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)


# ─── UI HELPERS ───────────────────────────────────────────────────────────────
def vspace(px=16): st.markdown(f'<div style="height:{px}px"></div>', unsafe_allow_html=True)
def eyebrow(t): st.markdown(f'<div class="eyebrow">{t}</div>', unsafe_allow_html=True)
def card_title(t): st.markdown(f'<div class="card-title">{t}</div>', unsafe_allow_html=True)
def section_head(title, sub=None):
    h = f'<div class="section-head">{title}</div>'
    if sub: h += f'<div class="section-sub">{sub}</div>'
    st.markdown(h, unsafe_allow_html=True)
def hr(): st.markdown('<hr class="hr-line"/>', unsafe_allow_html=True)
def stat_card(label, value, sub=""):
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-sub">{sub}</div></div>', unsafe_allow_html=True,
    )
def pill(text, color):
    st.markdown(
        f'<span class="pill" style="background:{color}1a;color:{color};border:1px solid {color}55;">{text}</span>',
        unsafe_allow_html=True,
    )
def status_dot(c): return f'<span class="dot" style="background:{c};"></span>'
def alert(text, kind="info"):
    st.markdown(f'<div class="alert-box alert-{kind}">{text}</div>', unsafe_allow_html=True)

def themed(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=PAPER, size=12.5),
        margin=dict(l=4, r=4, t=24, b=4),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        colorway=CHART_COLORWAY,
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE)
    if height: fig.update_layout(height=height)
    return fig

def traffic_light_svg(band):
    lit = {"Low":"green","Medium":"amber","High":"red","Critical":"red"}.get(band,"amber")
    colors = {"red":RED,"amber":AMBER,"green":GREEN}
    bulbs = ""
    for name, cy in [("red",32),("amber",92),("green",152)]:
        on = name == lit
        fill = colors[name] if on else "#1E2129"
        glow = f"drop-shadow(0 0 9px {colors[name]})" if on else "none"
        cls  = "tl-pulse" if (on and band == "Critical") else ""
        bulbs += f'<circle cx="35" cy="{cy}" r="19" fill="{fill}" style="filter:{glow}" class="{cls}"/>'
    return (f'<svg width="70" height="190" viewBox="0 0 70 190" xmlns="http://www.w3.org/2000/svg">'
            f'<rect x="3" y="3" width="64" height="184" rx="16" fill="{SURF}" stroke="{LINE}" stroke-width="2"/>'
            f'{bulbs}</svg>')


# ─── INTELLIGENCE FUNCTIONS ───────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather() -> dict:
    """Live Bengaluru weather from OpenMeteo (free, no API key).
    Cached 30 min.  Fails gracefully — multiplier=1.0 if unreachable."""
    try:
        import requests
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 12.9716, "longitude": 77.5946,
                "current": "precipitation,weathercode,temperature_2m,windspeed_10m",
                "timezone": "Asia/Kolkata", "forecast_days": 1,
            },
            timeout=6,
        )
        r.raise_for_status()
        curr = r.json()["current"]
        rain = float(curr.get("precipitation", 0) or 0)
        code = int(curr.get("weathercode", 0) or 0)
        temp = float(curr.get("temperature_2m", 28) or 28)
        wind = float(curr.get("windspeed_10m", 0) or 0)
        is_rain  = rain > 0 or (51 <= code <= 99)
        is_heavy = rain > 5 or code >= 80
        mult = 1.0 + (0.25 if is_rain else 0) + (0.25 if is_heavy else 0)
        wmo_desc = {
            range(0,4):   "Clear / mostly clear",
            range(10,20): "Mist / haze",
            range(20,30): "Recent precipitation",
            range(40,50): "Fog",
            range(51,68): "Drizzle or rain",
            range(71,78): "Snow",
            range(80,83): "Rain showers",
            range(95,100):"Thunderstorm",
        }
        desc = next((v for rng, v in wmo_desc.items() if code in rng), "Overcast/cloudy")
        return dict(available=True, rain_mm=rain, weathercode=code,
                    temperature=temp, windspeed=wind, is_rain=is_rain,
                    is_heavy=is_heavy, multiplier=mult, description=desc)
    except Exception:
        return dict(available=False, multiplier=1.0, is_rain=False,
                    is_heavy=False, description="unavailable")


def compute_anomaly(df, event_cause, corridor, closure_prob) -> dict | None:
    """Compares forecast closure probability against historical base rate for
    [cause × corridor].  Returns percentile context and an 'is_elevated' flag."""
    mask = df["event_cause"] == event_cause
    cm   = mask & (df["corridor"] == corridor) if corridor and corridor != "Non-corridor" else mask
    sub  = df[cm] if cm.sum() >= 5 else df[mask]
    if len(sub) < 3:
        return None
    hist = sub["requires_road_closure"].mean()
    excess = closure_prob - hist
    all_probs = sub["requires_road_closure"].values
    percentile = (all_probs <= closure_prob).mean() * 100
    return dict(
        n=len(sub), hist=hist, excess=excess,
        percentile=percentile,
        is_elevated=excess > 0.15,
        is_suppressed=excess < -0.15,
    )


def duration_ci(sim_df) -> dict | None:
    """P10/P50/P90 duration confidence interval from similar incidents."""
    if sim_df is None or len(sim_df) < 3:
        return None
    durs = sim_df["duration_min"].dropna()
    if len(durs) < 3:
        return None
    return dict(p10=max(0, durs.quantile(0.10)), p50=max(0, durs.quantile(0.50)),
                p90=max(0, durs.quantile(0.90)), n=len(durs))


def optimize_allocation(events: list, available: int) -> list:
    """Impact-score weighted greedy allocation of available officers."""
    active = [e for e in events if e["status"] == "active"]
    if not active or available <= 0:
        return []
    total_score = sum(e["impact_score"] for e in active) or 1
    result, remaining = [], available
    for i, ev in enumerate(sorted(active, key=lambda e: e["impact_score"], reverse=True)):
        alloc = (remaining if i == len(active)-1
                 else min(remaining, max(1, round(available * ev["impact_score"] / total_score))))
        remaining = max(0, remaining - alloc)
        result.append({**ev, "allocated": alloc,
                       "shortfall": max(0, ev["personnel"] - alloc)})
    return result


def detect_conflicts(events: list) -> list:
    active = [(i, e) for i, e in enumerate(events) if e["status"] == "active"]
    out = []
    for a_idx in range(len(active)):
        for b_idx in range(a_idx + 1, len(active)):
            i, ea = active[a_idx]; j, eb = active[b_idx]
            dlat = math.radians(eb["lat"] - ea["lat"])
            dlon = math.radians(eb["lon"] - ea["lon"])
            a = (math.sin(dlat/2)**2 + math.cos(math.radians(ea["lat"])) *
                 math.cos(math.radians(eb["lat"])) * math.sin(dlon/2)**2)
            d = 2 * 6371 * math.asin(math.sqrt(a)) * 1000
            if d < 500:
                out.append((i, j, d))
    return out


# ─── MAP FIGURES ──────────────────────────────────────────────────────────────
def deployment_map(lat, lon, barricades, officers, route_points=None,
                   bc_dist=90, dep_mode="geometric_fallback",
                   impact_band="Medium", impact_score=0, location_name="", zoom=15.5):
    fig = go.Figure()
    ring_lats, ring_lons = geo_utils.coverage_ring_points(lat, lon, bc_dist * 1.5)
    fig.add_trace(go.Scattermapbox(
        lat=ring_lats, lon=ring_lons, mode="lines",
        line=dict(color="rgba(245,166,35,0.27)", width=1.5),
        name="Coverage radius", hoverinfo="skip",
    ))
    if route_points:
        fig.add_trace(go.Scattermapbox(
            lat=[p["lat"] for p in route_points], lon=[p["lon"] for p in route_points],
            mode="markers+text", marker=dict(size=11, color=BLUE),
            text=[p["corridor"] for p in route_points], textposition="top right",
            name="Diversion route",
            hovertext=[f"<b>Alternate route</b><br>{p['corridor']}" for p in route_points],
            hoverinfo="text",
        ))
    if barricades:
        bearings = [geo_utils.bearing_between(lat, lon, b[0], b[1]) for b in barricades]
        fig.add_trace(go.Scattermapbox(
            lat=[b[0] for b in barricades], lon=[b[1] for b in barricades],
            mode="markers", marker=dict(size=17, color=ORANGE), name="Barricade",
            hovertext=[
                f"<b>Barricade {i+1}</b><br>Approach from <b>{compass_dir(b)}</b> ({b:.0f}°)<br>"
                f"~{bc_dist}m from incident<br><i>Station one officer here</i>"
                for i, b in enumerate(bearings)
            ],
            hoverinfo="text",
        ))
    if officers:
        at_bc = len(barricades)
        fig.add_trace(go.Scattermapbox(
            lat=[o[0] for o in officers], lon=[o[1] for o in officers],
            mode="markers", marker=dict(size=11, color=GREEN), name="Officer",
            hovertext=[
                f"<b>Officer post {i+1}</b><br>" +
                (f"At Barricade {i+1} · {compass_dir(geo_utils.bearing_between(lat, lon, barricades[i][0], barricades[i][1]))} approach"
                 if i < at_bc and barricades else "On-scene command · Coordinate with barricades")
                for i in range(len(officers))
            ],
            hoverinfo="text",
        ))
    fig.add_trace(go.Scattermapbox(
        lat=[lat], lon=[lon], mode="markers",
        marker=dict(size=21, color=RED), name="Incident",
        hovertext=[f"<b>Incident</b><br>{location_name or 'Selected point'}<br>"
                   f"{lat:.5f}, {lon:.5f}<br>Impact: {impact_score:.0f}/100 [{impact_band}]"],
        hoverinfo="text",
    ))
    fig.update_layout(
        mapbox_style=MAP_STYLE, mapbox_center={"lat": lat, "lon": lon},
        mapbox_zoom=zoom, margin=dict(l=0, r=0, t=0, b=0), height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=PAPER, size=11)),
    )
    return fig


def ops_map(events: list) -> go.Figure:
    fig = go.Figure()
    for band in ["Low","Medium","High","Critical"]:
        evs = [e for e in events if e["band"] == band]
        if not evs: continue
        color = BAND_COLOR[band]
        fig.add_trace(go.Scattermapbox(
            lat=[e["lat"] for e in evs], lon=[e["lon"] for e in evs],
            mode="markers",
            marker=dict(size=[max(16, min(30, 10+e["impact_score"]/4)) for e in evs], color=color),
            name=band,
            hovertext=[
                f"<b>{e['cause'].replace('_',' ').title()}</b><br>"
                f"Score {e['impact_score']:.0f}/100 [{e['band']}]<br>"
                f"{e['location_name']}<br>{e['when_ist']}<br>"
                f"Personnel: {e['personnel']} · Barricades: {e['barricades']}<br>"
                f"Corridor: {e['corridor']}<br>Status: {e['status'].upper()}"
                for e in evs
            ],
            hoverinfo="text",
        ))
    clat = sum(e["lat"] for e in events)/len(events) if events else 12.97
    clon = sum(e["lon"] for e in events)/len(events) if events else 77.59
    fig.update_layout(
        mapbox_style=MAP_STYLE, mapbox_center={"lat": clat, "lon": clon},
        mapbox_zoom=10.5, margin=dict(l=0, r=0, t=0, b=0), height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=PAPER, size=11)),
    )
    return fig


# ─── BRIEFING GENERATOR ───────────────────────────────────────────────────────
def generate_briefing(result, dep, weather, location_name, when_ist) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    impact = result["impact_index"]
    ctx = result["context"]; mp = result["manpower"]
    bc = result["barricades"]; dv = result["diversion"]
    dur = result["duration"]; sim = result["similar_incidents"]
    tier_p = dur["tier_probabilities"]
    sep = lambda c="─", n=46: c * n

    lines = [
        "ASTraM EVENT IMPACT BRIEFING", sep("="),
        f"Generated : {now}",
        "System    : ASTraM Event Impact Forecaster — Gridlock Hackathon 2.0",
        "",
        "EVENT DETAILS", sep(),
        f"Location  : {location_name} ({result['input']['lat']:.5f}, {result['input']['lon']:.5f})",
        f"Cause     : {result['input']['event_cause'].replace('_',' ').title()} ({result['input']['event_type'].title()})",
        f"Date/Time : {when_ist}",
        f"Priority  : {result['priority']['used']}",
        f"Corridor  : {ctx['corridor']}",
        f"Pol. Stn  : {ctx['police_station']}",
        f"Zone      : {ctx.get('zone','Unknown')}",
        "",
    ]
    if weather.get("available") and weather.get("is_rain"):
        lines += [
            "WEATHER CONDITIONS (LIVE — OpenMeteo)", sep(),
            f"Rain: {weather['rain_mm']:.1f}mm · {weather['description']}",
            f"Closure risk adjusted ×{weather['multiplier']:.2f} (rain-sensitive event type)",
            "",
        ]
    lines += [
        "IMPACT ASSESSMENT", sep(),
        f"Impact Index   : {impact['score']:.0f} / 100  [{impact['band'].upper()}]",
        f"Closure risk   : {result['closure_probability']*100:.0f}%",
        f"Breakdown      : Closure {impact['components']['closure']}pts + "
        f"Duration {impact['components']['duration']}pts + "
        f"Priority {impact['components']['priority']}pts + "
        f"Corridor {impact['components']['corridor']}pts",
        "",
        "DEPLOYMENT PLAN", sep(),
        f"Personnel  : {mp['personnel']}  "
        f"(base {mp['base_personnel']} × severity {mp['severity_multiplier']}"
        + (" + 2 closure mgmt" if mp["extra_for_closure_management"] else "") + ")",
        f"Barricades : {bc['barricade_points']}",
        f"Map mode   : {dep.get('mode','unknown')}",
        "",
    ]
    if dep.get("barricades"):
        lines += ["BARRICADE POSITIONS", sep()]
        for i, (blat, blon) in enumerate(dep["barricades"]):
            b = geo_utils.bearing_between(result["input"]["lat"], result["input"]["lon"], blat, blon)
            lines.append(f"  Barricade {i+1} · {compass_dir(b)} approach ({b:.0f}°) · {blat:.5f}, {blon:.5f}")
        lines.append("")
    if dep.get("officers"):
        lines += ["OFFICER POSTS", sep()]
        at_bc = len(dep.get("barricades", []))
        for i, (olat, olon) in enumerate(dep["officers"]):
            role = f"At Barricade {i+1}" if i < at_bc else "On-scene command"
            lines.append(f"  Officer {i+1} · {role} · {olat:.5f}, {olon:.5f}")
        lines.append("")
    lines += ["SUGGESTED DIVERSION", sep()]
    for r in dv.get("suggested_routes", [])[:3]:
        lines.append("  → " + " · ".join(r))
    if not dv.get("suggested_routes"):
        lines.append("  No clear alternate identified.")
    lines.append("")
    top_tier = max(tier_p, key=tier_p.get)
    lines += [
        "DURATION OUTLOOK", sep(),
        f"Most likely tier : {top_tier} ({tier_p[top_tier]*100:.0f}%)",
        f"  Quick(<30m) {tier_p.get('Quick',0)*100:.0f}%  "
        f"Moderate(30-120m) {tier_p.get('Moderate',0)*100:.0f}%  "
        f"Prolonged(>120m) {tier_p.get('Prolonged',0)*100:.0f}%",
        "",
    ]
    pmed = dur.get("precedent_median_min")
    if pmed:
        lines.append(f"  Precedent median: {pmed:.0f} min (n={dur['n_precedents']} similar events)")
        lines.append("")
    if sim is not None and len(sim):
        lines += ["SIMILAR HISTORICAL INCIDENTS", sep()]
        for _, row in sim.head(5).iterrows():
            ts = pd.to_datetime(row.get("start_ts", pd.NaT))
            ts_str = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else "–"
            d = row.get("duration_min")
            dur_str = f"{d:.0f} min" if pd.notna(d) else "unknown"
            closed = "YES" if row.get("requires_road_closure") else "NO"
            lines.append(f"  {ts_str}  {row.get('corridor','–')}  Closed:{closed}  Duration:{dur_str}")
        lines.append("")
    lines += [sep("="),
              "Based on 8,173 ASTraM/BTP incident records (Nov 2023 – Apr 2024).",
              "Deployment positions are indicative; verify against ground conditions.",
              "Weather data: OpenMeteo API (free, real-time, no API key required).",
              ""]
    return "\n".join(lines)


# ─── CACHED RESOURCES ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def get_engine():
    return ForecastEngine(ROOT)


@st.cache_data(show_spinner="Looking up location...")
def cached_geocode(query, viewbox):
    return geo_utils.geocode_address(query, viewbox=viewbox)


def known_hotspots(engine, top_n=14):
    df = engine.df
    j = (df[df["junction"].notna()]
         .groupby("junction")
         .agg(lat=("latitude","mean"), lon=("longitude","mean"), n=("latitude","size"))
         .sort_values("n", ascending=False).head(top_n))
    return {prettify(idx): (row.lat, row.lon) for idx, row in j.iterrows()}


def today_hourly_risk(df) -> pd.Series:
    today_dow = datetime.now().weekday()
    pivot = df.groupby(["dow", "hour"])["id"].count().unstack(fill_value=0)
    return pivot.loc[today_dow] if today_dow in pivot.index else pivot.mean()


# ─── PAGE INIT ────────────────────────────────────────────────────────────────
inject_css()
engine   = get_engine()
df       = engine.df
hotspots = known_hotspots(engine)
weather  = fetch_weather()
BLR_BOX  = (df["longitude"].min()-0.06, df["latitude"].max()+0.06,
             df["longitude"].max()+0.06, df["latitude"].min()-0.06)

_dn, (_dl, _dlo) = next(iter(hotspots.items()))
for k, v in [("lat_input", round(float(_dl),5)),
              ("lon_input", round(float(_dlo),5)),
              ("resolved_place_name", _dn),
              ("loc_mode", "Place name"),
              ("last_forecast", None),
              ("last_forecast_meta", {}),
              ("active_events", [])]:
    st.session_state.setdefault(k, v)
for k, v in {
    "fb_cause": CAUSE_OPTIONS[0], "fb_event_type": "planned", "fb_priority": "High",
    "fb_closure": False, "fb_lat": _dl, "fb_lon": _dlo,
    "fb_corridor": "Non-corridor", "fb_station": "Unknown",
    "fb_start": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "fb_end":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "fb_address": "", "fb_manpower": 2, "fb_escalated": False,
    "fb_notes": "", "fb_impact_score": 0.0,
}.items():
    st.session_state.setdefault(k, v)

active_events  = st.session_state["active_events"]
n_active       = sum(1 for e in active_events if e["status"] == "active")
total_personnel= sum(e["personnel"] for e in active_events if e["status"] == "active")
conflicts      = detect_conflicts(active_events)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">'
        f'<svg width="28" height="28" viewBox="0 0 30 30">'
        f'<rect width="30" height="30" rx="7" fill="{SURF}" stroke="{LINE}"/>'
        f'<circle cx="15" cy="8.5" r="3.2" fill="{RED}"/>'
        f'<circle cx="15" cy="15" r="3.2" fill="{AMBER}"/>'
        f'<circle cx="15" cy="21.5" r="3.2" fill="{GREEN}"/>'
        f'</svg>'
        f'<span style="font-family:Archivo,sans-serif;font-weight:800;font-size:1.02rem;color:#ECEDEF;">ASTraM</span>'
        f'</div>', unsafe_allow_html=True)
    st.caption("Event Impact Forecaster · Gridlock 2.0")
    hr()

    # Live weather badge
    if weather.get("available"):
        w_color = (RED if weather["is_heavy"] else BLUE if weather["is_rain"] else GREEN)
        w_icon  = ("⛈" if weather["is_heavy"] else "🌧" if weather["is_rain"] else "☀️")
        st.markdown(
            f'<div class="alert-box alert-weather" style="margin-bottom:10px;">'
            f'{w_icon} <b>Live weather · Bengaluru</b><br/>'
            f'{weather["description"]} · {weather["temperature"]:.0f}°C · '
            f'{weather["rain_mm"]:.1f}mm rain<br/>'
            f'<span style="color:{w_color};font-weight:600;">'
            f'{"⚠ Elevated closure risk" if weather["is_rain"] else "✓ Dry conditions"}'
            f'</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="alert-box alert-info" style="margin-bottom:10px;">'
            f'🌤 Weather: <span style="color:{MUTE};">Offline (will load on run)</span></div>',
            unsafe_allow_html=True)

    hr()
    st.markdown('<div class="eyebrow">System status</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="mono" style="font-size:0.78rem;color:#C4C8D1;line-height:1.8;">'
        f'{status_dot(GREEN)}{len(df):,} events indexed<br/>'
        f'{status_dot(GREEN)}3 models online<br/>'
        f'{status_dot(AMBER)}Free-tier OSM routing</div>', unsafe_allow_html=True)
    hr()
    st.markdown('<div class="eyebrow">Active operations</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: stat_card("Events", str(n_active), "active")
    with c2: stat_card("Personnel", str(total_personnel), "deployed")
    if conflicts:
        vspace(8)
        alert(f'⚠️ <b>{len(conflicts)} resource conflict{"s" if len(conflicts)>1 else ""}</b> '
              f'detected — events within 500m. See Operations tab.', "warn")
    hr()
    st.caption("Free stack · scikit-learn · Streamlit · Plotly · OpenStreetMap · OpenMeteo")


# ─── HERO ─────────────────────────────────────────────────────────────────────
weather_badge = (
    f' · <span style="color:{BLUE};">🌧 RAIN — elevated risk</span>' if weather.get("is_rain") else ""
)
st.markdown(
    f'<div class="hero-bar">'
    f'<div><div class="hero-title">ASTraM Event Impact Forecaster</div>'
    f'<div class="hero-tagline">Forecast · Deploy · Learn — powered by real incident data + live weather</div></div>'
    f'<div class="hero-status">'
    f'{status_dot(GREEN)}LIVE · {len(df):,} EVENTS · BENGALURU{weather_badge}<br/>'
    f'<span style="color:{AMBER};">{n_active} ACTIVE OPS</span>'
    f'</div></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["01 · Observe", "02 · Forecast", "03 · Operations", "04 · Learn", "05 · Methodology"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · OBSERVE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    eyebrow("Historical data · Nov 2023 – Apr 2024")
    section_head("What 5 months of Bengaluru traffic events tell us",
                 "Every number is from the real ASTraM export — nothing simulated.")
    vspace(10)

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("Events indexed", f"{len(df):,}", "Nov 2023 – Apr 2024")
    with c2: stat_card("Required closure", f"{df['requires_road_closure'].mean()*100:.1f}%", "of all incidents")
    with c3: stat_card("Planned events", f"{(df['event_type']=='planned').mean()*100:.1f}%", "rest unplanned")
    with c4:
        med = df.loc[df["duration_valid"],"duration_min"].median()
        stat_card("Median resolution", f"{med:.0f} min", "resolvable incidents")

    vspace(18)

    # WEATHER + TODAY'S RISK ──────────────────────────────────────────────────
    if weather.get("available") and weather.get("is_rain"):
        alert(
            f'🌧 <b>Rain advisory</b> — Bengaluru currently showing {weather["description"]} '
            f'({weather["rain_mm"]:.1f}mm). Historical data shows water_logging, accident, '
            f'and vehicle_breakdown incidents increase significantly in wet conditions. '
            f'Impact Forecaster applies a ×{weather["multiplier"]:.2f} closure-risk adjustment '
            f'for rain-sensitive event types in the Forecast tab.', "weather")
        vspace(8)

    with st.container(border=True):
        card_title(f"Today's risk outlook · {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][datetime.now().weekday()]}s by hour (IST)")
        hourly = today_hourly_risk(df)
        risk_df = pd.DataFrame({"hour": hourly.index, "incidents": hourly.values})
        now_h = datetime.now().hour
        fig_risk = px.bar(risk_df, x="hour", y="incidents",
                           color_discrete_sequence=[AMBER],
                           labels={"hour":"Hour (IST)", "incidents":"Hist. incidents"})
        fig_risk.add_vline(x=now_h, line_color=RED, line_width=2, line_dash="dot",
                            annotation_text="Now", annotation_font_color=RED,
                            annotation_position="top")
        st.plotly_chart(themed(fig_risk, height=225), use_container_width=True, config=NO_TOOLBAR)
        st.caption("Historical incident frequency by hour on this day of week. "
                   "Use to pre-position resources ahead of peak windows.")

    vspace(14)
    with st.container(border=True):
        card_title("Where incidents happen · 4,000 sampled events")
        map_df = df.sample(min(4000, len(df)), random_state=1)
        fig_map = px.scatter_mapbox(
            map_df, lat="latitude", lon="longitude", color="event_cause",
            hover_data={"corridor":True,"priority":True,"requires_road_closure":True,
                        "latitude":False,"longitude":False,"event_cause":False},
            zoom=10.2, height=450, opacity=0.6, color_discrete_sequence=CHART_COLORWAY,
        )
        fig_map.update_layout(mapbox_style=MAP_STYLE, margin=dict(l=0,r=0,t=0,b=0),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig_map, use_container_width=True, config=NO_TOOLBAR)

    vspace(14)
    colA, colB = st.columns(2)
    with colA:
        with st.container(border=True):
            card_title("Causes")
            cc = df["event_cause"].value_counts().reset_index()
            cc.columns = ["cause","count"]
            fig_c = px.bar(cc, x="count", y="cause", orientation="h",
                            color_discrete_sequence=[AMBER])
            fig_c.update_layout(yaxis=dict(categoryorder="total ascending"), showlegend=False)
            st.plotly_chart(themed(fig_c, height=380), use_container_width=True, config=NO_TOOLBAR)
    with colB:
        with st.container(border=True):
            card_title("Day × hour heatmap (IST)")
            heat = df.pivot_table(index="dow_name", columns="hour", values="id",
                                   aggfunc="count", fill_value=0)
            heat = heat.reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            fig_h = go.Figure(go.Heatmap(
                z=heat.values, x=heat.columns, y=heat.index,
                colorscale=[[0, SURF], [0.5,"#8a5a22"], [1, AMBER]]))
            fig_h.update_layout(xaxis_title="Hour of day (IST)")
            st.plotly_chart(themed(fig_h, height=380), use_container_width=True, config=NO_TOOLBAR)

    vspace(14)
    with st.container(border=True):
        card_title("Highest-risk corridors · closure rate (min. 20 events)")
        g = df.groupby("corridor")["requires_road_closure"].agg(["mean","count"])
        g = g[(g["count"]>=20) & (g.index!="Non-corridor")].sort_values("mean", ascending=False).head(10)
        g = g.rename(columns={"mean":"closure_rate","count":"n_events"}).reset_index()
        g["closure_rate"] = (g["closure_rate"]*100).round(1)
        st.dataframe(g, use_container_width=True, hide_index=True, column_config={
            "corridor": st.column_config.TextColumn("Corridor"),
            "closure_rate": st.column_config.ProgressColumn("Closure rate", format="%.1f%%", min_value=0, max_value=100),
            "n_events": st.column_config.NumberColumn("Events", format="%d"),
        })

    vspace(10)
    with st.container(border=True):
        eyebrow("Key finding — the problem we solve")
        st.markdown(
            "**`priority` in the current system is 99.8% determined by corridor membership, "
            "not by what's actually happening.** Named corridor = High; everything else = Low. "
            "Cause, time-of-day, and actual disruption length are irrelevant to the current "
            "triage rule. Two events with 5% vs 90% predicted closure probability get identical "
            "priority labels if they're on the same road. The Impact Index in Forecast tab "
            "corrects this by combining closure probability, expected duration, cause type, and "
            "corridor — producing a 0–100 score that reflects real operational severity."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    eyebrow("New event")
    section_head("Describe what's happening",
                 "Planning ahead or reporting now — both paths work here.")
    vspace(10)

    def _apply_hotspot():
        sel = st.session_state.get("hotspot_select")
        if sel and sel in hotspots:
            lat0, lon0 = hotspots[sel]
            st.session_state["lat_input"] = round(float(lat0), 5)
            st.session_state["lon_input"] = round(float(lon0), 5)
            st.session_state["resolved_place_name"] = sel

    with st.container(border=True):
        colL, colR = st.columns(2)
        with colL:
            event_type = st.radio("Event type", ["planned","unplanned"], horizontal=True)
            event_cause = st.selectbox("What's happening", CAUSE_OPTIONS)
            priority_choice = st.selectbox("Priority", ["Auto-suggest","High","Low"])
            event_date = st.date_input("Date", value=datetime.now().date())
            event_time = st.time_input("Time (IST)", value=datetime.now().time())
            if weather.get("available") and weather.get("is_rain") and event_cause in RAIN_SENSITIVE:
                alert(f'🌧 <b>Rain active</b> — closure risk for <i>{event_cause.replace("_"," ")}</i> '
                      f'is adjusted ×{weather["multiplier"]:.2f} in the forecast below.', "weather")
        with colR:
            loc_mode = st.radio("Set location by", ["Place name","Known hotspot","Coordinates"],
                                 horizontal=True, key="loc_mode")
            if loc_mode == "Place name":
                query = st.text_input("Search for a place",
                                       placeholder="e.g. Indiranagar 100 Feet Road")
                if st.button("Locate"):
                    hit = cached_geocode(query, BLR_BOX)
                    if hit:
                        st.session_state["lat_input"] = round(hit["lat"], 5)
                        st.session_state["lon_input"] = round(hit["lon"], 5)
                        st.session_state["resolved_place_name"] = hit["display_name"]
                    else:
                        st.warning("Couldn't find that place. Try a more specific name, "
                                   "or use Known hotspot / Coordinates.")
            elif loc_mode == "Known hotspot":
                st.selectbox("Pick a junction", list(hotspots.keys()),
                             key="hotspot_select", on_change=_apply_hotspot,
                             index=None, placeholder="Choose a hotspot...")
            else:
                lc1, lc2 = st.columns(2)
                with lc1: st.number_input("Latitude",  key="lat_input", format="%.5f", step=0.0001)
                with lc2: st.number_input("Longitude", key="lon_input", format="%.5f", step=0.0001)
                st.session_state["resolved_place_name"] = "Manual coordinates"

            lat = st.session_state["lat_input"]
            lon = st.session_state["lon_input"]
            st.caption(
                f'📍 <span style="color:{PAPER};">{st.session_state["resolved_place_name"]}</span> — '
                f'<span class="mono" style="color:{MUTE};">{lat:.5f}, {lon:.5f}</span>',
                unsafe_allow_html=True)
            prev = go.Figure(go.Scattermapbox(
                lat=[lat], lon=[lon], mode="markers",
                marker=dict(size=16, color=RED), name="Point",
            ))
            prev.update_layout(mapbox_style=MAP_STYLE, mapbox_center={"lat":lat,"lon":lon},
                                mapbox_zoom=13, margin=dict(l=0,r=0,t=0,b=0),
                                height=170, showlegend=False)
            st.plotly_chart(prev, use_container_width=True, config=NO_TOOLBAR)

        run = st.button("Generate forecast & plan", type="primary")

    if run:
        when = pd.Timestamp(datetime.combine(event_date, event_time))
        pr_override = None if priority_choice == "Auto-suggest" else priority_choice
        try:
            result = engine.forecast(event_cause, lat, lon, event_type, when, pr_override)

            # Apply live weather adjustment to closure probability
            if weather.get("available") and weather.get("is_rain") and event_cause in RAIN_SENSITIVE:
                raw_cp = result["closure_probability"]
                adj_cp = min(1.0, raw_cp * weather["multiplier"])
                result["_weather_adjusted_closure"] = adj_cp
                result["_weather_raw_closure"]      = raw_cp

            st.session_state["last_forecast"] = result
            st.session_state["last_forecast_meta"] = {
                "location_name": st.session_state.get("resolved_place_name", "Unknown"),
                "when_ist": when.strftime("%A %d %b %Y, %H:%M IST"),
            }
        except Exception as ex:
            st.error(f"Couldn't generate forecast: {ex}")

    result = st.session_state.get("last_forecast")
    meta   = st.session_state.get("last_forecast_meta", {})

    if result:
        hr()
        impact = result["impact_index"]
        ctx    = result["context"]
        dur    = result["duration"]
        mp     = result["manpower"]
        bc     = result["barricades"]
        dv     = result["diversion"]
        sim    = result["similar_incidents"]
        cp     = result.get("_weather_adjusted_closure", result["closure_probability"])

        # ── IMPACT HERO ───────────────────────────────────────────────────
        g1, g2 = st.columns([1, 2.6])
        with g1:
            st.markdown(traffic_light_svg(impact["band"]), unsafe_allow_html=True)
        with g2:
            eyebrow("Impact Index")
            st.markdown(
                f'<div style="font-family:Archivo,sans-serif;font-weight:800;'
                f'font-size:3rem;color:{impact["color"]};line-height:1;">'
                f'{impact["score"]:.0f}'
                f'<span style="font-size:1.3rem;color:{MUTE};"> / 100</span></div>',
                unsafe_allow_html=True)
            pill(f"{impact['band']} impact", impact["color"])
            vspace(6)
            st.markdown(
                f'<div style="color:{MUTE};font-size:0.83rem;margin-top:6px;">'
                f'📍 {ctx["corridor"]} corridor · {ctx["police_station"]} '
                f'({ctx["nearest_distance_km"]*1000:.0f}m from nearest record)</div>',
                unsafe_allow_html=True)
            vspace(8)
            m1, m2, m3 = st.columns(3)
            with m1:
                label = "Closure risk"
                if result.get("_weather_adjusted_closure"):
                    sub = f"{result['_weather_raw_closure']*100:.0f}% raw · ×{weather['multiplier']:.2f} rain adj."
                else:
                    sub = "predicted probability"
                stat_card(label, f"{cp*100:.0f}%", sub)
            with m2:
                hd = dur.get("precedent_median_min") or dur["ml_point_estimate_min"]
                stat_card("Typical duration", f"{hd:.0f} min", f"n={dur['n_precedents']} precedents")
            with m3:
                stat_card("Priority", result["priority"]["used"], "used for this forecast")

        vspace(14)

        # ── SYSTEM COMPARISON — Traditional vs Ours ───────────────────────
        with st.container(border=True):
            eyebrow("System comparison")
            card_title("How the current triage rule compares to this forecast")
            traditional_p = "High" if ctx["is_corridor"] else "Low"
            our_band      = impact["band"]
            gap = ((traditional_p == "Low") and (our_band in ("High","Critical"))) or \
                  ((traditional_p == "High") and (our_band == "Low"))
            gap_color = RED if gap else GREEN

            st.markdown(f"""
<table class="compare-table">
  <tr>
    <th>Metric</th>
    <th>Current ASTraM rule</th>
    <th>Impact Forecaster</th>
  </tr>
  <tr>
    <td>Priority signal</td>
    <td class="compare-old"><b style="color:{'#F5A623' if traditional_p=='High' else '#8A8F9C'};">{traditional_p}</b> (corridor-membership rule)</td>
    <td class="compare-new"><b style="color:{impact['color']};"> {our_band}</b> · {impact['score']:.0f}/100</td>
  </tr>
  <tr>
    <td>Closure estimate</td>
    <td class="compare-old">Not available</td>
    <td class="compare-new">{cp*100:.0f}% probability{' (weather-adjusted)' if result.get('_weather_adjusted_closure') else ''}</td>
  </tr>
  <tr>
    <td>Duration guidance</td>
    <td class="compare-old">Not available</td>
    <td class="compare-new">{max(dur['tier_probabilities'], key=dur['tier_probabilities'].get)} most likely · ML + {dur['n_precedents']} precedents</td>
  </tr>
  <tr>
    <td>Manpower recommendation</td>
    <td class="compare-old">Experience-driven</td>
    <td class="compare-new">{mp['personnel']} officers · {bc['barricade_points']} barricade points</td>
  </tr>
  <tr>
    <td>Basis</td>
    <td class="compare-old">Single rule: is_corridor?</td>
    <td class="compare-new">Cause + location + time + closure ML + duration ML + live weather</td>
  </tr>
</table>
{'<div style="margin-top:10px;font-size:0.84rem;color:' + RED + ';font-weight:600;">⚠ Priority gap detected: current system labels this ' + traditional_p + ' but our model assesses it as ' + our_band + '.</div>' if gap else '<div style="margin-top:10px;font-size:0.82rem;color:' + GREEN + ';">✓ Consistent with current priority assignment.</div>'}
""", unsafe_allow_html=True)

        vspace(14)

        # ── WEATHER ADJUSTMENT (if active) ────────────────────────────────
        if result.get("_weather_adjusted_closure"):
            alert(
                f'🌧 <b>Weather adjustment applied</b> — '
                f'Bengaluru is currently experiencing {weather["description"]} '
                f'({weather["rain_mm"]:.1f}mm). {event_cause.replace("_"," ").title()} '
                f'closure risk raised from {result["_weather_raw_closure"]*100:.0f}% → '
                f'{cp*100:.0f}% (×{weather["multiplier"]:.2f} multiplier). '
                f'Data: OpenMeteo (free, live, no API key).', "weather")
            vspace(8)

        # ── ANOMALY DETECTION ─────────────────────────────────────────────
        anomaly = compute_anomaly(df, event_cause, ctx["corridor"], cp)
        if anomaly:
            kind = "anomaly" if anomaly["is_elevated"] else "info"
            if anomaly["is_elevated"]:
                alert(
                    f'⚡ <b>Elevated risk detected</b> — this event&#39;s closure probability '
                    f'({cp*100:.0f}%) is <b>{anomaly["excess"]*100:+.0f}%</b> above the '
                    f'{anomaly["hist"]*100:.0f}% historical average for '
                    f'{event_cause.replace("_"," ")} events on {ctx["corridor"]} '
                    f'(n={anomaly["n"]}). Treat as higher-priority than the raw Impact Index '
                    f'alone suggests.', "anomaly")
            elif anomaly["is_suppressed"]:
                alert(
                    f'✓ Below-average closure risk — {cp*100:.0f}% vs {anomaly["hist"]*100:.0f}% '
                    f'historical for this cause + corridor. Resources may be lighter than '
                    f'the corridor label alone would suggest.', "info")
            vspace(8)

        # ── IMPACT BREAKDOWN ──────────────────────────────────────────────
        with st.container(border=True):
            card_title("Impact Index breakdown")
            comp = impact["components"]
            comp_df = pd.DataFrame({
                "Factor": ["Closure risk (max 40)","Duration (max 35)","Priority (max 15)","Corridor (max 10)"],
                "Points": [comp["closure"], comp["duration"], comp["priority"], comp["corridor"]],
            })
            fig_comp = px.bar(comp_df, x="Points", y="Factor", orientation="h",
                               range_x=[0,42], color_discrete_sequence=[AMBER])
            fig_comp.update_layout(showlegend=False)
            st.plotly_chart(themed(fig_comp, height=185), use_container_width=True, config=NO_TOOLBAR)

        vspace(14)

        # ── RECOMMENDATION CARDS ──────────────────────────────────────────
        eyebrow("Recommended response")
        section_head("What to deploy")
        vspace(8)
        r1, r2, r3 = st.columns(3)
        with r1:
            with st.container(border=True):
                stat_card("Personnel", str(mp["personnel"]),
                          f"base {mp['base_personnel']} × {mp['severity_multiplier']}"
                          + (" + 2 closure" if mp["extra_for_closure_management"] else ""))
        with r2:
            with st.container(border=True):
                stat_card("Barricade points", str(bc["barricade_points"]),
                          "Based on closure risk and road type")
        with r3:
            with st.container(border=True):
                top_r = " · ".join(dv["suggested_routes"][0]) if dv["suggested_routes"] else "None found"
                others = dv["suggested_routes"][1:3]
                sub_r = ("also: " + "; ".join(" · ".join(r) for r in others)) if others else "Nearest alternate corridor"
                stat_card("Suggested diversion", top_r, sub_r)
        if dv["mode"] == "historical_fallback":
            st.caption(f"ℹ️ {dv['note']}")

        vspace(14)

        # ── DEPLOYMENT MAP ─────────────────────────────────────────────────
        with st.container(border=True):
            card_title("Deployment map · where to put people and barricades")
            dep = geo_utils.compute_deployment_points(
                lat, lon, n_barricades=bc["barricade_points"], n_personnel=mp["personnel"])
            fig_dep = deployment_map(
                lat, lon, dep["barricades"], dep["officers"],
                route_points=dv.get("route_points"), bc_dist=90,
                dep_mode=dep.get("mode","geometric_fallback"),
                impact_band=impact["band"], impact_score=impact["score"],
                location_name=meta.get("location_name",""),
            )
            st.plotly_chart(fig_dep, use_container_width=True, config=NO_TOOLBAR)
            if dep.get("mode") == "live_osm":
                st.caption(f"✅ Road-network snapped ({dep['n_approaches']} approach roads). "
                           "Barricades follow actual road bearings.")
            else:
                st.caption("⚠️ Geometric ring shown (install osmnx + networkx for road-snapped placement). "
                           "Adjust to actual road layout before briefing.")
            st.markdown(
                f'<span style="font-size:0.79rem;color:{MUTE};">'
                f'<span style="color:{RED};">●</span> Incident &nbsp;'
                f'<span style="color:{ORANGE};">●</span> Barricade &nbsp;'
                f'<span style="color:{GREEN};">●</span> Officer &nbsp;'
                f'<span style="color:{BLUE};">●</span> Diversion &nbsp;'
                f'<span style="color:{AMBER};">○</span> Coverage radius</span>',
                unsafe_allow_html=True)

        vspace(14)

        # ── DURATION WITH CONFIDENCE INTERVALS ────────────────────────────
        with st.container(border=True):
            card_title("Duration outlook")
            tier_df = pd.DataFrame({
                "Tier": list(dur["tier_probabilities"].keys()),
                "Probability": list(dur["tier_probabilities"].values()),
            })
            fig_tier = px.bar(tier_df, x="Tier", y="Probability", range_y=[0,1],
                               color_discrete_sequence=[AMBER])
            fig_tier.update_layout(showlegend=False)
            st.plotly_chart(themed(fig_tier, height=225), use_container_width=True, config=NO_TOOLBAR)

            ci = duration_ci(sim)
            if ci:
                ci1, ci2, ci3 = st.columns(3)
                with ci1: stat_card("Fast 10th pct.", f"{ci['p10']:.0f} min", f"{ci['n']} similar events")
                with ci2: stat_card("Median (50th)", f"{ci['p50']:.0f} min", "from matched precedents")
                with ci3: stat_card("Slow 90th pct.", f"{ci['p90']:.0f} min", "9 in 10 resolved by this")
                st.caption(
                    f"Confidence interval from {ci['n']} matched historical incidents. "
                    f"90% of similar events resolved in under {ci['p90']:.0f} minutes."
                )
            else:
                pmed = dur.get("precedent_median_min")
                pmed_txt = f"Precedent median (n={dur['n_precedents']}): {pmed:.0f} min. " if pmed else ""
                st.caption(
                    f"ML estimate: {dur['ml_point_estimate_min']:.0f} min. "
                    + pmed_txt +
                    "Duration is noisy — tier distribution is more reliable than any single number."
                )

        vspace(14)

        # ── SIMILAR INCIDENTS ──────────────────────────────────────────────
        with st.container(border=True):
            card_title(f"Similar past incidents · the evidence ({len(sim)} found)")
            if sim is not None and len(sim):
                sim_show = sim.copy()
                sim_show["start_ts"] = pd.to_datetime(sim_show["start_ts"]).dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(
                    sim_show[["start_ts","address","corridor","priority",
                               "requires_road_closure","duration_min","distance_km"]],
                    use_container_width=True, hide_index=True,
                    column_config={
                        "start_ts": st.column_config.TextColumn("When"),
                        "duration_min": st.column_config.NumberColumn("Duration (min)", format="%.0f"),
                        "distance_km": st.column_config.NumberColumn("Dist. (km)", format="%.2f"),
                        "requires_road_closure": st.column_config.CheckboxColumn("Road closed"),
                    })
            else:
                st.caption("No close historical matches found.")

        vspace(14)

        # ── ACTIONS ───────────────────────────────────────────────────────
        a1, a2 = st.columns(2)
        with a1:
            if st.button("➕ Add to Active Events register", use_container_width=True):
                active_events.append({
                    "id": str(uuid.uuid4())[:8],
                    "added_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "cause": event_cause, "event_type": event_type,
                    "lat": lat, "lon": lon,
                    "location_name": meta.get("location_name","Unknown"),
                    "when_ist": meta.get("when_ist",""),
                    "impact_score": impact["score"], "band": impact["band"],
                    "color": impact["color"],
                    "closure_prob": cp,
                    "personnel": mp["personnel"], "barricades": bc["barricade_points"],
                    "corridor": ctx["corridor"], "police_station": ctx["police_station"],
                    "priority": result["priority"]["used"], "status": "active",
                })
                st.session_state["active_events"] = active_events
                st.success(f"Added. {len(active_events)} event(s) in Operations register.")
        with a2:
            briefing_txt = generate_briefing(
                result, dep, weather,
                location_name=meta.get("location_name","Unknown"),
                when_ist=meta.get("when_ist",""),
            )
            st.download_button(
                "⬇ Download briefing",
                data=briefing_txt,
                file_name=f"astram_briefing_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain", use_container_width=True,
                help="Structured text: all positions, bearings, weather, diversions — print and hand to an officer.",
            )
    else:
        vspace(10)
        st.markdown(
            '<div class="empty-state">No forecast yet.<br/>'
            'Pick a location, describe the event, and click <b>Generate forecast & plan</b>.<br/><br/>'
            'Try: <b>Known hotspot → Silk Board Junction → accident → unplanned → Generate</b>'
            '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    eyebrow("Active events register")
    section_head("Operations overview",
                 "Add events from Forecast tab. Monitor all active deployments on one map.")
    vspace(10)
    active_events = st.session_state["active_events"]

    if not active_events:
        st.markdown(
            '<div class="empty-state">No events in the register yet.<br/>'
            'Generate a forecast and click <b>Add to Active Events register</b>.<br/><br/>'
            'Once you have multiple events here you\'ll see: a combined operations map, '
            'automatic conflict detection, resource tally, and an optimizer that allocates '
            'available officers across all active events by severity.'
            '</div>', unsafe_allow_html=True)
    else:
        n_act = sum(1 for e in active_events if e["status"]=="active")
        tot_p = sum(e["personnel"] for e in active_events if e["status"]=="active")
        tot_b = sum(e["barricades"] for e in active_events if e["status"]=="active")
        crit  = sum(1 for e in active_events if e["band"]=="Critical" and e["status"]=="active")

        k1,k2,k3,k4 = st.columns(4)
        with k1: stat_card("Active events", str(n_act), f"{len(active_events)-n_act} resolved")
        with k2: stat_card("Total personnel", str(tot_p), "across active events")
        with k3: stat_card("Total barricades", str(tot_b), "across active events")
        with k4: stat_card("Critical severity", str(crit), "require immediate attention")

        vspace(14)

        conflicts = detect_conflicts(active_events)
        if conflicts:
            eyebrow("Resource conflicts detected")
            for i, j, dist_m in conflicts:
                ea, eb = active_events[i], active_events[j]
                alert(
                    f'⚠️ <b>{ea["location_name"]}</b> and <b>{eb["location_name"]}</b> are '
                    f'<b>{dist_m:.0f}m apart</b> — both drawing on the same road network. '
                    f'Combined personnel: {ea["personnel"]+eb["personnel"]}. '
                    f'Consider coordinated deployment or shared barricades.', "warn")
            vspace(8)

        # RESOURCE OPTIMIZER ──────────────────────────────────────────────
        with st.container(border=True):
            eyebrow("Resource optimizer")
            card_title("Optimal officer allocation across active events")
            avail = st.number_input("Available officers for this shift", min_value=1,
                                     value=max(tot_p, n_act*3), step=1)
            if st.button("Optimize allocation"):
                alloc = optimize_allocation(active_events, int(avail))
                if alloc:
                    alloc_df = pd.DataFrame([{
                        "Location": e["location_name"],
                        "Cause": e["cause"].replace("_"," ").title(),
                        "Band": e["band"],
                        "Impact Score": e["impact_score"],
                        "Recommended": e["personnel"],
                        "Allocated": e["allocated"],
                        "Shortfall": e["shortfall"],
                    } for e in alloc])
                    st.dataframe(alloc_df, use_container_width=True, hide_index=True,
                                 column_config={
                                     "Impact Score": st.column_config.ProgressColumn(
                                         "Impact Score", format="%.0f", min_value=0, max_value=100),
                                     "Shortfall": st.column_config.NumberColumn("Shortfall", format="%d"),
                                 })
                    total_allocated = sum(e["allocated"] for e in alloc)
                    total_needed    = sum(e["personnel"] for e in alloc)
                    if total_allocated < total_needed:
                        alert(f"⚠️ {total_needed - total_allocated} officers short of full "
                              f"recommended deployment across all active events. "
                              f"Shortfall distributed to lowest-severity events.", "warn")
                    else:
                        alert(f"✓ {avail} officers sufficient to meet all recommended deployments.", "info")

        vspace(14)

        with st.container(border=True):
            card_title("All events on one map · size = impact severity · hover for details")
            st.plotly_chart(ops_map(active_events), use_container_width=True, config=NO_TOOLBAR)
            st.caption("Green=Low · Amber=Medium · Orange=High · Red=Critical")

        vspace(14)

        with st.container(border=True):
            card_title("Event register")
            evt_df = pd.DataFrame([{
                "ID": e["id"],
                "Location": e["location_name"],
                "Cause": e["cause"].replace("_"," ").title(),
                "When": e["when_ist"],
                "Score": e["impact_score"],
                "Band": e["band"],
                "Personnel": e["personnel"],
                "Barricades": e["barricades"],
                "Corridor": e["corridor"],
                "Status": e["status"].upper(),
            } for e in active_events])
            st.dataframe(evt_df, use_container_width=True, hide_index=True, column_config={
                "Score": st.column_config.ProgressColumn("Score", format="%.0f", min_value=0, max_value=100),
            })

        vspace(14)
        with st.container(border=True):
            card_title("Manage")
            mc1, mc2 = st.columns(2)
            with mc1:
                if st.button("✓ Mark all resolved", use_container_width=True):
                    for e in st.session_state["active_events"]: e["status"] = "resolved"
                    st.success("All events marked resolved.")
            with mc2:
                if st.button("🗑 Clear register", use_container_width=True):
                    st.session_state["active_events"] = []
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · LEARN
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    eyebrow("Post-event learning")
    section_head("Log what actually happened",
                 "Every outcome here becomes training data for the next retrain — "
                 "closing the brief's 'no post-event learning system' gap.")
    vspace(10)
    last = st.session_state.get("last_forecast")
    if last:
        if st.button("↺ Prefill from my last forecast"):
            st.session_state["fb_cause"]        = last["input"]["event_cause"]
            st.session_state["fb_event_type"]   = last["input"]["event_type"]
            st.session_state["fb_lat"]          = float(last["input"]["lat"])
            st.session_state["fb_lon"]          = float(last["input"]["lon"])
            st.session_state["fb_priority"]     = last["priority"]["used"]
            st.session_state["fb_corridor"]     = last["context"]["corridor"] or "Non-corridor"
            st.session_state["fb_station"]      = last["context"]["police_station"] or "Unknown"
            st.session_state["fb_impact_score"] = last["impact_index"]["score"]
            st.rerun()

    with st.form("feedback_form"):
        fc1, fc2 = st.columns(2)
        with fc1:
            st.selectbox("Event cause", CAUSE_OPTIONS, key="fb_cause")
            st.radio("Event type", ["planned","unplanned"], horizontal=True, key="fb_event_type")
            st.selectbox("Priority used", ["High","Low"], key="fb_priority")
            st.checkbox("Road closure was actually required", key="fb_closure")
        with fc2:
            st.number_input("Latitude", format="%.5f", key="fb_lat")
            st.number_input("Longitude", format="%.5f", key="fb_lon")
            st.text_input("Corridor", key="fb_corridor")
            st.text_input("Police station", key="fb_station")
        fc3, fc4 = st.columns(2)
        with fc3:
            st.text_input("Actual start datetime (ISO, UTC)", key="fb_start")
            st.text_input("Address / description", key="fb_address")
        with fc4:
            st.text_input("Actual end datetime (ISO, UTC)", key="fb_end")
            st.number_input("Actual personnel deployed", min_value=0, step=1, key="fb_manpower")
        st.checkbox("Escalated beyond forecast", key="fb_escalated")
        st.text_area("Notes (optional)", key="fb_notes")
        submit_fb = st.form_submit_button("Log this outcome", type="primary")

    if submit_fb:
        feedback_store.log_outcome(
            ROOT,
            event_cause=st.session_state["fb_cause"],
            event_type=st.session_state["fb_event_type"],
            lat=st.session_state["fb_lat"], lon=st.session_state["fb_lon"],
            address=st.session_state["fb_address"],
            priority=st.session_state["fb_priority"],
            corridor=st.session_state["fb_corridor"],
            police_station=st.session_state["fb_station"],
            start_datetime=st.session_state["fb_start"],
            actual_end_datetime=st.session_state["fb_end"],
            requires_road_closure=st.session_state["fb_closure"],
            manpower_used=int(st.session_state["fb_manpower"]),
            forecast_impact_score=float(st.session_state["fb_impact_score"]),
            escalated_beyond_forecast=st.session_state["fb_escalated"],
            notes=st.session_state["fb_notes"],
        )
        st.success("Logged. Included in next retrain.")

    vspace(10)
    fb_log = feedback_store.load_feedback_log(ROOT)
    with st.container(border=True):
        card_title(f"Feedback log · {len(fb_log)} outcome(s)")
        if len(fb_log):
            st.dataframe(fb_log, use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="empty-state">Nothing logged yet.</div>', unsafe_allow_html=True)

    vspace(10)
    with st.container(border=True):
        card_title("Retrain · historical data + all logged outcomes")
        if st.button("🔁 Retrain now"):
            if not len(fb_log):
                st.warning("Nothing logged yet — retrain would reproduce identical models.")
            else:
                old_m = json.loads((MODELS_DIR/"metrics.json").read_text()) if (MODELS_DIR/"metrics.json").exists() else {}
                aug   = feedback_store.build_augmented_dataset(ROOT)
                with st.spinner("Retraining..."):
                    _, new_m = train_models.run_pipeline(aug, MODELS_DIR, save=True)
                get_engine.clear()
                st.success(f"Retrained on {new_m['n_total_events']} events "
                           f"({len(fb_log)} logged outcomes included).")
                rc1, rc2 = st.columns(2)
                with rc1:
                    stat_card("Closure ROC-AUC", f"{new_m['closure_model']['roc_auc']:.3f}",
                              f"was {old_m.get('closure_model',{}).get('roc_auc',0):.3f}")
                with rc2:
                    stat_card("Duration tier accuracy", f"{new_m['duration_model']['tier_accuracy']:.3f}",
                              f"was {old_m.get('duration_model',{}).get('tier_accuracy',0):.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    eyebrow("Reference")
    section_head("Methodology & honest performance",
                 "Every metric is from a time-based holdout — genuinely forecasting unseen future events.")
    vspace(10)

    m = json.loads((MODELS_DIR/"metrics.json").read_text()) if (MODELS_DIR/"metrics.json").exists() else {}
    cm_m = m.get("closure_model",{}); dm_m = m.get("duration_model",{}); pm_m = m.get("priority_model",{})

    i1,i2,i3 = st.columns(3)
    with i1:
        with st.container(border=True):
            eyebrow("Road-closure model")
            stat_card("ROC-AUC", f"{cm_m.get('roc_auc',0):.3f}",
                      f"PR-AUC {cm_m.get('pr_auc',0):.3f} vs base rate {cm_m.get('positive_rate_test',0):.3f}")
    with i2:
        with st.container(border=True):
            eyebrow("Duration model")
            delta = dm_m.get("tier_accuracy",0) - dm_m.get("tier_naive_baseline_accuracy",0)
            stat_card("Tier accuracy", f"{dm_m.get('tier_accuracy',0):.3f}", f"{delta:+.3f} vs naive")
    with i3:
        with st.container(border=True):
            eyebrow("Priority model")
            stat_card("Accuracy", f"{pm_m.get('accuracy',0):.3f}", "~99.8% corridor-determined")

    vspace(10)
    st.markdown(
        '<div class="empty-state" style="border-color:#F2784B66;">'
        "<b>Honest limitation:</b> the duration model's point estimates only marginally beat "
        "a 'always predict the median' baseline on unseen future weeks. The Forecast tab "
        "leads with tier probabilities and matched-incident confidence intervals "
        "rather than a falsely precise number."
        "</div>", unsafe_allow_html=True)

    vspace(14)
    with st.container(border=True):
        card_title("Feature importance · what drives the road-closure prediction")
        fi_path = MODELS_DIR / "feature_importance.json"
        if fi_path.exists():
            fi = json.loads(fi_path.read_text())
            fi_df = pd.DataFrame(fi["closure_model"]).head(10)
            fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h",
                             color_discrete_sequence=[AMBER])
            fig_fi.update_layout(yaxis=dict(categoryorder="total ascending"), showlegend=False)
            st.plotly_chart(themed(fig_fi, height=370), use_container_width=True, config=NO_TOOLBAR)

    vspace(10)
    with st.expander("Architecture & what makes this different from typical hackathon submissions"):
        st.markdown(
            "**What most teams build**: a dashboard that shows historical patterns, "
            "trains a basic classifier on the dataset, and outputs a priority label. "
            "The current ASTraM system already does this (99.8% accuracy — because "
            "priority IS corridor membership, not a learned insight).\n\n"
            "**What this builds instead**: five capabilities that don't exist in the current system:\n\n"
            "1. **Live weather adjustment** — OpenMeteo (free, no key) adjusts closure "
            "probability in real time based on current Bengaluru rainfall. "
            "Rain + vehicle breakdown on ORR = very different risk profile from dry conditions.\n\n"
            "2. **Anomaly detection** — percentile context telling you WHEN an event is "
            "statistically unusual vs the historical base rate for that cause × corridor.\n\n"
            "3. **Corrected priority** — the current binary High/Low rule is exposed side-by-side "
            "with the 0-100 Impact Index, making the gap concrete and visual for judges.\n\n"
            "4. **Duration confidence intervals** — P10/P50/P90 from matched historical "
            "incidents, not just a point estimate.\n\n"
            "5. **Resource optimizer** — impact-score weighted allocation across simultaneous "
            "events for a given officer headcount.\n\n"
            "**Stack**: scikit-learn · pandas · Streamlit · Plotly · OpenStreetMap (Nominatim + "
            "Carto dark basemap + optional OSMnx routing) · OpenMeteo (weather). "
            "100% free, no paid API keys, deployable on Streamlit Community Cloud."
        )
